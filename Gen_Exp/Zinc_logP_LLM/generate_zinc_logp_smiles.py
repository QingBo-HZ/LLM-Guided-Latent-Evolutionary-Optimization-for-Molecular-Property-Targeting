#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""LLM generator for the ZINC logP transfer experiment.

The script supports two paper groups:
- LLM-generated molecules: evaluate the generated SMILES directly by RDKit.
- LLM-initialized / iterative LLM-guided latent GA: encode accepted SMILES to
  PS-VAE latents and use them as GA initial seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

LLM_CONFIG_DIR = Path("/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm")
if str(LLM_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_CONFIG_DIR))

try:
    from llm_config import Config
except Exception as exc:  # pragma: no cover
    Config = None
    CONFIG_IMPORT_ERROR = exc
else:
    CONFIG_IMPORT_ERROR = None


SYSTEM_PROMPT = """You are a molecular design assistant.
Return ONLY parseable JSON.
Do not include markdown, explanations, or code fences.""".strip()


GEN0_PROMPT_TEMPLATE = """You are an expert medicinal chemist.

Task:
Generate ZINC-like drug-like organic molecules for a molecular optimization transfer experiment.

Objective:
- Target RDKit Crippen MolLogP close to {target_logp:.2f}.
- Success range: {success_low:.2f} <= MolLogP <= {success_high:.2f}.
- Prefer chemically plausible, synthesizable, neutral, single-fragment molecules.

Hard constraints:
- Return valid SMILES only.
- Allowed atoms: C, N, O, S, F, Cl, Br, I, P.
- No salts, metals, radicals, isotopes, disconnected fragments, or charged species.
- Heavy atom count should be between {min_heavy_atoms} and {max_heavy_atoms}.
- Prefer molecular weight between 150 and 500.
- Avoid duplicated scaffolds and near-identical analogues.

Design guidance:
- Use ZINC-like motifs such as substituted heterocycles, amides, esters, ethers,
  sulfones, nitriles, halogenated aromatics, and moderately lipophilic rings.
- To approach MolLogP near 3.0, include a balanced amount of hydrophobic
  substituents such as phenyl, cycloalkyl, alkyl, fluoro, chloro, bromo,
  trifluoromethyl, thioether, or tert-butyl groups.
- Avoid molecules that are too polar or too small; too many heteroatoms,
  multiple acids, or many H-bond donors often push logP below target.
- Balance logP accuracy and diversity.
- Do not collapse to one scaffold.

Generation setting:
- Generation index: 0.
- Random seed label: {seed}.
- Propose exactly {n_candidates} unique molecules.

Return exactly one JSON object:
{{
  "smiles": ["SMILES_1", "SMILES_2", "..."]
}}""".strip()


ITER_PROMPT_TEMPLATE = """You are optimizing ZINC-like molecules toward a target RDKit MolLogP.

Objective:
- Target RDKit Crippen MolLogP close to {target_logp:.2f}.
- Success range: {success_low:.2f} <= MolLogP <= {success_high:.2f}.

Hard constraints:
- Valid neutral single-fragment SMILES.
- Allowed atoms: C, N, O, S, F, Cl, Br, I, P.
- Heavy atom count between {min_heavy_atoms} and {max_heavy_atoms}.
- Prefer molecular weight 150-500.

Successful molecules from previous rounds:
{success_memory}

Rejected or weak molecules from previous rounds:
{failure_memory}

Instructions for generation {generation_idx}:
- Propose exactly {n_candidates} new unique SMILES.
- Reuse useful motifs only if they help reach logP ~= {target_logp:.2f}.
- If previous molecules were below target, increase moderate hydrophobicity
  using halogens, phenyl/cycloalkyl groups, alkyl chains, thioethers, or CF3.
- If previous molecules were above target, add limited polarity without creating
  salts or charged groups.
- Avoid repeating exact molecules from memory.
- Keep scaffold diversity.

Return exactly one JSON object:
{{
  "smiles": ["SMILES_1", "SMILES_2", "..."]
}}""".strip()


ALLOWED_ATOMS = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "P"}


def make_client():
    if Config is None:
        raise RuntimeError(f"Could not import llm_config.py: {CONFIG_IMPORT_ERROR}")
    from openai import OpenAI

    config = Config()
    return OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_KEY)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def parse_json_response(text: str) -> Dict:
    text = strip_code_fences(text or "")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        obj = json.loads(text[first:last + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Could not parse JSON response")


def extract_smiles(obj: Dict) -> List[str]:
    values = obj.get("smiles", [])
    if not values and isinstance(obj.get("candidates"), list):
        values = [
            x.get("smiles") if isinstance(x, dict) else x
            for x in obj["candidates"]
        ]
    if not isinstance(values, list):
        return []
    return [str(x).strip() for x in values if isinstance(x, str) and x.strip()]


def canonicalize_and_validate(
    smiles: str,
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    require_neutral: bool = True,
) -> Tuple[bool, Optional[str], str, Dict]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False, None, "rdkit_parse_failed", {}
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False, None, "sanitize_failed", {}

    if "." in Chem.MolToSmiles(mol):
        return False, None, "multi_fragment", {}
    if require_neutral and sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != 0:
        return False, None, "non_neutral", {}
    for atom in mol.GetAtoms():
        if atom.GetIsotope() != 0 or atom.GetNumRadicalElectrons() > 0:
            return False, None, "radical_or_isotope", {}
        if atom.GetSymbol() not in ALLOWED_ATOMS:
            return False, None, f"disallowed_atom_{atom.GetSymbol()}", {}

    heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if heavy_atoms < min_heavy_atoms:
        return False, None, "too_few_heavy_atoms", {}
    if heavy_atoms > max_heavy_atoms:
        return False, None, "too_many_heavy_atoms", {}

    cano = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not cano:
        return False, None, "canonicalize_failed", {}

    props = compute_rdkit_properties(cano)
    return True, cano, "ok", props


def compute_rdkit_properties(smiles: str) -> Dict:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {}
    logp = float(Crippen.MolLogP(mol))
    return {
        "rdkit_logP": logp,
        "qed": float(QED.qed(mol)),
        "mol_weight": float(Descriptors.MolWt(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "heavy_atoms": int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }


def logp_score(logp: float, target: float, sigma: float) -> float:
    return float(math.exp(-0.5 * ((logp - target) / sigma) ** 2))


def call_llm(
    client,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> str:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            raise RuntimeError(f"Empty response content: {response}")
        except Exception as exc:
            last_err = exc
            print(f"[WARN] LLM call failed attempt {attempt}/{max_retries}: {exc}", flush=True)
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def build_success_memory(entries: List[Dict], top_k: int) -> List[Dict]:
    accepted = [x for x in entries if x["status"] == "accepted"]
    accepted = sorted(accepted, key=lambda x: x["rdkit_abs_error"])
    out = []
    seen = set()
    for row in accepted:
        if row["smiles"] in seen:
            continue
        seen.add(row["smiles"])
        out.append({
            "smiles": row["smiles"],
            "rdkit_logP": round(row["rdkit_logP"], 4),
            "abs_error": round(row["rdkit_abs_error"], 4),
            "qed": round(row["qed"], 4),
        })
        if len(out) >= top_k:
            break
    return out


def build_failure_memory(entries: List[Dict], bottom_k: int) -> List[Dict]:
    rejected = [x for x in entries if x["status"] != "accepted"][-bottom_k:]
    weak = [x for x in entries if x["status"] == "accepted"]
    weak = sorted(weak, key=lambda x: x["rdkit_abs_error"], reverse=True)[:bottom_k]
    out = []
    for row in rejected:
        out.append({"smiles": row.get("raw_smiles"), "reason": row.get("reason")})
    for row in weak:
        out.append({
            "smiles": row.get("smiles"),
            "reason": "far_from_target",
            "rdkit_logP": round(float(row.get("rdkit_logP", 0.0)), 4),
        })
    return out[:bottom_k]


def build_prompt(
    generation_idx: int,
    args,
    success_memory: List[Dict],
    failure_memory: List[Dict],
) -> str:
    if generation_idx == 0 or args.mode == "direct":
        return GEN0_PROMPT_TEMPLATE.format(
            target_logp=args.target_logp,
            success_low=args.success_low,
            success_high=args.success_high,
            min_heavy_atoms=args.min_heavy_atoms,
            max_heavy_atoms=args.max_heavy_atoms,
            n_candidates=args.n_candidates_per_call,
            seed=args.seed,
        )
    return ITER_PROMPT_TEMPLATE.format(
        target_logp=args.target_logp,
        success_low=args.success_low,
        success_high=args.success_high,
        min_heavy_atoms=args.min_heavy_atoms,
        max_heavy_atoms=args.max_heavy_atoms,
        n_candidates=args.n_candidates_per_call,
        generation_idx=generation_idx,
        success_memory=json.dumps(success_memory, ensure_ascii=False, indent=2),
        failure_memory=json.dumps(failure_memory, ensure_ascii=False, indent=2),
    )


def run_generation(args) -> Dict:
    random.seed(args.seed)
    client = make_client()

    all_entries = []
    accepted = []
    accepted_set = set()
    generation_logs = []
    success_memory = []
    failure_memory = []

    for gen_idx in range(args.generations):
        if len(accepted) >= args.target_total:
            break
        prompt = build_prompt(gen_idx, args, success_memory, failure_memory)
        raw_text = call_llm(
            client,
            prompt=prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )

        parse_error = False
        try:
            obj = parse_json_response(raw_text)
            smiles_list = extract_smiles(obj)
        except Exception as exc:
            parse_error = True
            smiles_list = []
            obj = {"error": str(exc)}

        gen_records = []
        rejected_stats = {}
        seen_in_gen = set()

        for raw_smiles in smiles_list:
            if raw_smiles in seen_in_gen:
                reason = "duplicate_in_generation"
                rejected_stats[reason] = rejected_stats.get(reason, 0) + 1
                gen_records.append({
                    "generation": gen_idx,
                    "raw_smiles": raw_smiles,
                    "smiles": "",
                    "status": "rejected",
                    "reason": reason,
                })
                continue
            seen_in_gen.add(raw_smiles)

            ok, cano, reason, props = canonicalize_and_validate(
                raw_smiles,
                min_heavy_atoms=args.min_heavy_atoms,
                max_heavy_atoms=args.max_heavy_atoms,
                require_neutral=not args.allow_charged,
            )
            if not ok:
                rejected_stats[reason] = rejected_stats.get(reason, 0) + 1
                gen_records.append({
                    "generation": gen_idx,
                    "raw_smiles": raw_smiles,
                    "smiles": "",
                    "status": "rejected",
                    "reason": reason,
                })
                continue
            if cano in accepted_set:
                reason = "duplicate_global"
                rejected_stats[reason] = rejected_stats.get(reason, 0) + 1
                gen_records.append({
                    "generation": gen_idx,
                    "raw_smiles": raw_smiles,
                    "smiles": cano,
                    "status": "rejected",
                    "reason": reason,
                })
                continue

            props["rdkit_abs_error"] = abs(props["rdkit_logP"] - args.target_logp)
            props["rdkit_success"] = bool(args.success_low <= props["rdkit_logP"] <= args.success_high)
            props["score"] = logp_score(props["rdkit_logP"], args.target_logp, args.score_sigma)

            row = {
                "generation": gen_idx,
                "raw_smiles": raw_smiles,
                "smiles": cano,
                "status": "accepted",
                "reason": "ok",
                **props,
            }
            gen_records.append(row)
            accepted.append(cano)
            accepted_set.add(cano)
            if len(accepted) >= args.target_total:
                break

        all_entries.extend(gen_records)
        success_memory = build_success_memory(all_entries, args.success_memory_size)
        failure_memory = build_failure_memory(all_entries, args.failure_memory_size)

        generation_logs.append({
            "generation": gen_idx,
            "prompt": prompt,
            "raw_response": raw_text,
            "parse_error": parse_error,
            "raw_count": len(smiles_list),
            "accepted_count": sum(1 for x in gen_records if x["status"] == "accepted"),
            "accepted_total": len(accepted),
            "rejected_stats": rejected_stats,
            "success_memory": success_memory,
            "failure_memory": failure_memory,
        })
        print(
            f"[Gen {gen_idx:03d}] raw={len(smiles_list)} "
            f"accepted={generation_logs[-1]['accepted_count']} "
            f"total={len(accepted)}/{args.target_total} rejected={rejected_stats}",
            flush=True,
        )

    accepted_entries = [x for x in all_entries if x["status"] == "accepted"]
    accepted_entries = sorted(accepted_entries, key=lambda x: (x["rdkit_abs_error"], x["smiles"]))
    for i, row in enumerate(accepted_entries, start=1):
        row["rank_by_rdkit_abs_error"] = i

    summary = {
        "task": "zinc_logp_llm_generation",
        "mode": args.mode,
        "model": args.model,
        "seed": args.seed,
        "target_logP": args.target_logp,
        "success_low": args.success_low,
        "success_high": args.success_high,
        "generations_requested": args.generations,
        "n_candidates_per_call": args.n_candidates_per_call,
        "target_total": args.target_total,
        "accepted_total": len(accepted_entries),
        "rdkit_success_count": int(sum(1 for x in accepted_entries if x["rdkit_success"])),
        "rdkit_success_rate": (
            float(sum(1 for x in accepted_entries if x["rdkit_success"]) / len(accepted_entries))
            if accepted_entries else 0.0
        ),
        "best_smiles": accepted_entries[0]["smiles"] if accepted_entries else None,
        "best_rdkit_logP": accepted_entries[0]["rdkit_logP"] if accepted_entries else None,
        "best_rdkit_abs_error": accepted_entries[0]["rdkit_abs_error"] if accepted_entries else None,
        "top10_rdkit_abs_error_mean": (
            float(sum(x["rdkit_abs_error"] for x in accepted_entries[:10]) / min(10, len(accepted_entries)))
            if accepted_entries else None
        ),
        "accepted_smiles": [x["smiles"] for x in accepted_entries],
        "all_entries": all_entries,
        "accepted_entries_ranked": accepted_entries,
        "generation_logs": generation_logs,
    }
    return summary


def save_outputs(result: Dict, out_dir: Path, prefix: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{prefix}_{result['mode']}_{result['model'].replace('/', '_')}_seed{result['seed']}_{timestamp}"

    paths = {
        "summary_json": str(out_dir / f"{stem}.json"),
        "smiles_smi": str(out_dir / f"{stem}.smi"),
        "entries_jsonl": str(out_dir / f"{stem}.jsonl"),
        "accepted_csv": str(out_dir / f"{stem}_accepted_ranked.csv"),
    }

    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(paths["smiles_smi"], "w", encoding="utf-8") as f:
        for smi in result["accepted_smiles"]:
            f.write(smi + "\n")

    with open(paths["entries_jsonl"], "w", encoding="utf-8") as f:
        for row in result["all_entries"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "rank_by_rdkit_abs_error", "generation", "smiles", "rdkit_logP",
        "rdkit_abs_error", "rdkit_success", "score", "qed", "mol_weight",
        "tpsa", "hbd", "hba", "rotatable_bonds", "rings", "heavy_atoms",
        "fraction_csp3", "raw_smiles",
    ]
    with open(paths["accepted_csv"], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["accepted_entries_ranked"])

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ZINC-like logP-targeted SMILES with an LLM.")
    parser.add_argument("--mode", choices=["direct", "iterative"], default="direct")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--n_candidates_per_call", type=int, default=30)
    parser.add_argument("--target_total", type=int, default=500)
    parser.add_argument("--target_logp", type=float, default=3.0)
    parser.add_argument("--success_low", type=float, default=2.5)
    parser.add_argument("--success_high", type=float, default=3.5)
    parser.add_argument("--score_sigma", type=float, default=0.5)
    parser.add_argument("--min_heavy_atoms", type=int, default=10)
    parser.add_argument("--max_heavy_atoms", type=int, default=45)
    parser.add_argument("--allow_charged", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2200)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--success_memory_size", type=int, default=20)
    parser.add_argument("--failure_memory_size", type=int, default=12)
    parser.add_argument("--out_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_LLM/smiles"))
    parser.add_argument("--prefix", default="zinc_logp_llm")
    args = parser.parse_args()

    result = run_generation(args)
    paths = save_outputs(result, args.out_dir, args.prefix)

    print("\n========== LLM GENERATION DONE ==========")
    print(f"Mode: {result['mode']}")
    print(f"Accepted: {result['accepted_total']}/{result['target_total']}")
    print(f"RDKit success: {result['rdkit_success_count']} ({result['rdkit_success_rate']:.4f})")
    print(f"Best RDKit logP: {result['best_rdkit_logP']}")
    print(f"Best abs error: {result['best_rdkit_abs_error']}")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
