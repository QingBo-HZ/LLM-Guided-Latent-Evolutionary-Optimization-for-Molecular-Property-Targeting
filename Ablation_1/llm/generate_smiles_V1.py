import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from openai import OpenAI, OpenAIError
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from llm_config import Config

RDLogger.DisableLog("rdApp.*")

config = Config()
client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_KEY)


# ============================================================
# 0. 可编辑的提示词模板（单独列出，便于你后续改）
# ============================================================

SYSTEM_PROMPT = """
You are a molecular design assistant.
Return ONLY one valid JSON object.
Do NOT output any explanation, reasoning, analysis, markdown, code fences, or extra text.
Do NOT include prose before or after JSON.
The response must be parseable by json.loads().
""".strip()


GEN0_PROMPT_TEMPLATE = """
Task:
{task_name}

Objective:
{objective_text}

Constraints:
{constraints_text}

Design rules:
{design_rules_text}

Generation setting:
- This is generation 0 (cold start).
- Propose chemically plausible seed molecules.
- Balance diversity and task relevance.
- Avoid near-duplicates.

Output requirements:
Return exactly one JSON object with this schema:
{{
  "smiles": ["SMILES_1", "SMILES_2"]
}}

Strict requirements:
- Return exactly {n_candidates} unique SMILES in "smiles".
- "smiles" must be a JSON array of strings.
- No markdown.
- No commentary.
- No explanation.
- JSON only.
""".strip()


ITER_PROMPT_TEMPLATE = """
Task:
{task_name}

Objective:
{objective_text}

Constraints:
{constraints_text}

Current design rules:
{design_rules_text}

Success memory (high-scoring valid molecules):
{success_memory_json}

Failure memory (invalid or low-scoring molecules):
{failure_memory_json}

Generation setting:
- This is generation {generation_idx}.
- Use the success memory to refine proposals.
- Avoid repeating failure patterns.
- Keep some diversity; do not over-collapse to one motif.

Output requirements:
Return exactly one JSON object with this schema:
{{
  "smiles": ["SMILES_1", "SMILES_2", "SMILES_3"]
}}

Strict requirements:
- Return exactly {n_candidates} unique SMILES in "smiles".
- "smiles" must be a JSON array of strings.
- No markdown.
- No commentary.
- No explanation.
- JSON only.
""".strip()


# ============================================================
# 1. 任务配置
# ============================================================

DEFAULT_TASKS = {
    "qm9_gap_min": {
        "task_name": "Generate QM9-like small organic molecules",
        "objective_text": (
            "Generate molecules likely to have a SMALL HOMO-LUMO gap. "
            "Favor chemically plausible motifs that may reduce the gap under QM9-like constraints."
        ),
        "constraints_text": (
            "- Allowed heavy atoms: C, N, O, F\n"
            "- Maximum heavy atom count <= 9\n"
            "- Molecules must be neutral\n"
            "- Molecules must be single-fragment\n"
            "- Molecules must be RDKit-parseable and valence-valid\n"
            "- Avoid salts, metals, radicals, isotopes, disconnected structures\n"
            "- Prefer QM9-scale small molecules"
        ),
        "design_rules_text": (
            "- Favor moderate conjugation when chemically reasonable\n"
            "- Favor heteroatom arrangements that may lower the gap\n"
            "- Maintain structural plausibility and diversity\n"
            "- Avoid obviously unstable motifs"
        ),
    },
    "qm9_gap_max": {
        "task_name": "Generate QM9-like small organic molecules",
        "objective_text": (
            "Generate molecules likely to have a LARGE HOMO-LUMO gap. "
            "Favor chemically plausible motifs that may increase the gap under QM9-like constraints."
        ),
        "constraints_text": (
            "- Allowed heavy atoms: C, N, O, F\n"
            "- Maximum heavy atom count <= 9\n"
            "- Molecules must be neutral\n"
            "- Molecules must be single-fragment\n"
            "- Molecules must be RDKit-parseable and valence-valid\n"
            "- Avoid salts, metals, radicals, isotopes, disconnected structures\n"
            "- Prefer QM9-scale small molecules"
        ),
        "design_rules_text": (
            "- Favor saturated or less-conjugated motifs when chemically reasonable\n"
            "- Avoid excessive conjugation\n"
            "- Maintain structural plausibility and diversity\n"
            "- Avoid obviously unstable motifs"
        ),
    },
}


# ============================================================
# 2. LLM 调用
# ============================================================

def call_llm_json(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
    retry_delay: float = 1.5,
) -> str:
    last_err = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # -------- 更稳的 content 提取 --------
            msg = response.choices[0].message
            content = getattr(msg, "content", None)

            # 情况1：普通字符串
            if isinstance(content, str):
                text = content.strip()
                if text:
                    return text

            # 情况2：content 是 list / 分段结构
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and item.get("text"):
                            parts.append(item["text"])
                    else:
                        txt = getattr(item, "text", None)
                        if txt:
                            parts.append(txt)
                text = "".join(parts).strip()
                if text:
                    return text

            # 情况3：尝试把 message 整体转成 dict 看有没有别的字段
            try:
                if hasattr(msg, "model_dump"):
                    msg_dump = msg.model_dump()
                elif hasattr(msg, "dict"):
                    msg_dump = msg.dict()
                else:
                    msg_dump = {"repr": repr(msg)}
            except Exception:
                msg_dump = {"repr": repr(msg)}

            # 把完整 choice 也一起保存，方便后续解析
            try:
                if hasattr(response.choices[0], "model_dump"):
                    choice_dump = response.choices[0].model_dump()
                elif hasattr(response.choices[0], "dict"):
                    choice_dump = response.choices[0].dict()
                else:
                    choice_dump = {"repr": repr(response.choices[0])}
            except Exception:
                choice_dump = {"repr": repr(response.choices[0])}

            # 返回一个可解析的错误 JSON，别返回空串
            return json.dumps(
                {
                    "error": "empty_or_unexpected_response",
                    "message_dump": msg_dump,
                    "choice_dump": choice_dump,
                },
                ensure_ascii=False,
            )

        except (OpenAIError, Exception) as e:
            last_err = e
            print(f"[Retry {attempt + 1}/{max_retries}] Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return json.dumps({"error": str(last_err)}, ensure_ascii=False)


# ============================================================
# 3. JSON 解析
# ============================================================

def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def try_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None

    text = strip_code_fences(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


# ============================================================
# 4. SMILES 清洗与过滤（QM9 风格）
# ============================================================

ALLOWED_ATOMS = {"C", "N", "O", "F"}


def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def is_single_fragment(mol: Chem.Mol) -> bool:
    return "." not in Chem.MolToSmiles(mol)


def is_neutral(mol: Chem.Mol) -> bool:
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == 0


def has_only_qm9_atoms(mol: Chem.Mol) -> bool:
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == "H":
            continue
        if symbol not in ALLOWED_ATOMS:
            return False
    return True


def heavy_atom_count(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def has_radical_or_isotope(mol: Chem.Mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True
        if atom.GetIsotope() != 0:
            return True
    return False


def passes_qm9_filter(smiles: str) -> Tuple[bool, Optional[str], str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, None, "rdkit_parse_failed"

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False, None, "sanitize_failed"

    if not is_single_fragment(mol):
        return False, None, "multi_fragment"

    if not is_neutral(mol):
        return False, None, "non_neutral"

    if not has_only_qm9_atoms(mol):
        return False, None, "non_qm9_atoms"

    if has_radical_or_isotope(mol):
        return False, None, "radical_or_isotope"

    if heavy_atom_count(mol) > 9:
        return False, None, "too_many_heavy_atoms"

    cano = Chem.MolToSmiles(mol, canonical=True)
    if not cano:
        return False, None, "canonicalize_failed"

    return True, cano, "ok"


# ============================================================
# 5. 弱 LLEMA 的“打分器”
#    第一版先用 proxy score
#    后面你可以替换成真实 predictor
# ============================================================

def proxy_score_gap_min(smiles: str) -> float:
    """
    分数越高 = 越像低 gap 候选
    简化启发式：
    - 更多共轭
    - 更多芳香/不饱和
    - 适度杂原子
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1e9

    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    conjugated_bonds = 0
    double_triple_bonds = 0
    for bond in mol.GetBonds():
        if bond.GetIsConjugated():
            conjugated_bonds += 1
        if bond.GetBondTypeAsDouble() >= 2:
            double_triple_bonds += 1

    score = (
        2.0 * conjugated_bonds
        + 1.5 * double_triple_bonds
        + 1.2 * n_aromatic_rings
        + 0.4 * n_hetero
        - 0.2 * n_rot
    )
    return float(score)


def proxy_score_gap_max(smiles: str) -> float:
    """
    分数越高 = 越像高 gap 候选
    简化启发式：
    - 更少共轭
    - 更少芳香
    - 更小更饱和
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1e9

    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    ha = heavy_atom_count(mol)

    conjugated_bonds = 0
    double_triple_bonds = 0
    for bond in mol.GetBonds():
        if bond.GetIsConjugated():
            conjugated_bonds += 1
        if bond.GetBondTypeAsDouble() >= 2:
            double_triple_bonds += 1

    score = (
        2.5 * max(0, 9 - ha)
        - 2.0 * conjugated_bonds
        - 1.5 * double_triple_bonds
        - 1.0 * n_aromatic_rings
        - 0.2 * n_hetero
    )
    return float(score)


def score_smiles(smiles: str, score_mode: str) -> float:
    if score_mode == "proxy_gap_min":
        return proxy_score_gap_min(smiles)
    elif score_mode == "proxy_gap_max":
        return proxy_score_gap_max(smiles)
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")


# ============================================================
# 6. 记忆池管理
# ============================================================

def build_success_memory(entries: List[Dict], top_k: int) -> List[Dict]:
    valid_entries = [x for x in entries if x["status"] == "accepted"]
    valid_entries = sorted(valid_entries, key=lambda x: x["score"], reverse=True)
    out = []
    seen = set()
    for x in valid_entries:
        if x["smiles"] in seen:
            continue
        seen.add(x["smiles"])
        out.append({
            "smiles": x["smiles"],
            "score": round(float(x["score"]), 4),
            "source": x.get("source", "unknown"),
        })
        if len(out) >= top_k:
            break
    return out


def build_failure_memory(entries: List[Dict], bottom_k: int) -> List[Dict]:
    """
    失败池包括两类：
    1) invalid / filtered
    2) accepted 但低分
    """
    failed = [x for x in entries if x["status"] != "accepted"]
    low_score = [x for x in entries if x["status"] == "accepted"]
    low_score = sorted(low_score, key=lambda x: x["score"])[:bottom_k]

    out = []
    seen = set()

    for x in failed:
        key = (x["raw_smiles"], x["reason"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "smiles": x["raw_smiles"],
            "reason": x["reason"],
        })

    for x in low_score:
        key = (x["smiles"], "low_score")
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "smiles": x["smiles"],
            "reason": "low_score",
            "score": round(float(x["score"]), 4),
        })

    return out[:bottom_k]


def update_design_rules(
    base_rules_text: str,
    success_memory: List[Dict],
    failure_memory: List[Dict],
    objective_name: str,
) -> str:
    """
    第一版弱 LLEMA：不让 LLM 自动总结规则，先程序化拼接
    """
    extra_rules = []

    if success_memory:
        extra_rules.append(
            f"- Build on recurring motifs from recent successful candidates relevant to {objective_name}"
        )

    if failure_memory:
        extra_rules.append(
            "- Avoid repeating patterns observed in invalid or low-scoring molecules"
        )

    if objective_name == "gap_min":
        extra_rules.append("- Keep some conjugated or heteroatom-rich candidates for exploration")
    elif objective_name == "gap_max":
        extra_rules.append("- Keep some saturated and less-conjugated candidates for exploration")

    return base_rules_text + "\n" + "\n".join(extra_rules)


# ============================================================
# 7. Prompt 构建
# ============================================================

def get_task_config(task_key: str) -> Dict:
    if task_key not in DEFAULT_TASKS:
        raise ValueError(f"Unsupported task_key: {task_key}")
    return DEFAULT_TASKS[task_key]


def build_gen0_prompt(task_cfg: Dict, n_candidates: int) -> str:
    return GEN0_PROMPT_TEMPLATE.format(
        task_name=task_cfg["task_name"],
        objective_text=task_cfg["objective_text"],
        constraints_text=task_cfg["constraints_text"],
        design_rules_text=task_cfg["design_rules_text"],
        n_candidates=n_candidates,
    )


def build_iter_prompt(
    task_cfg: Dict,
    generation_idx: int,
    n_candidates: int,
    success_memory: List[Dict],
    failure_memory: List[Dict],
    design_rules_text: str,
) -> str:
    return ITER_PROMPT_TEMPLATE.format(
        task_name=task_cfg["task_name"],
        objective_text=task_cfg["objective_text"],
        constraints_text=task_cfg["constraints_text"],
        design_rules_text=design_rules_text,
        success_memory_json=json.dumps(success_memory, ensure_ascii=False, indent=2),
        failure_memory_json=json.dumps(failure_memory, ensure_ascii=False, indent=2),
        generation_idx=generation_idx,
        n_candidates=n_candidates,
    )


# ============================================================
# 8. 从 LLM 响应中提取候选
# ============================================================

def parse_generation_output(raw_text: str) -> Dict:
    obj = try_parse_json(raw_text)
    if obj is None:
        return {
            "generation": None,
            "applied_rules": [],
            "candidate_sources": [],
            "smiles": [],
            "parse_error": True,
            "raw_text": raw_text,
        }

    # 如果是错误 JSON，也按 parse_error 处理
    if "error" in obj:
        return {
            "generation": None,
            "applied_rules": [],
            "candidate_sources": [],
            "smiles": [],
            "parse_error": True,
            "raw_text": raw_text,
        }

    generation = obj.get("generation", None)
    applied_rules = obj.get("applied_rules", [])
    candidate_sources = obj.get("candidate_sources", [])
    smiles_list = obj.get("smiles", [])

    if not smiles_list and "candidates" in obj and isinstance(obj["candidates"], list):
        tmp = []
        for x in obj["candidates"]:
            if isinstance(x, dict) and isinstance(x.get("smiles"), str):
                tmp.append(x["smiles"])
            elif isinstance(x, str):
                tmp.append(x)
        smiles_list = tmp

    if not smiles_list and candidate_sources and isinstance(candidate_sources, list):
        tmp = []
        for x in candidate_sources:
            if isinstance(x, dict) and isinstance(x.get("smiles"), str):
                tmp.append(x["smiles"])
        smiles_list = tmp

    if not isinstance(applied_rules, list):
        applied_rules = []
    if not isinstance(candidate_sources, list):
        candidate_sources = []
    if not isinstance(smiles_list, list):
        smiles_list = []

    clean_smiles = []
    for s in smiles_list:
        if isinstance(s, str) and s.strip():
            clean_smiles.append(s.strip())

    # 如果没有任何 smiles，也视为 parse_error，避免“空成功”
    if len(clean_smiles) == 0:
        return {
            "generation": generation,
            "applied_rules": applied_rules,
            "candidate_sources": candidate_sources,
            "smiles": [],
            "parse_error": True,
            "raw_text": raw_text,
        }

    return {
        "generation": generation,
        "applied_rules": applied_rules,
        "candidate_sources": candidate_sources,
        "smiles": clean_smiles,
        "parse_error": False,
        "raw_text": raw_text,
    }

def build_source_map(candidate_sources: List[Dict]) -> Dict[str, str]:
    source_map = {}
    for item in candidate_sources:
        if not isinstance(item, dict):
            continue
        s = item.get("smiles", "")
        src = item.get("source", "unknown")
        if isinstance(s, str) and s.strip():
            source_map[s.strip()] = src
    return source_map


# ============================================================
# 9. 主循环：弱 LLEMA 生成
# ============================================================

def run_weak_llema_generation(
    model_name: str,
    task_key: str,
    score_mode: str,
    generations: int,
    n_candidates_per_gen: int,
    target_total: int,
    success_memory_size: int,
    failure_memory_size: int,
    temperature: float,
    max_tokens: int,
) -> Dict:
    task_cfg = get_task_config(task_key)

    all_entries = []
    accepted_global = []
    accepted_set = set()
    generation_logs = []

    success_memory = []
    failure_memory = []
    current_rules = task_cfg["design_rules_text"]

    objective_name = "gap_min" if "gap_min" in task_key else "gap_max"

    parse_err_dir = Path("./smiles_generated")
    parse_err_dir.mkdir(parents=True, exist_ok=True)

    for gen_idx in range(generations):
        if len(accepted_global) >= target_total:
            break

        if gen_idx == 0:
            prompt = build_gen0_prompt(task_cfg, n_candidates_per_gen)
        else:
            current_rules = update_design_rules(
                base_rules_text=task_cfg["design_rules_text"],
                success_memory=success_memory,
                failure_memory=failure_memory,
                objective_name=objective_name,
            )
            prompt = build_iter_prompt(
                task_cfg=task_cfg,
                generation_idx=gen_idx,
                n_candidates=n_candidates_per_gen,
                success_memory=success_memory,
                failure_memory=failure_memory,
                design_rules_text=current_rules,
            )

        raw_text = call_llm_json(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = parse_generation_output(raw_text)
        source_map = build_source_map(parsed["candidate_sources"])

        gen_records = []
        rejected_stats = {}

        if parsed["parse_error"]:
            print(f"[Gen {gen_idx}] parse_error=True")
            print("[RAW RESPONSE]")
            print(raw_text[:1000] if raw_text else raw_text)

            err_path = parse_err_dir / f"parse_error_gen_{gen_idx}.txt"
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(raw_text if raw_text else "")

            prompt_path = parse_err_dir / f"parse_error_gen_{gen_idx}_prompt.txt"
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)

            generation_logs.append({
                "generation": gen_idx,
                "prompt": prompt,
                "raw_response": raw_text,
                "parse_error": True,
                "accepted_count": 0,
                "rejected_stats": {"json_parse_error": 1},
                "applied_rules": [],
            })
            continue

        smiles_seen_in_gen = set()

        for raw_smiles in parsed["smiles"]:
            if raw_smiles in smiles_seen_in_gen:
                rejected_stats["duplicate_in_generation"] = rejected_stats.get("duplicate_in_generation", 0) + 1
                gen_records.append({
                    "generation": gen_idx,
                    "raw_smiles": raw_smiles,
                    "smiles": None,
                    "source": source_map.get(raw_smiles, "unknown"),
                    "status": "rejected",
                    "reason": "duplicate_in_generation",
                    "score": None,
                })
                continue
            smiles_seen_in_gen.add(raw_smiles)

            ok, cano, reason = passes_qm9_filter(raw_smiles)
            if not ok:
                rejected_stats[reason] = rejected_stats.get(reason, 0) + 1
                gen_records.append({
                    "generation": gen_idx,
                    "raw_smiles": raw_smiles,
                    "smiles": None,
                    "source": source_map.get(raw_smiles, "unknown"),
                    "status": "rejected",
                    "reason": reason,
                    "score": None,
                })
                continue

            if cano in accepted_set:
                rejected_stats["duplicate_global"] = rejected_stats.get("duplicate_global", 0) + 1
                gen_records.append({
                    "generation": gen_idx,
                    "raw_smiles": raw_smiles,
                    "smiles": cano,
                    "source": source_map.get(raw_smiles, "unknown"),
                    "status": "rejected",
                    "reason": "duplicate_global",
                    "score": None,
                })
                continue

            score = score_smiles(cano, score_mode)
            accepted_set.add(cano)
            accepted_global.append(cano)

            rec = {
                "generation": gen_idx,
                "raw_smiles": raw_smiles,
                "smiles": cano,
                "source": source_map.get(raw_smiles, "unknown"),
                "status": "accepted",
                "reason": "ok",
                "score": float(score),
            }
            gen_records.append(rec)

            if len(accepted_global) >= target_total:
                break

        all_entries.extend(gen_records)

        success_memory = build_success_memory(all_entries, top_k=success_memory_size)
        failure_memory = build_failure_memory(all_entries, bottom_k=failure_memory_size)

        generation_logs.append({
            "generation": gen_idx,
            "prompt": prompt,
            "raw_response": raw_text,
            "parse_error": False,
            "accepted_count": sum(1 for x in gen_records if x["status"] == "accepted"),
            "rejected_stats": rejected_stats,
            "applied_rules": parsed["applied_rules"],
            "success_memory": success_memory,
            "failure_memory": failure_memory,
        })

        print(f"[Gen {gen_idx}] accepted_total = {len(accepted_global)}/{target_total}")

    summary = {
        "model_name": model_name,
        "task_key": task_key,
        "score_mode": score_mode,
        "generations_requested": generations,
        "n_candidates_per_gen": n_candidates_per_gen,
        "target_total": target_total,
        "accepted_total": len(accepted_global),
        "accepted_smiles": accepted_global,
        "all_entries": all_entries,
        "generation_logs": generation_logs,
    }
    return summary


# ============================================================
# 10. 保存输出
# ============================================================

def save_outputs(result: Dict, out_dir: str) -> Tuple[str, str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"weak_llema_{result['task_key']}_{result['model_name'].replace('/', '_')}_{timestamp}"

    summary_json = out_path / f"{stem}.json"
    smiles_smi = out_path / f"{stem}.smi"
    entries_jsonl = out_path / f"{stem}.jsonl"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(smiles_smi, "w", encoding="utf-8") as f:
        for s in result["accepted_smiles"]:
            f.write(s + "\n")

    with open(entries_jsonl, "w", encoding="utf-8") as f:
        for row in result["all_entries"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return str(summary_json), str(smiles_smi), str(entries_jsonl)


# ============================================================
# 11. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Weak LLEMA-style SMILES generator for ablation experiments.")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--task", type=str, default="qm9_gap_min", choices=list(DEFAULT_TASKS.keys()))
    parser.add_argument("--score_mode", type=str, default="proxy_gap_min", choices=["proxy_gap_min", "proxy_gap_max"])
    parser.add_argument("--generations", type=int, default=5, help="最多生成多少代")
    parser.add_argument("--n_candidates_per_gen", type=int, default=20, help="每代请求多少个候选")
    parser.add_argument("--target_total", type=int, default=100, help="最终保留多少个可用分子")
    parser.add_argument("--success_memory_size", type=int, default=20, help="success memory 保留 top-k")
    parser.add_argument("--failure_memory_size", type=int, default=10, help="failure memory 保留多少条")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--out_dir", type=str, default="./smiles_generated")
    args = parser.parse_args()

    result = run_weak_llema_generation(
        model_name=args.model,
        task_key=args.task,
        score_mode=args.score_mode,
        generations=args.generations,
        n_candidates_per_gen=args.n_candidates_per_gen,
        target_total=args.target_total,
        success_memory_size=args.success_memory_size,
        failure_memory_size=args.failure_memory_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    summary_json, smiles_smi, entries_jsonl = save_outputs(result, args.out_dir)

    print("\n========== DONE ==========")
    print(f"Accepted total: {result['accepted_total']}/{result['target_total']}")
    print(f"Summary JSON : {summary_json}")
    print(f"SMILES .smi  : {smiles_smi}")
    print(f"Entries JSONL: {entries_jsonl}")


if __name__ == "__main__":
    main()