# ZINC logP LLM transfer experiment

This folder stores the LLM-side transfer experiments for target RDKit MolLogP = 3.0.

## Groups

1. `LLM-Generated Molecules`
   - LLM directly proposes ZINC-like SMILES.
   - Molecules are evaluated by RDKit Crippen MolLogP.
   - No latent GA is used.

2. `LLM-Initialized Latent GA`
   - The same direct LLM-generated SMILES are encoded by the ZINC PS-VAE encoder.
   - The resulting `llm_init_latent.npy` initializes decode-aware latent GA.

3. `Iterative LLM-Guided Latent GA`
   - LLM generates SMILES over multiple rounds using success/failure memory.
   - Accepted SMILES are encoded by the ZINC PS-VAE encoder.
   - The resulting latent pool initializes decode-aware latent GA.

The existing `ZINC-Seeded Latent GA` baseline is stored under:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek/results
```

## Fixed resources

```text
PS-VAE checkpoint:
/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt

Correct kekulized latent pool:
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek/train/zinc_logp_latent.npy

Correct logP predictor:
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek/logp_predictor/best_logp_predictor.pt
```

## Run

```bash
cd /root/autodl-tmp/sweeteners_evolve
GPU=0 bash /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_LLM/run_zinc_logp_llm_transfer.sh
```

Recommended detached run:

```bash
tmux new-session -d -s zinc_logp_llm_transfer \
  'cd /root/autodl-tmp/sweeteners_evolve && GPU=0 bash /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_LLM/run_zinc_logp_llm_transfer.sh > /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_LLM/logs/pipeline.log 2>&1'
```

## Outputs

```text
smiles/
  LLM direct and iterative SMILES, JSONL logs, and RDKit-ranked CSV files.

latent/
  direct_seed*/llm_init_latent.npy
  iterative_seed*/llm_init_latent.npy

results/
  llm_llm_initialized_seed*/summary_decode_aware.json
  llm_iterative_llm_guided_seed*/summary_decode_aware.json
  zinc_logp_llm_transfer_3groups_detail.csv
  zinc_logp_llm_transfer_3groups_mean_std.csv

logs/
  Per-step logs and pipeline.log.
```

## Reporting

Use RDKit molecule-level metrics for the paper:

- `best_rdkit_abs_error`
- `top10_rdkit_abs_error_mean`
- `decode_latent_validity_final`
- `archive_unique_valid_molecules`
- `archive_rdkit_success_unique`
- `diversity_unique_valid`

