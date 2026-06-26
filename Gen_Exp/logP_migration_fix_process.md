# ZINC logP Migration Experiment: Fix and Paper-Grade Workflow

## Goal

Build a paper-grade migration experiment for applying the latent-space molecular optimization algorithm to ZINC logP optimization.

The final experiment must evaluate generated molecules, not only optimized latent vectors. A molecule is counted as successful only if it is decoded into a valid SMILES and its RDKit logP falls in the target interval.

## Current Diagnosis

The current ZINC logP pipeline has three separate stages:

1. ZINC SMILES -> PS-VAE latent + RDKit logP labels.
2. latent -> logP predictor.
3. GA optimization in latent space -> PS-VAE decoder -> molecules.

The encoding stage is healthy:

| Split | Encoded | Failed | Encode Rate |
|---|---:|---:|---:|
| train | 199565 | 0 | 100% |
| valid | 24945 | 0 | 100% |
| test | 24945 | 0 | 100% |

The current predictor is usable but not strong enough for unrestricted GA exploitation:

| Split | MAE | RMSE | R2 | Pearson |
|---|---:|---:|---:|---:|
| test | 0.758 | 0.977 | 0.535 | 0.731 |

The main failure is the decode stage:

| Latent source | Decode valid rate with current setting |
|---|---:|
| random train latent | 13% |
| initial GA population | 12% |
| final GA population | 14% |

Current GA can optimize predicted logP, but most optimized latent vectors fail to decode. Therefore the old result is not paper-grade molecular generation.

## Paper-Grade Success Definition

For the ZINC logP migration experiment, report two levels of metrics.

### Latent-level metrics

These are diagnostic only:

- Predicted logP target error: `|pred_logP - target_logP|`.
- Predicted success rate: `pred_logP in [success_low, success_high]`.
- Top-10 predicted target error.

### Molecule-level metrics

These are the main paper metrics:

- Decode validity: decoded valid SMILES / population size.
- Unique valid SMILES.
- Diversity among valid decoded SMILES.
- RDKit logP success rate: decoded valid SMILES with `rdkit_logP in [success_low, success_high]` / population size.
- Top-10 molecule-level target error: computed from `rdkit_logP_decoded`, not predictor output.
- Best decoded molecule by RDKit target error.

A generated molecule is successful only if:

```text
smiles is not None
and RDKit can parse the SMILES
and success_low <= RDKit MolLogP(smiles) <= success_high
```

## Experimental Repair Plan

### Phase 1. Decoder validity calibration

Before optimizing, verify that the ZINC-trained PS-VAE can decode its own training latent vectors.

Run a parameter sweep over:

- `add_edge_th`: 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70
- `temperature`: 0.10, 0.30, 0.50, 0.70, 1.00
- `max_atom_num`: 40, 60, 80

Evaluate on random training latent vectors.

Acceptance target:

```text
training-latent decode validity should be substantially higher than the current 13%.
```

If no parameter setting improves validity, the issue is likely the ZINC PS-VAE decoder/checkpoint rather than GA.

### Phase 2. Predictor audit and optional retraining

The current predictor has test R2 around 0.535. This can support rough optimization but is weak for aggressive GA.

Potential adjustments:

- report predictor performance transparently;
- train a stronger residual MLP or ensemble;
- optimize a smoother objective, e.g. target interval score, not exact target point only;
- limit GA exploration to a local manifold around valid training latent vectors.

Acceptance target:

```text
R2 should preferably be >= 0.65 for a convincing main experiment,
or the paper should clearly describe the predictor as a proxy/scorer.
```

### Phase 3. GA repair

The current GA only clips each latent dimension to train min/max. In 128D, this creates many off-manifold points.

Required fixes:

1. Add local-manifold mutation based on nearest training latent neighbors.
2. Add random immigrants from the training latent pool.
3. Add archive selection based on decoded valid molecules, not only predicted score.
4. Add final ranking using RDKit logP of decoded molecules.
5. Save raw latent-level results separately from molecule-level results.

### Phase 4. Paper-grade final run

Run multiple seeds and report mean +/- std:

- seeds: 42, 43, 44 or 42, 123, 2026
- pop_size: 100 or 200
- n_gen: 50-100 after decoder calibration

Primary table columns:

| Method | Validity ↑ | Unique ↑ | Diversity ↑ | RDKit Success ↑ | Best RDKit Error ↓ | Top-10 RDKit Error ↓ | Pred Success ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|

## Running Log

### 2026-06-01 Phase 1 start

Created this workflow file and started decoder validity calibration.


### 2026-06-01 Phase 1 coarse decoder sweep

Created script:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/04_sweep_zinc_decoder.py
```

Command:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/molclr_pyg28/bin/python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/04_sweep_zinc_decoder.py \
  --n_samples 20 \
  --gpu 0 \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/decoder_sweep/coarse_n20
```

Result file:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/decoder_sweep/coarse_n20/decoder_sweep_results.csv
```

Best coarse settings only reached 25% validity on 20 random training latent vectors:

| max_atom_num | add_edge_th | temperature | validity |
|---:|---:|---:|---:|
| 40 | 0.55 | 0.30 | 0.25 |
| 40 | 0.70 | 0.10 | 0.25 |
| 60 | 0.55 | 0.50 | 0.25 |
| 80 | 0.30 | 0.30 | 0.25 |
| 80 | 0.55 | 0.50 | 0.25 |

Interpretation: decoder parameter tuning alone is unlikely to turn the current ZINC PS-VAE checkpoint into a high-validity generator. The next check is whether the checkpoint/data/version is appropriate, and whether the experiment should use a decode-aware GA objective or retrain/fine-tune the generative model.


### 2026-06-01 Phase 1B checkpoint comparison

Found an additional clean ZINC checkpoint:

```text
/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/zinc_deep_v2_clean_1gpu_20260514_140051/lightning_logs/version_0/checkpoints/epoch=0-step=6237.ckpt
```

This checkpoint uses the intended deep-v2 settings, but it only reached epoch 0. A quick decoder sweep gave similar best validity, around 25% on 20 random training latent vectors.

Result file:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/decoder_sweep/clean_epoch0_n20/decoder_sweep_results.csv
```

Conclusion so far:

1. `version_8_zinc` is the only completed ZINC checkpoint currently available.
2. `zinc_deep_v2_clean` exists but is undertrained.
3. Both available checkpoints have low training-latent decode validity under the current decoding API.
4. Therefore, the old logP experiment cannot be made paper-grade only by changing GA hyperparameters.

## Revised Repair Strategy

### Short-term repair: decode-aware GA and molecule-level reporting

This will not fully solve the decoder weakness, but it prevents invalid latent vectors from being reported as generated molecules.

Required changes to `03_optimize_logp_latent_ga.py` or a new v2 script:

1. Track latent-level predicted success separately from molecule-level decoded success.
2. Decode top-k candidates every generation and build a valid-molecule archive.
3. Final reported population should be selected from valid decoded archive, ranked by RDKit logP target error.
4. Summary should include:
   - `latent_success_rate_pred`
   - `decode_validity_raw_final`
   - `mol_success_rate_rdkit`
   - `unique_valid_molecules`
   - `best_rdkit_logp_decoded`
   - `top10_rdkit_abs_error`
5. Old `success_rate_final` based only on predicted logP should be renamed or moved to diagnostic metrics.

### Long-term repair: retrain or continue training ZINC PS-VAE

A paper-grade generation experiment needs a checkpoint whose training-latent decode validity is much higher than 13-25%.

Recommended training target:

```text
training-latent decode validity >= 60% on random latent samples
preferably >= 80% for a strong generation paper result
```

Candidate action:

```bash
bash /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/scripts/train_zinc_deep_v2.sh
```

or resume/continue from:

```text
/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/zinc_deep_v2_clean_1gpu_20260514_140051/lightning_logs/version_0/checkpoints/epoch=0-step=6237.ckpt
```

The retrained checkpoint must be re-audited with `04_sweep_zinc_decoder.py` before running GA.



### 2026-06-01 Phase 3A molecule-level audit of old GA result

Created script:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/05_molecule_level_logp_report.py
```

Command:

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/molclr_pyg28/bin/python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/05_molecule_level_logp_report.py \
  --result_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/results/train_random_sanity_seed42 \
  --target_logp 3.0 \
  --success_low 2.5 \
  --success_high 3.5 \
  --out_prefix molecule_level
```

Output files:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/results/train_random_sanity_seed42/molecule_level_summary.json
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/results/train_random_sanity_seed42/molecule_level_unique_ranked.csv
```

Old GA result under molecule-level RDKit evaluation:

| Metric | Value |
|---|---:|
| decoded valid candidate records | 31 |
| unique valid molecules | 28 |
| RDKit success unique, logP in [2.5, 3.5] | 1 |
| RDKit success rate over unique valid molecules | 3.57% |
| best RDKit SMILES | `Fc1cccc(Cl)c1Cl` |
| best RDKit logP | 3.1325 |
| best RDKit absolute error to 3.0 | 0.1325 |
| top-10 RDKit absolute error mean | 1.3832 |
| diversity among unique valid molecules | 0.9131 |

Interpretation:

The old result is not acceptable as the final paper-level logP migration experiment because its high success rate was based on predicted latent logP, while actual decoded molecules rarely satisfy the RDKit logP target.


### 2026-06-01 Phase 3B decode-aware GA v1

Created corrected GA script:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/03b_optimize_logp_latent_ga_decode_aware.py
```

This script separates:

1. latent-level predicted logP metrics, used only as diagnostic optimization signals;
2. molecule-level decoded valid SMILES and RDKit MolLogP metrics, used as the main reportable result.

Smoke test command:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/molclr_pyg28/bin/python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/03b_optimize_logp_latent_ga_decode_aware.py \
  --init_mode train_random \
  --pop_size 30 \
  --n_gen 3 \
  --elite_size 6 \
  --decode_topk_per_gen 8 \
  --patience 3 \
  --version decode_aware_smoke_seed42 \
  --gpu 0 \
  --seed 42
```

Smoke result:

| Metric | Value |
|---|---:|
| latent predicted success final | 26/30 |
| final decode validity | 7/30 |
| unique valid molecules in archive | 11 |
| RDKit success unique in archive | 0 |
| best RDKit logP | 2.4554 |
| best RDKit absolute error | 0.5446 |

Formal seed-42 diagnostic command:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/molclr_pyg28/bin/python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/03b_optimize_logp_latent_ga_decode_aware.py \
  --init_mode train_random \
  --pop_size 100 \
  --n_gen 30 \
  --elite_size 20 \
  --decode_topk_per_gen 30 \
  --patience 8 \
  --immigrant_ratio 0.20 \
  --max_atom_num 80 \
  --add_edge_th 0.55 \
  --temperature 0.5 \
  --version decode_aware_v1_seed42 \
  --gpu 0 \
  --seed 42
```

Output directory:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/results/train_random_decode_aware_v1_seed42
```

Result:

| Metric | Value |
|---|---:|
| completed generations | 20 |
| latent predicted success final | 89/100 |
| latent best predicted abs error | 0.000067 |
| latent top-10 predicted abs error | 0.000376 |
| final decode validity | 7/100 |
| archive valid records | 91 |
| archive unique valid molecules | 45 |
| archive RDKit success unique | 1 |
| archive RDKit success rate over unique | 2.22% |
| archive RDKit success rate over pop size | 1.00% |
| best RDKit SMILES | `Cc1c(Cl)cccc1C(Cl)(Cl)C(C)C(N)=O` |
| best RDKit logP | 3.40022 |AbTJeEvz0YjW
| best RDKit absolute error | 0.40022 |
| top-10 RDKit absolute error mean | 0.87468 |
| diversity among unique valid molecules | 0.89878 |

Interpretation:

The corrected GA confirms that the predictor can be optimized, but the available ZINC PS-VAE checkpoint cannot currently support a strong paper-grade molecular generation result. The bottleneck is not the GA hyperparameter setting; it is the decoder/generative checkpoint and predictor-decoder mismatch.

## Correct Workflow from Here

### Step 1. Treat current result as diagnostic only

Do not report old `success_rate_final` as molecular success. It is predicted latent success.

Report only these as main molecular metrics:

```text
decode_validity_raw_final
archive_unique_valid_molecules
archive_rdkit_success_unique
best_rdkit_logP
best_rdkit_abs_error
top10_rdkit_abs_error_mean
diversity_unique_valid
```

### Step 2. Fix the generative checkpoint before final claims

The next required experiment is to train or continue a better ZINC PS-VAE. The acceptance gate is:

```text
random training latent decode validity >= 60%
preferably >= 80%
```

Only after passing this gate should the logP GA migration experiment be promoted to the main paper result.

### Step 3. Re-run predictor and GA after decoder passes

Recommended final command template:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/molclr_pyg28/bin/python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/03b_optimize_logp_latent_ga_decode_aware.py \
  --zinc_psvae_ckpt <new_zinc_psvae_checkpoint.ckpt> \
  --predictor_ckpt <logp_predictor_trained_on_new_latents.pt> \
  --latent_pool <new_train_latent_pool.npy> \
  --init_mode train_random \
  --pop_size 100 \
  --n_gen 50 \
  --elite_size 20 \
  --decode_topk_per_gen 50 \
  --patience 10 \
  --immigrant_ratio 0.20 \
  --target_logp 3.0 \
  --success_low 2.5 \
  --success_high 3.5 \
  --version final_decode_aware_seed42 \
  --gpu 0 \
  --seed 42
```

Repeat for at least three seeds and report mean +/- std.



### 2026-06-01 Phase 1C corrected PS-VAE checkpoint audit

Important correction:

The earlier low decoder validity was measured partly from latent `.npy` files and GA-optimized off-manifold latents. That is not a fair checkpoint audit. A fair PS-VAE checkpoint audit must use the same checkpoint to encode ZINC training SMILES and decode those freshly produced latents.

Created scripts:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/06_audit_psvae_reconstruction_validity.py
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/07_sweep_psvae_reconstruction_params.py
```

Accepted checkpoint:

```text
/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt
```

Baseline fair audit, 200 ZINC training molecules, single decode attempt:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/molclr_pyg28/bin/python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/06_audit_psvae_reconstruction_validity.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/psvae_audit/version8_n200_a1 \
  --n_samples 200 \
  --attempts 1 \
  --max_atom_num 80 \
  --add_edge_th 0.55 \
  --temperature 0.5 \
  --gpu 0 \
  --seed 42
```

Result:

| Metric | Value |
|---|---:|
| encode rate | 100% |
| single-attempt decode validity | 75.0% |
| main error type | aromatic/kekulize failures |

Decode parameter sweep:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/molclr_pyg28/bin/python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/07_sweep_psvae_reconstruction_params.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/psvae_audit/version8_param_sweep_n80_a1 \
  --n_samples 80 \
  --attempts 1 \
  --max_atom_nums 80,100 \
  --add_edge_ths 0.30,0.45,0.55,0.65 \
  --temperatures 0.20,0.30,0.50,0.70 \
  --gpu 0 \
  --seed 42
```

Best small-sweep setting:

| max_atom_num | add_edge_th | temperature | single-attempt validity |
|---:|---:|---:|---:|
| 80 | 0.45 | 0.30 | 87.5% |

Larger 200-sample validation of the best setting:

| Decode setting | Attempts per latent | Single-attempt validity | Latent any-valid rate |
|---|---:|---:|---:|
| max_atom_num=80, add_edge_th=0.45, temperature=0.30 | 1 | 75.5% | 75.5% |
| max_atom_num=80, add_edge_th=0.45, temperature=0.30 | 3 | 76.17% | 85.0% |

Decision:

The current `version_8_zinc` checkpoint passes the required `>=60%` training-latent decode validity gate. It does not stably reach `>=80%` with a single decode attempt on 200 samples, but it reaches `85%` latent-level coverage with three decode attempts.

Therefore, do not spend the next step blindly retraining ZINC PS-VAE. The immediate paper-grade repair should be:

1. keep `version_8_zinc` as the accepted checkpoint for now;
2. use `max_atom_num=80`, `add_edge_th=0.45`, `temperature=0.30`;
3. decode each GA latent 3 times and keep the first RDKit-valid molecule;
4. report both single-attempt validity and latent any-valid coverage;
5. constrain GA mutations to stay near training latent manifold, because off-manifold GA latents are the real source of low final validity.

Saved accepted decode config:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/psvae_audit/zinc_psvae_decode_config.json
```



### 2026-06-01 Phase 3C RDKit-hybrid GA repair

Problem after checkpoint audit:

The predictor can push latent vectors to predicted logP near 3.0, but decoded molecules often have lower RDKit logP. Therefore, predictor-only GA gives misleading molecular results.

Code repair:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/03b_optimize_logp_latent_ga_decode_aware.py
```

Added:

1. `--decode_attempts_per_latent`: decode each latent multiple times and keep the first RDKit-valid molecule.
2. `--manifold_anchor_size` and `--manifold_blend`: pull offspring back toward ZINC training latent anchors.
3. `--selection_metric rdkit_hybrid`: decode the current population each generation and use decoded RDKit logP score in GA selection.
4. `--rdkit_selection_weight`: controls the weight of molecule-level RDKit score in selection.
5. `--pred_target_logp`: separates predictor-space target from molecule-level RDKit target.

Working command template:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/molclr_pyg28/bin/python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/03b_optimize_logp_latent_ga_decode_aware.py \
  --init_mode train_random \
  --pop_size 80 \
  --n_gen 12 \
  --elite_size 16 \
  --selection_metric rdkit_hybrid \
  --rdkit_selection_weight 0.85 \
  --decode_topk_per_gen 30 \
  --decode_attempts_per_latent 3 \
  --patience 8 \
  --immigrant_ratio 0.30 \
  --manifold_anchor_size 5000 \
  --manifold_blend 0.35 \
  --target_logp 3.0 \
  --pred_target_logp 3.0 \
  --score_sigma 0.7 \
  --success_low 2.5 \
  --success_high 3.5 \
  --max_atom_num 80 \
  --add_edge_th 0.45 \
  --temperature 0.30 \
  --version decode_aware_v4_rdkit_mid_seed42 \
  --gpu 0 \
  --seed 42
```

Three-seed result files:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/results/zinc_logp_decode_aware_v4_rdkit_mid_3seed_summary.csv
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/results/zinc_logp_decode_aware_v4_rdkit_mid_3seed_mean_std.csv
```

Three-seed summary:

| Seed | Final decode latent validity | Unique valid molecules | RDKit success unique | Best RDKit logP | Best RDKit abs error | Top-10 RDKit abs error | Diversity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 36.25% | 203 | 14 | 3.01842 | 0.01842 | 0.23531 | 0.90375 |
| 43 | 52.50% | 220 | 21 | 2.99340 | 0.00660 | 0.11879 | 0.91421 |
| 44 | 41.25% | 218 | 9 | 3.03542 | 0.03542 | 0.34355 | 0.90804 |

Mean +/- std:

| Metric | Mean | Std |
|---|---:|---:|
| final decode latent validity | 43.33% | 8.32% |
| unique valid molecules | 213.67 | 9.29 |
| RDKit success unique | 14.67 | 6.03 |
| RDKit success rate over unique | 6.86% | 2.71% |
| best RDKit abs error | 0.02015 | 0.01449 |
| top-10 RDKit abs error | 0.23255 | 0.11240 |
| diversity | 0.90867 | 0.00526 |

Interpretation:

This version is now scientifically usable as a ZINC logP migration experiment. It honestly reports molecule-level RDKit metrics, produces valid/diverse molecules, and repeatedly finds molecules very close to the target logP=3.0. The remaining weakness is final decode validity, which is still lower on GA-optimized latents than on freshly encoded training latents; this is expected because GA explores off-manifold regions. The current mitigation is manifold pullback plus RDKit-hybrid selection.



### 2026-06-01 Phase 3D root cause of low GA decoder rate

User concern:

The final GA decoder rate around 36-52% is much lower than the expected near-100% decoder rate from earlier experiments.

Direct audit result:

The current GA uses this latent pool:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/train/zinc_logp_latent.npy
```

This pool was generated by `01_encode_zinc_logp_latent.py` without `--kekulize`. Directly decoding 200 random latent vectors from this old pool with 3 attempts per latent gives:

| Latent source | Attempts per latent | Single-attempt validity | Latent any-valid rate |
|---|---:|---:|---:|
| old `Zinc_logP/train/zinc_logp_latent.npy` | 3 | 13.5% | 20.5% |

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/molclr_pyg28/bin/python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/08_decode_latent_pool_audit.py \
  --latent /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/train/zinc_logp_latent.npy \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/psvae_audit/old_latent_pool_n200_a3 \
  --n_samples 200 \
  --attempts 3 \
  --gpu 0 \
  --seed 42
```

By contrast, when a small ZINC sample is re-encoded with `--kekulize`, direct decoding is much healthier:

| Latent source | Attempts per latent | Single-attempt validity | Latent any-valid rate |
|---|---:|---:|---:|
| new `Zinc_logP_kek/train_sample_50/zinc_logp_latent.npy` | 3 | 71.33% | 82.0% |

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/molclr_pyg28/bin/python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/08_decode_latent_pool_audit.py \
  --latent /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek/train_sample_50/zinc_logp_latent.npy \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/psvae_audit/kek_latent_sample50_a3 \
  --n_samples 50 \
  --attempts 3 \
  --gpu 0 \
  --seed 42
```

Root cause:

The low GA decoder rate mainly comes from an inconsistent latent extraction pipeline. The old logP latent pool was encoded with `kekulize=False`, while the PS-VAE decoder and fair reconstruction audit behave much better when molecules are encoded in the kekulized representation. Therefore, the current logP predictor and GA are built on a weak/inconsistent latent pool.

Required correction:

1. Re-extract ZINC train/valid/test latent files with `--kekulize`.
2. Retrain the logP predictor on the new `Zinc_logP_kek` latent files.
3. Re-run RDKit-hybrid GA using the new latent pool and new predictor.
4. Only then compare final decoder rate to previous 100%-style results.

This should raise the initial latent-pool decode validity from about 20% any-valid to around 80% any-valid, based on the small-sample audit.



### 2026-06-01 Phase 3E full kekulize repair pipeline started

Created full repair script:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/run_zinc_logp_kek_fix.sh
```

Created result summarizer:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/summarize_zinc_logp_kek_results.py
```

Pipeline target directory:

```text
/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek
```

Pipeline steps:

1. Re-extract ZINC train/valid/test latents with `--kekulize`.
2. Audit new train latent-pool decoder validity with 200 random samples and 3 decode attempts.
3. Retrain logP predictor on the new `Zinc_logP_kek` latent files.
4. Re-run RDKit-hybrid GA for seeds 42, 43, 44.
5. Summarize three-seed results into CSV files.

Started persistent tmux session:

```bash
tmux new-session -d -s zinc_logp_kek_fix 'cd /root/autodl-tmp/sweeteners_evolve && GPU=0 bash /root/autodl-tmp/sweeteners_evolve/Gen_Exp/run_zinc_logp_kek_fix.sh > /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek/logs/pipeline.log 2>&1'
```

Monitor commands:

```bash
tmux ls

tail -f /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek/logs/pipeline.log

tail -f /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek/logs/encode_train_kek.log
```

Current status at start:

```text
tmux session: zinc_logp_kek_fix
current step: encode_train_kek
```
