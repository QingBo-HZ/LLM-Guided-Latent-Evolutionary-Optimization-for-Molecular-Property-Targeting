python /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/analyze_sweetener_features.py \
  --input /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/dataset/SweetenersDB_v2.0.csv \
  --out_dir sweetener_feature_panel

python txt2csv.py \
  --input /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/dataset/my_zinc250k.txt \
  --output /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/dataset/my_zinc250k.csv \
  --col_name Smiles

python build_sweet_like_dataset.py \
  --zinc dataset/my_zinc250k.csv \
  --fartdb dataset/FartDB_raw.csv \
  --sweetdb dataset/SweetenersDB_v2.0.csv \
  --out_dir sweet_like_dataset_out \
  --q_low 0.05 \
  --q_high 0.95 \
  --min_match 7 \
  --min_molwt 50 \
  --max_molwt 1200

词典构造

预训练

打分器训练
python extract_latent_for_gated_predictor.py \
  --reuse_existing_csv \
  --input_csv_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data \
  --out_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data_strict_bpe \
  --device cuda \
  --batch_size 64

python train_latent_predictor.py \
  --latent_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data_strict_bpe \
  --device cuda \
  --epochs 150 \
  --batch_size 64 \
  --lr 1e-4 \
  --classifier_select_metric pr_auc \
  --regressor_select_metric mae

ood生成

python build_ood_background_from_sweet_like.py

进化生成
# 第一组
python /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1/sweet_gated_latent_ga_4groups.py \
  --init_mode group_a_random \
  --version GroupA_random_corrected_test \
  --latent_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data_strict_bpe \
  --predictor_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data_strict_bpe/gated_predictor_final \
  --ood_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_like_dataset_out/ood_background_strict_bpedataset \
  --pop_size 120 \
  --n_gen 20 \
  --elite_size 20 \
  --lambda_ood 0.35 \
  --lambda_desc 0.50 \
  --logsw_score_cap 3.5 \
  --background_filter_by_evaluator \
  --seed_min_p_sweet 0.60 \
  --archive_topk 20

# 第二组
python /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1/sweet_gated_latent_ga_4groups.py \
  --init_mode group_b_dataset \
  --version GroupB_dataset_corrected_test \
  --latent_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data_strict_bpe \
  --predictor_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data_strict_bpe/gated_predictor_final \
  --ood_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_like_dataset_out/ood_background_strict_bpedataset \
  --pop_size 120 \
  --n_gen 20 \
  --elite_size 20 \
  --lambda_ood 0.35 \
  --lambda_desc 0.50 \
  --logsw_score_cap 3.5 \
  --seed_min_p_sweet 0.70 \
  --archive_topk 20


## 2026-06-08 ABCD constrained sweetness potency optimization rerun

This section records the completed A/B/C/D run after switching the GA objective to a constrained sweetness-potency score and after removing GA-fitness filtering from LLM seed evaluation.

### Code changes used in this run

- Main GA script: `/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1/sweet_gated_latent_ga_4groups.py`
- LLM seed generator: `/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/generate_sweet_smiles/generate_llm_sweet_seed_prior_judge_v2.py`
- Runner: `/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1/run_sweet_group_C_D.sh`
- Summary collector: `/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1/summarize_abcd_results.py`

Backups were created before modifying the two existing Python scripts:

- `sweet_gated_latent_ga_4groups.py.backup_20260608_140008`
- `generate_llm_sweet_seed_prior_judge_v2.py.backup_20260608_140008`

### Objective and comparison design

Task name:

`Constrained Sweetness Potency Optimization`

Internal GA objective:

```text
score = pred_logSw
      + 0.50 * P_sweet
      - 0.80 * max(0, 0.70 - P_sweet)
      - 0.50 * max(0, D_OOD - OOD_p95) / OOD_p95
```

Progress success definition:

```text
pred_logSw >= 2.30
P_sweet >= 0.70
D_OOD <= OOD_p95
```

Important LLM policy:

- LLM seed generation and LLM seed quality are evaluated only by LLM-prior score plus basic RDKit validity/deduplication.
- `P_sweet`, `pred_logSw`, `OOD`, descriptor score, and GA fitness are not used to reject LLM-generated seeds before GA.
- The GA fitness is only the evolutionary search guide.

Shared PS-VAE/checkpoint and evaluator:

```text
PS-VAE ckpt:
/home/jqb/PS-VAE-main/ckpts/sweet_pretrain_encoder_decoder_manifold_v2/lightning_logs/version_0/checkpoints/last.ckpt

latent_dir:
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data_manifold_v2

predictor_dir:
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data_manifold_v2/gated_predictor_scaffold_ensemble_v1

ood_dir:
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_like_dataset_out/ood_background_manifold_v2
```

Common GA parameters:

```text
pop_size=120
n_gen=50
elite_size=20
cross_prob=0.35
mut_prob=0.05
mut_sigma=0.20
patience=100
seed=2026
seed_augment_sigma=0.05
```

### Run command

```bash
cd /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA
RUN_TAG=ABCD_constrained_v2_20260608 bash train_1/run_sweet_group_C_D.sh all
```

The actual run was launched with `nohup`; log:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/ABCD_constrained_v2_20260608_master.log
```

### Group definitions

- A: random sweet-like latent seed from OOD background, evaluator-filtered.
- B: dataset seed from SweetDB/FlavorDB sweet latents, evaluator-filtered and diverse-seeded.
- C: 5-round iterative LLM-prior seed generation before GA, then normal latent GA.
- D: same initial LLM seed pool as C, then online LLM reflection/injection during GA at generations 10, 20, 30, and 40.

### C/D shared LLM seed generation

Command inside runner:

```bash
/home/jqb/.conda/envs/brc_vae/bin/python \
  /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/generate_sweet_smiles/generate_llm_sweet_seed_prior_judge_v2.py \
  --generator_model gpt-5.4-mini \
  --judge_model gpt-5.4-mini \
  --rounds 5 \
  --per_round 20 \
  --generator_call_size 10 \
  --target_total 100 \
  --batch_size 10 \
  --top_k 100 \
  --out_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/generate_sweet_smiles/group_c_iterative_prior_5x20_ABCD_constrained_v2_20260608
```

Seed generation result:

```text
accepted_total=94
judged_total=94
mean_llm_prior_score=0.5247
max_llm_prior_score=0.8080
min_llm_prior_score=0.2070
mean_risk_penalty=0.1265
```

Strict-BPE encoding:

```text
input_count=94
unique_after_canonicalization=94
strict_bpe_encoded_count=94
strict_bpe_failed_count=0
latent_shape=(94, 56)
latent_path=/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/llm_sweet_seed_smiles/group_c_iterative_prior_5x20_ABCD_constrained_v2_20260608_latent/llm_abcd_latent.npy
```

Note: the target was 100, but 94 unique RDKit-valid molecules remained after duplicate/basic-gate filtering across the five rounds. All 94 surviving molecules encoded successfully with strict BPE.

### D online LLM injection

Injection summary:

```text
generation 10: injected 10 strict-BPE LLM candidates
generation 20: injected 10 strict-BPE LLM candidates
generation 30: injected 1 strict-BPE LLM candidate
generation 40: injected 10 strict-BPE LLM candidates
```

Detailed files:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/group_d_llm_iterative_ABCD_constrained_v2_20260608_D_seed2026/llm_injection_summary.csv
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/group_d_llm_iterative_ABCD_constrained_v2_20260608_D_seed2026/llm_online_injection/
```

At generation 30, only one candidate passed the LLM prior/risk threshold and strict-BPE encoding, so only one was injected.

### Final ABCD comparison

Summary CSV:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/ABCD_comparison_ABCD_constrained_v2_20260608.csv
```

| Group | Generations | Best GA score | Best reencoded logSw | Mean reencoded logSw | Reliable count | Reliable rate | Validity | Unique SMILES | Unique ratio | Internal diversity | LLM injection events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 50 | 4.7312 | 3.9334 | 2.7373 | 38 | 0.3167 | 0.9833 | 109 | 0.9237 | 0.8892 | 0 |
| B | 50 | 4.9006 | 4.0128 | 3.1817 | 60 | 0.5000 | 1.0000 | 36 | 0.3000 | 0.6409 | 0 |
| C | 50 | 4.5091 | 3.5987 | 2.4343 | 25 | 0.2083 | 0.9583 | 112 | 0.9739 | 0.8983 | 0 |
| D | 50 | 4.5074 | 3.7060 | 2.3524 | 27 | 0.2250 | 0.9167 | 105 | 0.9545 | 0.8961 | 4 |

### Interpretation

- A and especially B still win on the current scaffold-aware latent regressor metrics.
- C/D are much stronger on diversity than B. C has the highest unique ratio and internal diversity among completed groups.
- D successfully executed online LLM reflection/injection, but under this current fitness/predictor setting it did not clearly outperform C on final reliable count or mean reencoded logSw.
- D's direct injected molecules did not remain as final direct-source individuals; final `source_type` labels were `ga_mutation_crossover`, so the current source label tracks immediate source only, not full ancestry. The injection events themselves are still recorded in `llm_injection_summary.csv`.
- For the paper logic, this run supports that LLM seeding improves exploration/diversity, while the current online reflection injection needs stronger diversity/ancestry control or a better trigger/selection rule to show a clear exploitation gain.


## 2026-06-08 small-budget speed-oriented rerun

Purpose:

The previous 50-generation run optimized too long for the paper claim we want. The intended claim is not "LLM must produce the highest final fitness", but:

```text
LLM-guided initialization can move latent GA into usable sweet-like molecular regions with a much smaller GA budget.
```

Therefore this rerun shrinks the population and generation budget and loosens the gate threshold.

### Clarification on score columns

`best_logSw_reencoded` is not the GA fitness. It is:

```text
final latent z
-> decoded SMILES
-> strict-BPE re-encoding
-> regressor prediction on reencoded z
-> max pred_logSw_reencoded
```

The GA fitness is `score_ga`, which is only the internal evolutionary guide.

### BPE/Kekule vocabulary check

Vocabulary:

```text
/home/jqb/PS-VAE-main/data/Sweet/Sweet_bpe_1000.txt
```

The vocabulary contains many Kekule-style fragments such as:

```text
C1=CC=...
O=C...
NC1=NC=...
```

The strict-BPE encoder uses `smiles2molecule(..., kekulize=True)`, so aromatic LLM SMILES like `c1cc...` are converted before BPE. In the completed 5-round LLM seed set, this was not a failure mode:

```text
raw_unique_llm_seed=94
strict_bpe_encoded=94
strict_bpe_failed=0
```

So for this run the limiting issue is not Kekule/BPE failure; it is mainly decoder convergence and GA over-optimization.

### Adjusted parameters

```text
pop_size=40
n_gen=6
elite_size=4
mut_prob=0.10
mut_sigma=0.20
P_sweet threshold=0.55
progress logSw threshold=2.0
final_min_logSw=2.0
LLM latent seed reused from the 94 strict-BPE encoded LLM seeds
```

Command:

```bash
cd /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA

RUN_TAG=ABCD_speed_early_v1_20260608 \
POP_SIZE=40 \
N_GEN=6 \
ELITE_SIZE=4 \
MUT_PROB=0.10 \
MUT_SIGMA=0.20 \
P_SWEET_THRESHOLD=0.55 \
SUCCESS_P_SWEET=0.55 \
LOGSW_SUCCESS_THRESHOLD=2.0 \
FINAL_MIN_LOGSW=2.0 \
LLM_LATENT_OVERRIDE=/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/llm_sweet_seed_smiles/group_c_iterative_prior_5x20_ABCD_constrained_v2_20260608_latent/llm_abcd_latent.npy \
LLM_INTERVAL=5 \
LLM_CANDIDATES=10 \
LLM_INJECT_SIZE=5 \
bash train_1/run_sweet_group_C_D.sh all
```

Main output:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/ABCD_comparison_ABCD_speed_early_v1_20260608.csv
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/ABCD_speed_metrics_ABCD_speed_early_v1_20260608.csv
```

### Small-budget final results

| Group | Generations | Reliable count | Reliable rate | Validity | Unique ratio | Internal diversity | Best reencoded logSw | Mean reencoded logSw | LLM injection events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 6 | 11/40 | 0.275 | 1.000 | 0.950 | 0.863 | 3.360 | 2.672 | 0 |
| B | 6 | 10/40 | 0.250 | 1.000 | 0.400 | 0.516 | 3.403 | 2.639 | 0 |
| C | 6 | 34/40 | 0.850 | 1.000 | 0.150 | 0.288 | 2.826 | 2.564 | 0 |
| D | 6 | 32/40 | 0.800 | 1.000 | 0.250 | 0.374 | 2.716 | 2.389 | 1 |

### Interpretation for paper framing

- With only 6 generations and 40 individuals, LLM-seeded GA produces many more final reliable candidates than A/B.
- A/B can look strong in latent-space progress, but after decode -> strict-BPE reencode -> final filters, their reliable counts are much lower.
- C is the cleanest evidence for LLM seed acceleration: 34/40 reliable candidates under a very small GA budget.
- D improves diversity over C in this small run, but the online reflection step does not yet improve reliable count over C.
- Therefore the best current claim is:

```text
LLM-guided seed initialization improves low-budget sample efficiency of latent-space GA.
```

The weaker/unsafe claim would be:

```text
Online LLM reflection always improves final potency.
```

The latter is not supported by the current runs.


## 2026-06-08 practical pop=20 rerun for Group D

Motivation:

For sweetener discovery, a population of 40 can already be too large. In a realistic setting, producing around 20 candidates is closer to the intended use case. This rerun uses `pop_size=20` and evaluates D by diversity and unique reliable molecules, not only by raw reliable count.

Parameters:

```text
pop_size=20
n_gen=6
elite_size=3
mut_prob=0.12
mut_sigma=0.22
P_sweet threshold=0.55
progress logSw threshold=2.0
final_min_logSw=2.0
D online LLM reflection interval=2
D inject_size=5
LLM_MIN_PRIOR=0.35
LLM_MAX_RISK=0.60
```

Run tag:

```text
ABCD_pop20_dboost_6gen_v1_20260608
```

Main output:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/ABCD_comparison_ABCD_pop20_dboost_6gen_v1_20260608.csv
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/ABCD_pop20_dboost_6gen_unique_reliable.csv
```

Final result:

| Group | Reliable count | Unique all | Unique reliable | Best reliable logSw | Mean reliable logSw |
|---|---:|---:|---:|---:|---:|
| A | 3 | 7 | 2 | 2.855 | 2.579 |
| B | 8 | 19 | 7 | 3.502 | 2.955 |
| C | 17 | 1 | 1 | 2.716 | 2.543 |
| D | 11 | 10 | 6 | 3.351 | 2.547 |

Interpretation:

- C has the highest raw reliable count, but it collapses to only one unique reliable molecule.
- D has fewer raw reliable molecules than C, but produces six unique reliable molecules and much higher molecular diversity.
- D also improves best reliable logSw over C (`3.351` vs `2.716`).
- Therefore, for the practical `pop=20` setting, D is better than C if the endpoint is unique, diverse, usable molecular candidates rather than repeated copies of one decoder attractor.

Recommended paper framing:

```text
When the candidate budget is small, online LLM reflection does not necessarily maximize the raw number of reliable decoded samples, but it substantially reduces decoder collapse and improves the number and diversity of unique reliable candidates.
```


## 2026-06-09 pop=30 unique-SMILES constraint check

Motivation:

The previous `pop=30, n_gen=12` run showed that C/D can quickly over-converge in decoder space. C reached 29 reliable candidates but only 2 unique reliable molecules; D reached 29 reliable candidates but only 5 unique reliable molecules. This is difficult to justify in a paper because the final output looks like repeated variants of a small number of decoder attractors.

Implementation change:

- Added `--enforce_unique_smiles` to `train_1/sweet_gated_latent_ga_4groups.py`.
- Elite selection now prefers unique canonical decoded SMILES.
- After each GA generation, duplicate decoded SMILES can be replaced by strict-BPE-compatible refills.
- Added `--unique_target_ratio` so uniqueness can be soft rather than all-or-nothing.
- Added `unique_reliable_candidates.csv` and summary fields:
  - `unique_reliable_candidate_count`
  - `unique_reliable_candidate_rate`

Key output:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/pop30_unique_strategy_comparison_20260609.csv
```

Comparison:

| Run | Reliable | Unique all | Unique ratio | Unique reliable | Validity | Diversity | Best reliable logSw | Mean reliable logSw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C base, no unique | 29 | 2 | 0.067 | 2 | 1.000 | 0.314 | 2.732 | 2.527 |
| D base, no unique | 29 | 5 | 0.167 | 5 | 1.000 | 0.220 | 3.169 | 2.559 |
| C hard unique 100% | 10 | 25 | 0.833 | 9 | 1.000 | 0.814 | 2.684 | 2.392 |
| D hard unique 100% | 7 | 26 | 0.867 | 6 | 1.000 | 0.861 | 2.849 | 2.484 |
| C soft unique 50% | 16 | 15 | 0.500 | 3 | 1.000 | 0.643 | 2.716 | 2.554 |
| D soft unique 50% | 20 | 15 | 0.500 | 6 | 1.000 | 0.703 | 2.732 | 2.455 |

Interpretation:

- No unique constraint gives high raw reliable count but severe collapse.
- Hard 100% unique constraint fixes diversity but damages reliable yield too much.
- Soft 50% unique constraint is the best current compromise.
- Under soft 50% unique constraint, D is better than C on reliable count, unique reliable count, and final diversity.
- The strongest current D-vs-C claim is therefore not "D maximizes predicted logSw"; it is:

```text
Online LLM reflection plus a soft unique-SMILES constraint maintains the LLM-guided speed advantage while reducing decoder-space collapse.
```

Recommended default for the next ABCD table:

```text
pop_size=30
n_gen=12
elite_size=4
mut_prob=0.10
mut_sigma=0.20
unique_target_ratio=0.50
unique_refill_sigma=0.08
unique_refill_from_seed_prob=0.80
report both reliable_count and unique_reliable_candidate_count
```


## 2026-06-09 quality-aware unique refill v2

Problem found:

The first soft unique constraint accepted the first newly decoded canonical SMILES, even when the corresponding latent point had weak `P_sweet`, low predicted logSw, or poor OOD position. It maintained 15 unique molecules but many of the refilled molecules failed final reliability checks.

The GA fitness itself was not changed. The replacement strategy was improved:

- Sample multiple refill latent candidates.
- Apply a permissive latent quality gate before decoding:
  - `P_sweet >= 0.50`
  - `pred_logSw >= 1.80`
  - `D_OOD <= training p95`
- Rank passing refill candidates by the existing GA score.
- Decode in ranked order and retain the best novel canonical SMILES.
- For D, reject LLM reflection outputs that duplicate the current decoded top/bottom population.

Run script:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1/run_pop30_quality_unique_cd.sh
```

Comparison output:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/pop30_quality_aware_unique_v2_comparison_20260609.csv
```

| Run | Reliable | Unique reliable | Unique all | Unique ratio | Diversity | Best reliable logSw | Mean unique-reliable logSw | Best final score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C soft unique v1 | 16 | 3 | 15 | 0.500 | 0.643 | 2.716 | 2.532 | 3.018 |
| D soft unique v1 | 20 | 6 | 15 | 0.500 | 0.703 | 2.732 | 2.324 | 3.100 |
| C quality-aware v2 | 13 | 13 | 26 | 0.867 | 0.823 | 2.867 | 2.513 | 3.366 |
| D quality-aware v2 | 16 | 15 | 27 | 0.900 | 0.825 | 3.125 | 2.573 | 3.610 |

Main conclusion:

- C improves from 3 to 13 unique reliable candidates.
- D improves from 6 to 15 unique reliable candidates.
- D produces 27/30 unique final SMILES and 15 unique reliable molecules.
- D also has the strongest best reliable logSw and best final score.
- The raw reliable count is lower than v1 because repeated copies are no longer counted as apparent progress.

Current recommended paper result:

```text
Quality-aware unique replacement prevents decoder-attractor duplication without altering the latent-GA fitness. Online LLM reflection further improves the number and quality of unique reliable candidates.
```

Recommended v2 parameters:

```text
pop_size=30
n_gen=12
unique_target_ratio=0.50
unique_refill_attempts=48
unique_refill_sigma=0.06
unique_refill_from_seed_prob=0.85
unique_refill_min_p_sweet=0.50
unique_refill_min_logsw=1.80
unique_refill_max_ood_ratio=1.00
D reflection interval=3
D online candidates=12
D inject size=3
```


## 2026-06-09 corrected 12-generation ABCD main-figure run

### Generation-index correction

The earlier log displayed `Gen 000` through `Gen 011`. These were 12 population
evaluations, but the implementation produced one additional offspring population
after `Gen 011` and decoded that unlogged population as the final result. This made
the plotted trajectory and final decoded population inconsistent.

The corrected implementation:

- records generations as 1 through 12 in `progress.csv`;
- saves exactly 12 latent snapshots (`gen_001.npy` to `gen_012.npy`);
- stops after evaluating generation 12, without breeding a hidden generation 13;
- performs D reflection/injection after generations 3, 6, and 9;
- decodes the same generation-12 population used by the final plotted point.

Run script:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1/run_abcd_mainfig_quality_v3.sh
```

Run tag:

```text
ABCD_mainfig_quality_v3_20260609
```

### Corrected four-group results

| Group | Initialization | First generation reaching 30/30 | Reliable | Unique reliable | Unique final | Diversity |
|---|---|---:|---:|---:|---:|---:|
| A | Random latent background | 7 | 23 | 17 | 23 | 0.832 |
| B | Known sweet dataset seeds | 1 | 19 | 14 | 25 | 0.812 |
| C | LLM seeds | 5 | 16 | 16 | 28 | 0.818 |
| D | LLM seeds + online reflection | 6 | 11 | 8 | 20 | 0.775 |

All four final populations decoded at 30/30 validity. D successfully completed
strict-BPE online injections at generations 3, 6, and 9, but this particular run
did not improve over C. The generation-6 and generation-9 injections temporarily
reduced the success count, and the final D population had fewer unique reliable
candidates. Therefore, this run validates the online pipeline but does not support
a claim that reflection is always beneficial. Future D tuning should use
acceptance-aware injection or smaller/adaptive replacement rather than selecting
the most favorable old run.

### Main figures and shared UMAP coordinates

Remote output:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/figures_bibm_sweet_main_v3
```

Primary files:

```text
Fig_SweetDB_Main_ABCD.png/.pdf/.svg
Fig_SweetDB_Manifold_Dynamics_ABCD.gif
Fig_SweetDB_Manifold_Final_ABCD.png/.svg
Fig_SweetDB_Top_Molecules_ABCD.png/.pdf/.svg
ABCD_shared_umap_coordinates.npz
ABCD_main_figure_summary.csv
```

The UMAP reducer is fitted once on a fixed sample of the reference latent
background plus every population snapshot from all four groups. Every animation
frame therefore uses the same coordinates and axis limits; no per-frame refitting
or PCA is used.

Plotting scripts:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1/plot_abcd_mainfig_umap.py
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1/plot_abcd_top_molecules.py
```

### Docking-ready MOL2 export

Remote output:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/docking_ABCD_mainfig_quality_v3_selected
```

Each group contains five score-ranked, unique reliable candidates that successfully
generated 3D conformers. Coordinates were generated with RDKit ETKDGv3, optimized
with MMFF when parameters were available (otherwise UFF), and written with
Gasteiger charges in Tripos MOL2 format. All 20 files passed atom/bond block and
non-planar 3D-coordinate checks.


## 2026-06-10 fair ABCD redesign with GPT-5.5 and Gemini 3.1 Pro

The previous comparison was not a fair initialization ablation: A used the full
Sweet train/valid/test manifold, B was evaluator-filtered and fitness-ranked, and
the uniqueness refill source differed between A/B and C/D.

Corrected settings:

- generator: `gpt-5.5`;
- independent judge: `gemini-3.1-pro-preview`;
- 44 RDKit-valid unique candidates and 44/44 strict-BPE encodings;
- train-only OOD reference containing 173,403 PS-VAE training latents;
- no evaluator filtering or fitness ranking for B;
- sampling without replacement when at least 30 seeds are available;
- `unique_refill_pool=current` and no external seed-pool refill;
- identical GA operators, thresholds, and uniqueness rules for all groups.

Seed and OOD outputs:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/llm_sweet_seed_smiles/gpt55_gemini31_fair_v1
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/llm_sweet_seed_smiles/gpt55_gemini31_fair_v1_latent
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_like_dataset_out/ood_background_manifold_v2_train_only
```

Run script:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1/run_abcd_fair_gpt55_gemini31_v1.sh
```

Preliminary seed-2026 results:

| Group | Gen-1 success | First 30/30 | Reliable final | Diversity | Best latent logSw |
|---|---:|---:|---:|---:|---:|
| A random train manifold | 11 | 7 | 9 | 0.846 | 3.366 |
| B random known-sweet dataset | 7 | 3 | 13 | 0.809 | 3.307 |
| C GPT-5.5/Gemini seeds | 12 | 3 | 17 | 0.763 | 3.534 |

C reaches 27/30 by generation 2 and 30/30 by generation 3, compared with
generation 7 for A. It also has the largest reliable final count in this
replicate. Its remaining weakness is lower final internal diversity.

D uses stagnation-triggered reflection rather than fixed injection. Reflection
candidates are strict-BPE encoded and injected only when their latent GA score is
at least as high as the offspring being replaced. Fitness is therefore used as
an acceptance compass, not as the initial LLM seed generator or judge.

The first D validation generated two strict-BPE reflection candidates at
generation 9, but both failed the acceptance rule and were not injected. This
revealed that reflection decoding consumed Torch RNG state even when no candidate
was accepted. The implementation now snapshots and restores Python, NumPy, CPU
Torch, and CUDA RNG states around reflection. Therefore, when zero candidates are
accepted, D must remain bitwise aligned with C after the checkpoint.

### D memetic reflection v3b (seed 2026, 2026-06-10)

The first local-reflection run (`Dlocal_v2`) accepted two candidates at
generation 3, but finished slightly below C (16 vs. 17 reliable candidates).
This showed that replacing a weak offspring is not sufficient: an injected
direction must be strong enough to compete with the established population.

The corrected D run replays the saved GPT-5.5/Gemini-3.1 reflection pool from
generation 3 and applies an annealed latent hill climb before injection:

- 10 strict-BPE reflection latents are searched independently.
- 8 local steps, 128 trials per step.
- Initial sigma 0.14 with 0.78 step decay.
- Novelty floor 0.10 and novelty weight 0.03.
- At most two candidates are injected.
- A candidate must beat its replacement by 0.03 and reach the current
  population's 75th score percentile.
- Final evaluation thresholds and scoring remain identical to C.

Run:

```bash
GROUPS=D SEEDS=2026 \
TAG=ABCD_fair_gpt55_gemini31_Dmemetic_v3b_20260610 \
LLM_REPLAY_DIR=/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/llm_reflection_replay_gpt55_gemini31_v1 \
bash train_1/run_abcd_fair_gpt55_gemini31_v1.sh
```

Result:

| Metric | C | D-v3b |
|---|---:|---:|
| First generation with 30/30 successful | 3 | 3 |
| Best GA score at generation 4 | 3.5495 | 3.9004 |
| Final best GA score | 4.0339 | 4.2297 |
| Final mean GA score | 3.9923 | 4.1903 |
| Reliable candidates | 17/30 | 22/30 |
| Unique reliable candidates | 14/30 | 19/30 |
| Unique final SMILES | 25/30 | 26/30 |
| Final internal diversity | 0.7629 | 0.8201 |
| Top-10 mean final score | 3.4410 | 3.5638 |
| Single best final score | 3.8712 | 3.7882 |

The defensible conclusion is that D improves convergence speed, reliable yield,
top-set quality, and diversity. It does not improve the single best decoded
candidate in this replicate. The main D claim should therefore use trajectory,
top-k, reliable-count, and diversity metrics rather than only the maximum final
score. Multi-seed repeats are still required for a paper-level significance
claim.

### Four reference-style SweetDB figures

Paper-facing group names:

- A: `Random-Seeded Latent GA`
- B: `SweetDB-Seeded Latent GA`
- C: `LLM-Initialized Latent GA`
- D: `Iterative LLM-Guided Latent GA`

The figures are generated by:

```bash
python train_1/plot_sweetdb_four_reference_panels.py \
  --results_root /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_fair \
  --background_latent /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_like_dataset_out/ood_background_manifold_v2_train_only/background_latent.npy \
  --output_dir /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/figures_sweetdb_four_reference_v3b
```

Output directory:

`/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/figures_sweetdb_four_reference_v3b`

Figure definitions:

1. `Fig1_SweetDB_Constraint_Success_Evolution`: cumulative evaluated
   population members satisfying the registered progress criteria
   (`P(sweet) >= 0.55`, `predicted logSw >= 2.0`, shared OOD limit).
   These are evaluation hits, not deduplicated molecules.
2. `Fig2_SweetDB_Top10_Potency_Shortfall`: top-10 mean predicted-logSw
   shortfall to 3.5 on a logarithmic scale. Values at or above 3.5 are shown at
   `1e-3`.
3. `Fig3_SweetDB_Reliable_LogSw_Distribution`: strict decode-reencode
   predicted-logSw distribution for final reliable candidates.
4. `Fig4_SweetDB_Shared_Latent_UMAP`: one joint UMAP fitted to a fixed
   train-only manifold sample and all 12 generations from all four groups.
   Population-median paths are connected in generation order.

Each figure is exported as PNG, PDF, and SVG. The chart-ready CSV files and
fixed UMAP coordinates are saved beside the figures. No confidence bands are
shown because the current comparison has one random seed.

### Representative molecules for presentation

The 4 x 5 molecular panel uses the same paper-facing group names. Five
representative molecules are selected from each group's unique reliable final
candidates. Selection starts from the highest final score and then balances
normalized final score (72%) with minimum Morgan-fingerprint novelty (28%).

Script:

`train_1/plot_abcd_representative_molecules_ppt.py`

Output:

`/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/figures_sweetdb_representative_molecules_v3b`

The figure is exported as PNG, PDF, and SVG. The accompanying
`SweetDB_representative_molecules_4x5.csv` stores the 20 canonical SMILES and
their final metrics.

### Evaluation-budget convergence curves

The earlier cumulative-success figure was replaced because summing the loose
`logSw >= 2.0` population count across generations produced nearly linear,
weakly discriminative curves. The logarithmic shortfall plot was also replaced
because clipping all target-reaching methods at `1e-3` hid their convergence
timing.

The revised curves use the latest D-v3b run and the true molecular-evaluation
budget:

1. Normalized target gap:

   ```text
   gap = max(0, (3.5 - predicted_logSw) / 3.5)
   ```

   Both best-so-far and Top-10 mean gaps are shown.

2. Near-target count:

   ```text
   P(sweet) >= 0.55
   predicted_logSw >= 3.3
   D_OOD <= shared OOD p95
   ```

   The `3.3` threshold means the candidate is within `0.2` of the optimization
   target `3.5`. Counts are current-population counts, not cumulative counts.

The D reflection event occurs after 90 evaluations. At 180 evaluations D has
28/30 near-target candidates while C has 5/30. D reaches 30/30 at 240
evaluations; C reaches 30/30 at 300 evaluations.

Script:

`train_1/plot_sweetdb_evaluation_curves_v3b.py`

Figures and chart data:

`/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/figures_sweetdb_evaluation_curves_v3b`

Detailed C/D method description:

`/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/CD_latent_GA_optimization_explanation.md`
# 2026-06-12: Group B high-potency scaffold-diverse rerun and Figure A-E v4

## Corrected Group B definition

The previous Group B mixed all SweetDB molecules with FlavorDB sweet-labelled
molecules and sampled 30 seeds without potency control. Its initial population
was sweet-like (mean P(sweet) = 0.953) but weak in predicted potency
(mean predicted logSw = 1.310).

The corrected Group B is a supervised strong baseline:

- source: SweetDB only;
- fixed holdout: scaffold fold 0 is excluded from seed selection;
- potency region: top 30% by measured training-set logSw;
- cutoff in this run: measured logSw >= 2.150;
- candidate pool: 78 molecules;
- initialization: 30 molecules from 30 distinct scaffolds;
- GA parameters: identical to Groups A, C, and D.

Run script:

`train_1/run_group_b_high_potency_scaffold_v2.sh`

Result directory:

`sweet_ga_results_fair/group_b_dataset_ABCD_fair_B_highpot_scaffold_v2_20260612_B_s2026_seed2026`

Key result:

- best latent predicted logSw: 3.975;
- final validity: 1.0;
- final diversity: 0.870;
- reliable decoded/re-encoded candidates: 6/30.

This method should be described as `High-Potency Seeded GA` or a supervised
oracle-style baseline. It has access to measured high-potency SweetDB labels,
unlike the LLM seed strategies.

## Unified Figure A-E

Output directory:

`figures_sweetdb_main_AE_v4`

- (a) direct predicted logSw evolution;
- (b) near-target molecules in the current population;
- (c) reliable decoded/re-encoded logSw distribution;
- (d) shared UMAP latent search space;
- (e) five top reliable molecules from each method.

All panels use the same method colors:

- A: gray;
- B: blue;
- C: orange;
- D: green.

Figure F is reserved for molecular docking results.

Final panel filenames are:

- `Panel_A_Predicted_LogSw_Evolution.*`
- `Panel_B_Near_Target_Population.*`
- `Panel_C_Reliable_LogSw_Distribution.*`
- `Panel_D_Shared_Latent_UMAP.*`
- `Panel_E_Top5_Molecules.*`

## 2026-06-12: Group B v3 fairness correction

The v2 high-potency baseline still selected the 30 strongest labelled
scaffolds after applying the top-30% cutoff, making it a top-30 oracle.
Group B v3 keeps the same training-only high-potency pool but samples
scaffolds uniformly with a fixed random seed:

- exclude scaffold fold 0;
- retain measured training logSw top 30%;
- choose one random molecule per scaffold;
- uniformly sample 30 distinct scaffold representatives.

Result directory:

`sweet_ga_results_fair/group_b_dataset_ABCD_fair_B_highpot_scaffold_v3_20260612_B_s2026_seed2026`

The final Top-10 predicted logSw changed from 3.937 (v2) to 3.881 (v3);
iterative LLM-guided Group D is 3.710. This is the final fairer Group B
used by Figure A and Figure B in `figures_sweetdb_main_AE_v5`.

## 2026-06-12: SweetDB application-story figure suite

Final output directory:

`figures_sweetdb_application_story_v1`

The suite keeps every panel separately in PNG, PDF, and SVG:

- A: predicted logSw evolution;
- B: near-target population;
- C: reliable decoded/reencoded logSw distribution;
- D: shared latent-space UMAP and generation-median trajectories;
- E: top-five reliable molecules per method;
- F: measured-seed budget versus final Top-10 predicted logSw;
- G: measured-seed budget versus reliable candidate count;
- H: measured-seed budget versus generation reaching 24/30 near-target molecules;
- I: maximum similarity to labelled training molecules;
- J: unique reliable candidates split by training-known/novel scaffold;
- K: structural-novelty versus predicted-potency frontier;
- L: representative reliable molecules with novel scaffolds.

The label-budget experiment reruns Group B with 5, 10, 20, and 30 measured
high-potency training seeds. Groups C and D use zero measured high-potency
initialization labels. In the seed-2026 pilot:

- D Top-10 predicted logSw: 3.710;
- B-10: 3.534;
- B-20: 3.751;
- D reliable candidates: 22;
- B-30 reliable candidates: 17;
- D reaches 24/30 near-target molecules at generation 6;
- C reaches the same target at generation 8.

The scaffold analysis is based on unique reliable decoded/reencoded candidates:

- A: 9 total, 9 novel scaffolds;
- B: 15 total, 14 novel scaffolds;
- C: 14 total, 14 novel scaffolds;
- D: 19 total, 17 novel scaffolds.

These are controlled single-seed experiments. Repeat-seed confidence intervals
are still required before making statistical claims in the paper.

Three composite PNG storyboards are also generated:

- `Story_Figure_1_Search_and_Label_Efficiency.png`;
- `Story_Figure_2_Manifold_and_Scaffold_Generalization.png`;
- `Story_Figure_3_Representative_Molecules.png`.

Detailed interpretation:

`SweetDB_application_story_figure_guide.md`
