# SweetDB LLM-Guided Latent GA Application

This directory contains the SweetDB / sweetener application experiments for LLM-guided latent evolutionary optimization. It is a curated export of the 2026-06 SweetDB runs, including the main four-method experiment, the fitness ablation, real docking evaluation, and paper-ready figure panels.

## Scope

The PS-VAE backbone is not duplicated here because the repository already contains the shared PS-VAE code under `QM9_test/PS-VAE/`. The scripts in this folder add the SweetDB-specific latent GA, LLM seed/reflection intervention, gated sweetness fitness, docking-aware evaluation, and figure generation logic.

Large local-only artifacts are intentionally excluded: raw datasets, model checkpoints, latent `.npy` arrays, full generation histories, docking work folders, API keys, and molecule archives.

## Four Search Methods

The main SweetDB benchmark uses the same PS-VAE latent space and population budget across four groups:

| Group | Method | Initial population | LLM feedback during GA |
| --- | --- | --- | --- |
| A | Random-Seeded Latent GA | Random latent samples | No |
| B | SweetDB-Seeded Latent GA | High-potency SweetDB molecules | No |
| C | LLM-Initialized Latent GA | LLM-generated sweetener-like SMILES encoded into latent space | No |
| D | Iterative LLM-Guided Latent GA | Same LLM seed protocol as C | Yes, reflection-based reinjection when stagnation/diversity criteria are triggered |

## Fitness And Gold-Standard Evaluation

The final selected fitness route is the gated sweetness fitness:

1. A latent classifier estimates whether a candidate is sweetener-like.
2. Only candidates passing the sweet-likeness gate are mainly optimized by the latent sweetness regressor.
3. Validity, re-encoding consistency, uniqueness, and OOD distance are tracked as reliability constraints.
4. Real Vina docking is used as an external evaluation layer, not as the main GA fitness in the final SweetDB route.

The paper panels use a unified external gold-standard endpoint so that methods are not evaluated by their own internal fitness:

- `P(sweet) >= 0.80`
- `predicted logSw >= 2.60` after re-encoding
- `real Vina score <= -6.8 kcal/mol`

## Reproduction Entry Points

Main four-method SweetDB run:

```bash
cd /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1
bash run_abcd_v8_hard_metrics.sh
```

Fitness ablation:

```bash
cd /home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/train_1
bash run_sweet_fitness_ablation_v1.sh
bash run_gold_standard_no_fusion_v6.sh
```

Real docking evaluation and merge:

```bash
python build_abcd_docking_audit.py
python prepare_v8_docking_submission.py
python batch_docking_parallel_safe.py
python merge_v8_real_vina_and_plot.py
```

Final figure generation:

```bash
python plot_fig4_ab_scatter_consistent.py
python plot_v8_main_three_panels.py
python replot_v8_clean_main_panels.py
python draw_v8_top5_by_gold_standard.py
```

Paths inside the scripts may need to be updated for a new machine because the original experiments were run under `/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/`.

## Directory Layout

- `scripts/`: SweetDB GA, fitness ablation, docking evaluation, and plotting scripts.
- `results/fig4_panels/`: final Fig. 4 panels for gold-standard fitness ablation and candidate yield.
- `results/main_experiment/`: evolution curves, LLM feedback intervention panel, and main summary tables.
- `results/fitness_ablation/`: four-fitness ablation tables and gold-standard evaluation results.
- `results/docking/`: merged real Vina docking outputs used for final external evaluation.
- `results/top5_molecules/`: SVG/PNG molecule panels for top-5 candidates per method.
- `dataset_metadata/`: local dataset provenance and excluded data notes.

## Figure 4 Title

Recommended paper title:

**Fig. 4. SweetDB application: gold-standard evaluation of LLM-guided latent GA for sweetener candidate discovery.**

Recommended caption skeleton:

(a) Fitness ablation under the gold-standard endpoint defined by sweet-likeness, predicted sweetness, and real Vina docking. (b) Gold-standard candidate yield of four latent GA search strategies. (c) Evolution of top-5 predicted logSw during latent-space optimization. (d) LLM feedback intervention during iterative GA search. (e) Sweet taste receptor binding pocket and representative gold-standard candidate molecules.

## V8 Main Experiment (ABCD, Hard Metrics)

The reported v8 main experiment lives under `results/main_experiment/`. It is the four-method comparison with the **hard-metrics gated fitness** used as the final SweetDB route. Each method was repeated with five seeds (2026, 2027, 2028, 2029, 2030) over 12 generations, population size 30.

Final v8 fitness function (from `summary.json`):

```python
ga_fitness = pred_logSw
           + 0.50 * P_sweet
           - 0.80 * max(0, 0.80 - P_sweet)        # hard P(sweet) >= 0.80 gate
           - 0.50 * max(0, D_OOD - 7.29) / 7.29  # OOD distance penalty (p95 = 7.29)
```

Other v8 hyper-parameters:

- `pop_size = 30`, `n_gen = 12`, `elite_size = 3`
- `cross_prob = 0.38`, `mut_prob = 0.14`, `mut_sigma = 0.22`
- `lambda_ood = 0.35`, `lambda_reg_uncertainty = 0.25`, `lambda_desc = 0.5`, `lambda_llm = 0.1`
- `logsw_score_cap = 3.5`
- `objective = "constrained_sweetness"`
- `final_scoring = "decode -> strict re-encode -> re-score -> descriptor filter -> final_score"`

Four method groups:

| Group | init_mode | seed_source | online LLM |
| --- | --- | --- | --- |
| A | `group_a_random` | OOD background random | no |
| B | `group_b_dataset` | high-potency SweetDB molecules | no |
| C | `group_c_llm` | 30 LLM-generated, decontaminated, then BPE-encoded | no |
| D | `group_d_llm_iterative` | same as C | **yes, reflective injection** |

### Per-seed raw outputs (results/main_experiment/per_seed_runs/ABCD_v8_hard_metrics/)

Layout: `ABCD_v8_hard_metrics/{A,B,C,D}_seed{2026..2030}/`

Each run directory contains 11 small files (per run ~110 KB):

- `summary.json`: full config + final metrics (`best_final_score`, `mean_final_score`, `reliable_candidate_count`, `best_record_so_far`, etc.)
- `config.json`: frozen run config snapshot.
- `progress.csv`: per-generation (12 rows) curves of `mean_score`, `best_score`, `mean_p_sweet`, `mean_pred_logsw`, `mean_d_ood`, `success_count`, `no_improve`.
- `progress_metrics.csv`: alias of `progress.csv` from an earlier dump.
- `best_candidates_over_time.csv`: per-generation best candidate SMILES and metrics.
- `reliable_candidates.csv`: candidates that pass the final hard gate (P_sweet >= 0.80, logSw >= 2.60, re-encoding-consistent).
- `unique_reliable_candidates.csv`: deduped reliable candidates.
- `top_candidates.csv` / `top50_candidates_corrected.csv`: top-K final ranking.
- `final_population_corrected.csv`: full final 30-individual population.
- `topk_archive.csv`: top-K archive across generations.

### LLM online injection details (D group only)

LLM is invoked only in the D-group runs. Across all 5 seeds, injection is **deterministic at generations 3, 6, 9** (every 3 generations, before the 12-generation budget ends), with the same acceptance rule:

```
locally_refined candidate_score >= incumbent_score + min_gain
  AND population quantile floor
  AND novelty >= min_novelty
```

For each trigger:

- The top-3 current individuals (SMILES, score, P_sweet, logSw, D_OOD) are sent to the LLM as reflection context.
- The LLM returns proposed SMILES; `strict_bpe_injected_count` records how many pass the strict BPE re-encoding check.
- Locally in latent space: 3 refinement steps × 32 neighbors (sigma=0.1, decay=1.0), filtered by `min_novelty=0.08`.
- Up to `llm_inject_size=10` accepted candidates replace the worst individuals in the population.

Per-trigger injection summaries are recorded in two places:

- `summary.json.llm_online_injections`: list of `{generation, generated_basic_gate_count, strict_bpe_injected_count, replace_indices, acceptance_rule}`.
- `results/llm_online_injection_logs/D_seed{2026..2030}/gen_{003,006,009}_local_refinement.csv`: per-candidate table with `seed_idx, initial_score, refined_score, score_gain, novelty_to_population, p_sweet, pred_logsw, d_ood`.

Observed behavior in the reported runs: generations 3 and 6 each accept 1 refined candidate (`strict_bpe_injected_count=1`, `replace_indices=[12]` then `[24]`); generation 9 accepts 0 in all seeds.

## V6B Fitness Ablation (runs_v6b_gold_no_fusion_fast)

This is the fitness-route ablation, run under the **gold-standard evaluation endpoint** (P_sweet >= 0.80, logSw >= 2.60, Vina <= -6.8), without fusion.

Four fitness modes × 5 seeds = 20 runs. Layout:

`results/fitness_ablation/per_seed_runs/V6B_gold_no_fusion/{Sweet_Predictor|Docking_Predictor|Gated_Sweet|Gated_Docking}_seed{2026..2030}/`

Each run ships the same 11 small files as the v8 main experiment. Top-level summary tables are kept at `results/fitness_ablation/`:

- `ablation_group_summary.csv`, `ablation_progress_all.csv`, `ablation_run_metrics.csv`: cross-group aggregates.
- `fitness_ablation_overview.png`: paper-ready ablation overview.
- `fitness_gold_summary.csv`, `fitness_gold_by_seed.csv`: gold-standard pass counts.
- `gold_standard_summary.csv`, `gold_standard_by_seed.csv`, `gold_standard_per_molecule.csv`, `gold_standard_candidate_pool.csv`: per-molecule gold evaluation.
- `gold_threshold_sensitivity.csv`: threshold sensitivity sweep.
- `gold_standard_fitness_selection_v6b.svg` / `_highres.png`: the Fig. 4(a) panel source.
- `supplementary_fitness_ablation_table.csv`: supplementary material table.

## Nature-style fitness ablation panels (results/nature_style_panels/)

Five Nature-style figures for the four-fitness ablation (Sweet / Docking / Gated-Sweet / Gated-Docking) on the v5_vfdsurrogate 4-mode run, also covering the top-5 candidate yield, uniqueness, diversity, predicted logSw and docking score endpoints. Each figure is shipped in both SVG (vector) and `_highres.png` (raster).

- `A_final_unique_reliable_yield.{svg,png}`: reliable candidate yield across the four fitness modes.
- `B_near_target_population_evolution.{svg,png}`: per-generation evolution of population mass near the sweet-target region.
- `C_final_uniqueness_and_diversity.{svg,png}`: final population uniqueness and intra-population diversity.
- `D_reliable_candidate_predicted_logsw.{svg,png}`: predicted logSw distribution of reliable candidates.
- `E_reliable_candidate_docking_score.{svg,png}`: predicted docking score of reliable candidates.
- `four_fitness_progress_all.csv`, `four_fitness_run_metrics.csv`: per-generation and per-run metrics behind the panels.
