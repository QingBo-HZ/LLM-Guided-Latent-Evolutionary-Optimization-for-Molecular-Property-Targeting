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
