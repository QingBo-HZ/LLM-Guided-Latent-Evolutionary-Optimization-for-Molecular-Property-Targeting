# SweetDB Figures 0618: Result Logic

## 1. Regressor Choice

The original scaffold-aware ensemble regressor (`v1`) remains the best model by global OOF MAE:

- v1 MAE = 0.559, R2 = 0.587, Spearman = 0.724
- highsweet_v4 MAE = 0.563, R2 = 0.585, Spearman = 0.723

However, the GA does not need the lowest average error everywhere. It needs fewer exploitable false-high predictions and better behavior in the high-sweet region. Under the high-sweet threshold logSw >= 2.8:

- v1 high-sweet MAE = 1.270, high-sweet recall = 11.4%, false-high rate = 3.9%
- highsweet_v4 high-sweet MAE = 1.174, high-sweet recall = 20.0%, false-high rate = 1.1%

Therefore, the 0618 formal ABCD rerun uses `gated_predictor_scaffold_ensemble_highsweet_v4` with a conservative ensemble score:

```text
pred_logSw = ensemble_mean - 0.25 * ensemble_std
```

This is not a fused fitness. It is a conservative version of the latent sweet regressor.

## 2. Formal ABCD Rerun

Path:

```text
/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_0618_v4_fast
```

Protocol:

- 5 seeds: 2026-2030
- 12 generations
- population size = 30
- broad application threshold: P(sweet) >= 0.55 and pred_logSw >= 2.0
- final yield is counted after canonical SMILES de-duplication
- D uses replayed GPT-5.5/Gemini reflection latents, so the rerun is reproducible and does not depend on live API variability

Main formal ABCD summary:

| Group | Method | Unique reliable candidates/run | Unique ratio | Diversity | Mean best logSw |
|---|---|---:|---:|---:|---:|
| A | Random latent GA | 13.4 | 0.907 | 0.861 | 3.697 |
| B | SweetDB-seeded GA | 6.8 | 0.720 | 0.702 | 3.532 |
| C | LLM-initialized GA | 7.4 | 0.360 | 0.454 | 3.156 |
| D | Reflection-guided LLM GA | 10.6 | 0.667 | 0.746 | 3.329 |

Interpretation:

- D is clearly better than C: iterative reflection improves both unique reliable yield and population diversity.
- A remains strong under this broad application endpoint because random latent search samples from the sweet-like latent manifold, not from arbitrary chemistry.
- The defensible LLM claim is not "D beats every possible latent baseline"; it is "LLM reflection improves the LLM-initialized route and reduces the collapse seen in one-shot LLM initialization."

## 3. Fitness Ablation Logic

Gold-standard no-fusion fitness ablation path:

```text
/home/jqb/sweet_fitness_5seeds_panels/runs_v6b_gold_no_fusion_fast
```

Methods:

- Sweet-only: direct latent sweet regressor
- Docking-only: docking surrogate
- Gate+Sweet: sweet classifier gate + latent sweet regressor
- Gate+Docking: sweet classifier gate + docking surrogate

External evaluation avoids circular scoring:

1. Take top-10 candidates per seed.
2. Re-encode and re-score with the external evaluator.
3. Run real Vina docking for the selected molecules.
4. Gold endpoint requires high sweet and real Vina support.

Strict gold endpoint summary:

| Fitness | High-sweet | Vina-supported | Both/gold | Mean real Vina | Best real Vina |
|---|---:|---:|---:|---:|---:|
| Gate+Sweet | 1.0 | 4.0 | 0.6 | -7.45 | -8.48 |
| Gate+Docking | 0.4 | 9.2 | 0.4 | -7.90 | -9.24 |
| Sweet-only | 0.8 | 5.0 | 0.4 | -7.30 | -8.89 |
| Docking-only | 0.0 | 9.8 | 0.0 | -8.28 | -9.24 |

Conclusion:

- Docking-only is excellent for binding but fails the sweet endpoint.
- Gate+Docking preserves binding better than Gate+Sweet but still reduces high-sweet hits.
- Gate+Sweet is the best no-fusion choice for the SweetDB generative objective.
- Docking should be used as secondary validation or downstream selection, not as the primary GA fitness for sweetener discovery.

## 4. Recommended Story for Figure 4

Use the following visual order:

1. `formal_abcd_near_target_population.png`: broad endpoint convergence speed.
2. `formal_abcd_top10_logsw_evolution.png`: potency trajectory.
3. `formal_abcd_unique_reliable_yield.png`: final unique reliable yield.
4. `formal_abcd_uniqueness_diversity.png`: collapse/diversity evidence.
5. `formal_abcd_latent_umap_search_regions.png`: search region visualization.
6. `formal_abcd_representative_molecules.png`: examples for PPT/figure panel.
7. `fitness_ablation_gold_endpoint_counts.png`: objective-function ablation.
8. `fitness_ablation_sweet_docking_tradeoff.png`: sweet vs docking trade-off.
9. `regressor_highsweet_constraint_behavior.png`: why v4 is used for GA.

The clean final claim:

```text
The high-sweet constrained regressor is selected because it is less exploitable in the high-potency region. In the generative experiment, iterative LLM reflection improves the LLM-initialized latent GA by increasing unique reliable yield and preserving diversity. In the fitness ablation, the sweet-gated latent regressor remains the best primary no-fusion fitness, while docking is more suitable as an orthogonal validation endpoint.
```
