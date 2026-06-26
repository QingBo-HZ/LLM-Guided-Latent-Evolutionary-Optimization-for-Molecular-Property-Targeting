# figures_0618 index

Generated from v4 high-sweet constrained regressor for formal ABCD and v6b no-fusion gold-standard fitness ablation.

## Formal ABCD v4
- Predictor: `gated_predictor_scaffold_ensemble_highsweet_v4`
- Conservative ensemble score: `pred_logSw = mean - 0.25*std`
- GA: 12 generations, population size 30, seeds 2026-2030
- D uses replayed GPT-5.5/Gemini reflection latents at generation 9; no online API call in this rerun.
- Final plots use canonical SMILES de-duplication for yield metrics.

## Fitness ablation
- Four no-fusion modes: Sweet-only, Docking-only, Gate+Sweet, Gate+Docking.
- External gold endpoint: high sweet plus real Vina <= -7.0 on top-10 candidate pools.

## Files
- `fitness_ablation_gold_endpoint_counts.png`
- `fitness_ablation_real_vina_distribution.png`
- `fitness_ablation_sweet_docking_tradeoff.png`
- `formal_abcd_best_fitness_evolution.png`
- `formal_abcd_latent_umap_search_regions.png`
- `formal_abcd_near_target_population.png`
- `formal_abcd_representative_molecules.png`
- `formal_abcd_top10_logsw_evolution.png`
- `formal_abcd_unique_reliable_yield.png`
- `formal_abcd_uniqueness_diversity.png`
- `protocol_external_evaluation_flow.png`
- `reference_old_fair_abcd_sixpanel.png`
- `reference_old_panel_near_target.png`
- `reference_old_panel_predicted_logsw.png`
- `regressor_highsweet_constraint_behavior.png`
- `regressor_oof_mae_comparison.png`
