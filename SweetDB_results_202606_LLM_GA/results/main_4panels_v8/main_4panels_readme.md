# v8 main four-panel figure set

Recommended four chart panels:

1. `main_v8_cumulative_hard_pass_evolution_llm_marked.png`
2. `main_v8_top5_logsw_evolution_llm_marked.png` as supplementary/mechanistic potency curve, not the sole success metric
3. `main_v8_final_logsw_vs_predicted_vina_surrogate.png` until real Vina scores are backfilled
4. `main_v8_llm_feedback_intervention_effect.png`

Top-5 molecule SVGs are in `top5_molecule_svgs/`.
Molecules are selected from the v8 final external pool by hard-pass first, then final_score, then predicted logSw, with duplicate SMILES removed within each method.
