# SweetDB Fitness Ablation v2

## Experimental question

Does the proposed gated fitness guide latent-space GA more reliably than:

1. direct optimization of the latent logSw regressor; and
2. direct optimization of a molecular-docking score?

## Controlled setup

- Methods: Direct Regressor, Docking Surrogate, Proposed Gated.
- Seeds: 2026, 2027, 2028.
- Population: 30 latent vectors.
- Generations: 12.
- Same PS-VAE checkpoint, latent bounds, crossover, mutation, elitism, and
  unique-SMILES control.
- The three methods use byte-identical initial populations for each seed,
  verified by SHA-256 hashes.
- Only the scalar fitness used for GA selection differs.
- Every final population is decoded and evaluated with the same strict
  decode -> re-encode -> classifier/regressor/OOD/descriptor pipeline.

## Fitness definitions

Direct regressor:

`F_reg(z) = predicted_logSw(z)`

Docking surrogate:

`F_dock(z) = mean_m[-predicted_Vina_affinity_m(z)]`

Proposed gated fitness:

`F_gate(z) = predicted_logSw(z) + 0.50 P_sweet(z)
             - 0.80 max(0, 0.55 - P_sweet(z))
             - 0.50 max(0, D_OOD(z) - D_p95) / D_p95`

## How docking is used

The 316 SweetDB molecules have real Vina docking affinities. Their IDs were
joined to the corresponding 56-dimensional PS-VAE latent vectors. A five-model
MLP ensemble was trained with the existing scaffold-aware folds to predict
`-Vina affinity`, so a larger value is always better for GA maximization.

Scaffold-aware out-of-fold docking-surrogate performance:

- R2: 0.819
- Pearson: 0.905
- Spearman: 0.778
- MAE: 0.649 kcal/mol

During GA, all candidate latent vectors are evaluated by the ensemble and the
mean prediction is used as the docking fitness. This is a fast surrogate
screening step. For prospective use, the final shortlisted decoded molecules
must still be docked with Vina; the surrogate does not replace final physical
docking.

## Main result

| Method | Unique reliable candidates | Unique SMILES ratio | Internal diversity | Mean P(sweet) | Mean predicted logSw | Mean OOD distance |
|---|---:|---:|---:|---:|---:|---:|
| Direct Regressor | 6.67 +/- 2.89 | 0.844 | 0.838 | 0.871 | 2.737 | 6.353 |
| Docking Surrogate | 5.33 +/- 1.15 | 0.889 | 0.804 | 0.864 | 2.490 | 6.548 |
| Proposed Gated | **11.00 +/- 1.00** | **0.889** | **0.851** | **0.890** | **2.739** | **6.212** |

The proposed gate improves unique reliable yield by 65% relative to direct
regressor optimization and by 106% relative to direct docking optimization.
It also has the best mean sweet probability, the lowest mean OOD distance, and
the highest internal diversity among the two high-yield comparisons.

The trajectory explains why. Direct-regressor optimization increases predicted
logSw while often driving the latent population away from the sweet class.
Docking optimization successfully increases predicted binding affinity, but
binding affinity alone is not equivalent to sweetness potency. The gated
fitness aligns selection with all three requirements: sweet-class membership,
potency, and latent-manifold reliability.

## Recommended paper statement

Under identical initialization and GA budgets, the gated fitness produced
11.0 +/- 1.0 unique reliable candidates, compared with 6.7 +/- 2.9 for direct
logSw-regressor optimization and 5.3 +/- 1.2 for Vina-surrogate optimization.
This result indicates that optimizing a single proxy objective can cause
objective misalignment, whereas classifier-gated regression with an OOD
constraint yields a larger and more diverse set of sweet-like candidates.

This is a complete three-seed ablation. For the final manuscript, five seeds
and actual Vina re-docking of the final top candidates would strengthen the
statistical and physical validation.
