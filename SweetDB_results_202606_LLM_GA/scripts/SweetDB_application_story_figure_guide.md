# SweetDB application figure guide

## Central claim

The LLM is not presented as a replacement for the latent regressor or as an
oracle that must beat a supervised high-potency seed set on every potency
metric. Its role is a label-efficient prior and an adaptive search controller.
With zero measured high-potency initialization labels, iterative LLM guidance
approaches the potency obtained by a 20-label supervised seed budget while
returning more unique reliable and novel-scaffold candidates.

All results in this pilot comparison use seed 2026, population size 30, and
12 generations. Publication claims should be confirmed with repeated seeds.

## Search pipeline

1. The PS-VAE encoder maps random, measured SweetDB, or LLM-proposed SMILES to
   the shared latent manifold.
2. A sweet-likeness classifier first gates candidates.
3. The latent regressor guides potency optimization only after the gate.
4. The GA applies selection, crossover, mutation, strict decoding, and
   decode-reencode reliability checks.
5. Group D periodically reflects on stagnation, duplicates, failures, and OOD
   patterns, then injects new LLM proposals into the weakest population slots.

## Individual panels

- **A: Predicted logSw evolution.** Shows optimization of the direct task
  objective over 12 generations.
- **B: Near-target population.** Counts how many of the current 30 individuals
  satisfy the shared sweet-probability, potency, and OOD criteria.
- **C: Reliable logSw distribution.** Compares only unique candidates that
  survive strict decode-reencode and reliability filtering.
- **D: Shared latent UMAP.** Projects every method and all generations jointly
  against a fixed training-manifold background. Lines connect generation
  medians and describe search trajectories rather than molecular reactions.
- **E: Top-five molecules.** Provides directly inspectable representatives from
  each method under the same reliability rules.
- **F: Label budget versus potency.** Group D uses zero measured high-potency
  seeds yet reaches Top-10 predicted logSw 3.710, close to B with 20 labels
  (3.751), and above B with 10 labels (3.534).
- **G: Label budget versus reliable candidates.** The discrete blue points are
  single controlled runs, not a fitted monotonic learning curve. D returns 22
  reliable candidates, versus 17 for B with 30 measured seeds.
- **H: Label budget versus search speed.** D reaches 24/30 near-target
  individuals by generation 6; C needs generation 8. `NR` means the threshold
  was not reached in 12 generations.
- **I: Similarity to labelled training molecules.** Maximum Morgan Tanimoto
  similarity measures proximity to any labelled training molecule.
- **J: Novel-scaffold composition.** D returns 19 unique reliable candidates,
  including 17 with scaffolds absent from the labelled training folds.
- **K: Novelty-potency frontier.** Shows the trade-off between structural
  novelty and reencoded predicted potency instead of reducing quality to one
  fitness score.
- **L: Novel-scaffold molecules.** Displays representative reliable structures
  selected jointly for predicted potency and structural novelty.

## Suggested presentation order

1. Use A and B to establish that the methods optimize the same latent task.
2. Use F-H to make the label-efficiency argument.
3. Use D to show where the four searches move on the shared manifold.
4. Use I-K to establish structural generalization and candidate quality.
5. End with E and L, then transition to docking as the external validation.

## Important wording

Use: “Iterative LLM guidance supplies a label-efficient molecular prior and
recovers diversity when latent GA search stagnates.”

Avoid: “The LLM predicts true sweetness better than the supervised oracle.”
The present evidence concerns search efficiency, candidate reliability,
diversity, and scaffold novelty under a learned surrogate.
