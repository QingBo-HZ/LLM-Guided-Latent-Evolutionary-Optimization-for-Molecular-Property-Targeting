# LLM-Guided-Latent-Evolutionary-Optimization-for-Molecular-Property-Targeting

LLM-guided latent-space genetic optimization for molecular property targeting,
with a real-world sweetener discovery application on the SweetDB / SweetSpaceDB
datasets.

This repository develops a single framework that combines:

- a **PS-VAE** latent molecular representation
- a **latent-space genetic algorithm** (latent GA)
- **LLM-generated seeds** and **LLM reflection / re-injection** during evolution
- a **gated sweet-fitness** with external gold-standard evaluation
  (latent sweet-likeness, predicted sweetness, real Vina docking)

and validates it on three progressively more realistic settings: a QM9
target-property benchmark, a ZINC logP transfer experiment, and finally the
**SweetDB sweetener discovery application**, which is the reported target of
this work and the basis of the BIBM 2026 submission.

## Application Target: SweetDB / SweetSpaceDB Sweetener Discovery

The reported application is **de novo sweetener design on SweetDB /
SweetSpaceDB**. This is a small-data, noisy-label, application-driven task where
the objective is not just to maximize a single scalar property, but to generate
chemically valid and latent-space-consistent molecules that are simultaneously
sweet-like, predicted to have high sweetness, and plausible under
receptor-level docking evaluation.

The central claim of the SweetDB application is:

> LLM-generated sweetener-like seeds provide a practical way to initialize
> latent-space evolutionary search in a low-data sweetener design task. A
> gated sweet-fitness strategy gives a better balance between sweet-likeness
> and docking-supported molecular plausibility than using a sweetness
> regressor or docking score alone. The iterative LLM-guided version further
> shows that feedback from stagnating generations can be injected back into
> the latent GA, producing measurable post-intervention improvements in
> top-5 predicted logSw, on top of an already strong one-shot LLM
> initialization baseline.

Four search strategies are compared (full details in
`SweetDB_results_202606_LLM_GA/`):

- **A — Random-Seeded Latent GA** (lower-bound baseline)
- **B — SweetDB-Seeded Latent GA** (known high-potency SweetDB molecules)
- **C — LLM-Initialized Latent GA** (LLM-generated seeds, no online LLM)
- **D — Iterative LLM-Guided Latent GA** (LLM seeds + reflection / re-injection
  at predefined generations 3, 6, 9)

Fitness is a hard-metrics gated sweet-fitness with P(sweet) >= 0.80 as an
explicit gate, OOD distance penalty, and a final decode -> strict re-encode ->
re-score pipeline. Real Vina docking is used as an **external gold-standard
endpoint** (P(sweet) >= 0.80, predicted logSw >= 2.60, real Vina <= -6.8
kcal/mol), not as the internal GA fitness, to avoid circular evaluation.

## Contents

- `Main_results_202604_LLM_GA/`: main QM9 target-property experiments, used as
  the framework's QM9-level validation. This is where the latent GA,
  LLM-init, and hybrid pipelines are first prototyped.
- `Ablation_1/`: seed strategy and iterative guidance ablation experiments
  on the QM9 / ZINC backbone. Establishes the role of LLM seeds and online
  reflection independently of the application.
- `Gen_Exp/`: ZINC logP transfer and PS-VAE audit utilities. Used to check
  reconstruction validity, decoder sweeps, and the logP transferability of
  the latent space before applying it to the sweetener task.
- `QM9_test/PS-VAE/`: shared PS-VAE model code, training, and evaluation
  scripts. This is the latent representation used by all experiments above,
  including the SweetDB application.
- `SweetDB_results_202606_LLM_GA/`: **BIBM 2026 submission main result
  package**. Contains the SweetDB four-method (A/B/C/D) main experiment,
  the four-mode fitness ablation (Sweet-only / Docking-only / Gate+Sweet /
  Gate+Docking), the gold-standard external evaluation tables, real Vina
  docking merges, the per-seed GA raw outputs, the LLM online injection
  logs, and the paper-ready figure panels (Fig. 4 a–e and the Nature-style
  fitness ablation panels). The PS-VAE backbone is not duplicated here;
  scripts in this folder consume the shared PS-VAE under
  `QM9_test/PS-VAE/`.

## How The Pieces Fit Together

```
QM9 / ZINC (Main_results_202604_LLM_GA, Gen_Exp, Ablation_1)
        framework validation
                |
                v
   QM9_test/PS-VAE/  (shared PS-VAE backbone)
                |
                v
   SweetDB_results_202606_LLM_GA/  (BIBM 2026 submission)
        - four-method latent GA on SweetDB (A/B/C/D)
        - gated sweet-fitness ablation
        - external gold-standard evaluation with real Vina
        - paper-ready figure panels
```

The QM9 / ZINC results are the framework-level evidence that LLM
initialization and online LLM reflection work in a controlled benchmark
setting. The SweetDB directory is where that framework is applied to a real
application task (sweetener discovery) and is the basis of the BIBM 2026
submission.

## Reproducing The SweetDB / BIBM 2026 Result

The SweetDB experiments expect a local working environment with a PS-VAE
checkpoint, a SweetDB CSV, a docking surrogate, and (for live runs) LLM API
access. All scripts in `SweetDB_results_202606_LLM_GA/scripts/` are
documented at the top of the directory's own README. Large local-only
artifacts (raw datasets, checkpoints, latent `.npy` arrays, full generation
histories, docking work folders, API keys, molecule archives) are
intentionally excluded from version control.

## License

See `LICENSE`.
