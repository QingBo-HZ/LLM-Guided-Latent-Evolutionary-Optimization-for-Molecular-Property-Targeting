# LLM-Guided Latent Evolutionary Optimization for Molecular Property Targeting

This repository contains the code and curated result exports for LLM-guided molecular property optimization in latent space. The project combines PS-VAE molecular representations, latent-space genetic algorithms, LLM-based seed generation, iterative LLM feedback, and property-specific evaluation pipelines.

## Project Scope

The project is organized around three datasets / application settings:

| Dataset / task | Purpose | Current server status | Repository location |
| --- | --- | --- | --- |
| QM9 | Main target-property optimization benchmark, focused on molecular property targeting in PS-VAE latent space. | Present on this server. | `Main_results_202604_LLM_GA/`, `Ablation_1/`, `QM9_test/PS-VAE/` |
| ZINC / logP | Transfer experiment for ZINC logP optimization and PS-VAE reconstruction / decoding audits. | Present on this server. | `Gen_Exp/`, `QM9_test/PS-VAE/` |
| SweetDB / sweeteners | Sweetener discovery application with SweetDB seeds, gated sweetness fitness, docking evaluation, and final figure exports. | Run on a separate machine; the curated export has already been uploaded to GitHub. | `SweetDB_results_202606_LLM_GA/` |

Only the QM9 and ZINC workflows are expected to be runnable from the current server layout. SweetDB paths in the exported scripts may point to the original remote machine and should be treated as provenance / reproduction notes unless they are adapted to a new environment.

## Method Overview

The optimization pipeline uses a shared latent evolutionary framework:

1. Encode molecules into PS-VAE latent representations.
2. Build property predictors or scoring functions for the target task.
3. Initialize candidate populations from random sampling, dataset molecules, LLM-generated SMILES, or hybrid strategies.
4. Run latent-space genetic search with mutation, crossover, elitism, and task-specific validity checks.
5. Optionally inject LLM feedback when the search stagnates or loses diversity.
6. Decode, re-encode, rescore, and summarize final candidates for evaluation and figures.

## Repository Layout

- `Main_results_202604_LLM_GA/`: QM9 main experiments comparing random search, SMILES GA, latent GA, LLM-initialized latent GA, and iterative LLM-guided latent GA.
- `Ablation_1/`: QM9 ablations for seed strategy, population size, LLM rounds, and optimization variants.
- `Gen_Exp/`: ZINC/logP transfer experiments, latent logP predictor training, decode-aware GA, PS-VAE audits, and plotting utilities.
- `QM9_test/PS-VAE/`: shared PS-VAE code, training scripts, data preparation scripts, and property-prediction utilities used by QM9 and ZINC.
- `SweetDB_results_202606_LLM_GA/`: curated SweetDB application export from the separate sweetener machine, including scripts, summary tables, docking-derived evaluation files, and paper-ready figure panels.

## Data And Artifact Policy

Large raw datasets, trained checkpoints, latent arrays, model caches, full generated histories, docking work folders, logs, and API keys are intentionally excluded from version control unless they are small curated outputs required to reproduce a table or figure.

For this server, the active local tasks are QM9 and ZINC/logP. SweetDB data and training artifacts should not be regenerated or copied into this machine-specific checkout unless a separate migration is planned.

## Entry Points

QM9 main experiments:

```bash
cd Main_results_202604_LLM_GA
# See pipeline.md for the full command list.
```

QM9 ablations:

```bash
cd Ablation_1
# See run scripts and result summaries in this directory.
```

ZINC/logP transfer:

```bash
cd Gen_Exp
# See pipline.md and Zinc_logP* subdirectories for encoding, predictor training, and GA runs.
```

SweetDB application export:

```bash
cd SweetDB_results_202606_LLM_GA
# See README.md. These scripts document the separate-machine SweetDB runs.
```

## Notes

- Local absolute paths in older scripts reflect the original experiment server layout and may need editing before rerunning.
- Secret-bearing configuration files should stay out of Git. API credentials must be provided through local environment variables or machine-local config files.
- The SweetDB export is part of the project record, but this server should be treated as the QM9/ZINC working environment.
