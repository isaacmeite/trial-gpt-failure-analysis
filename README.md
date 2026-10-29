# TrialGPT Failure Analysis

## Overview

This project investigates two failure mechanisms in language-model-based clinical trial eligibility assessment: confusion between exclusion eligibility labels and unsupported clinical inferences from missing documentation.

The analysis builds on the TrialGPT criterion-annotation dataset and evaluates adapted prompts using `gpt-5-mini-2025-08-07`. These experiments are not an exact reproduction of the original TrialGPT GPT-4 system.

## Research Findings

### F1 — Exclusion applicability-label confusion

In a frozen evaluation of 148 exclusion criteria from 10 patients, agreement with expert annotations increased from 79/148 (53.4%) under the adapted four-label prompt to 96/148 (64.9%) when `not applicable` was removed from the permitted output labels.

Among 105 cases with expert label `not excluded`, incorrect `not applicable` predictions declined from 6/105 (5.7%) to 2/105 (1.9%).

The intervention also reduced recall for genuinely `not applicable` cases, demonstrating a tradeoff rather than an unconditional improvement.

### F2 — Unsupported absence inference

Twenty unique synthetic clinical cases were evaluated under neutral and inference-oriented prompts: 12 with unknown evidence status, four explicitly positive controls, and four explicitly negative controls.

The neutral prompt incorrectly classified 0/12 unknown-evidence cases as `absent`; the inference prompt incorrectly classified 3/12 (25%) as `absent`. Both prompts correctly classified all eight explicit controls.

This finding characterizes behavior on a constructed stress test, not the error rate in real-world clinical practice.

## Dataset and Model

- Dataset: `ncbi/TrialGPT-Criterion-Annotations` on Hugging Face
- Dataset revision: `1cfcafde94a1560a33b4addc1638664fe28fc059`
- Experiment model: `gpt-5-mini-2025-08-07`
- Reference labels: expert annotations from the TrialGPT dataset

## Setup

Requires Python 3.12, `uv`, and an OpenAI API key for experiments requiring new model calls.

```bash
uv sync
```

For API evaluations, configure `OPENAI_API_KEY` in your local environment.

## Validation

```bash
uv run ruff check .
uv run pytest -q
```

## Reproduce Analyses

With the recorded prediction files available in `results/`:

```bash
uv run python -m alterra_trialgpt.analyze_f1
uv run python -m alterra_trialgpt.bootstrap_f1
uv run python -m alterra_trialgpt.analyze_f2
```

To rebuild the underlying model predictions, use the evaluation runners with the pinned dataset and the documented experiment protocols.

The raw F1 prediction file is not included in the repository's initial artifact set. Consequently, the F1 analysis cannot be rerun offline without that file or regenerated predictions. Aggregate F1 results are provided separately.

## Results

Selected output artifacts:

- `results/f1_analysis_v1.json`
- `results/f1_bootstrap_v1.json`
- `results/f2_case_manifest_v1.json`
- `results/f2_analysis_v1.json`
- `results/f2_evaluation_predictions_v1.jsonl`

The F1 study uses patient-cluster bootstrap intervals, while F2 is reported descriptively because of the small synthetic sample.

## Limitations

The F1 experiment uses adapted prompts and a small number of patient clusters. Expert labels may themselves have ambiguous applicability boundaries. The F2 evaluation uses researcher-authored synthetic cases and does not establish actual patient outcomes or real-world error prevalence.

## References

1. Jin Q, Wang Z, Floudas CS, et al. Matching patients to clinical trials with large language models. *Nature Communications*. 2024;15:9074. https://doi.org/10.1038/s41467-024-53081-z
2. NCBI. [TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations). Hugging Face dataset.