# TrialGPT Failure Investigation

## Research Objective
Identify historical GPT-4 eligibility classification disagreements in the TrialGPT criterion-annotation dataset, develop a small failure taxonomy, and investigate whether selected failure mechanisms persist in a newer language model.

## Unit of Analysis
One patient–trial eligibility criterion annotation.

## Reference Standard
Physician-consensus eligibility labels provided in the TrialGPT dataset.

## Initial Exploratory Metric
Criterion-level disagreement between historical GPT-4 predictions and physician-consensus labels.

## Data Quality Requirements
Before calculating evaluation metrics, we verify:

    - Required fields and valid eligibility labels
    - Missing values and annotation identifiers
    - Duplicate annotation identifiers
    - Dataset source and revision
    - Number of annotations and distinct patients

## Statistical Considerations

Multiple criterion annotations may originate from the same patient. Descriptive counts will be reported at the criterion level, while inferential analyses will account for patient-level clustering.

## Discovery vs. Confirmation

The historical error analysis is exploratory. Mechanism hypotheses, controls, and primary outcomes will be specified before confirmatory experiments.

## Current Limitations

The initial structural audit does not establish the clinical correctness of expert labels. Disagreements must undergo further review before being classified as clinically meaningful model failures.



### Exploratory finding: Applicability-label confusion

**Source:** TrialGPT historical criterion annotations.

**Observed pattern:** Of 129 GPT-4/expert label disagreements, 48 concern exclusion criteria labeled `not excluded` by experts but `not applicable` by GPT-4.

**Case 44:** GPT-4 correctly identifies a patient aged 58 as not meeting an under-18 exclusion criterion, but predicts `not applicable` rather than the expert reference `not excluded`.

**Cases 108 and 109:** GPT-4 uses absence of documented hemodynamic instability or septic shock to justify `not applicable`. Both annotations belong to the same patient; the notes do not independently establish absence of these conditions.

**Candidate mechanisms:**
- H1: Incorrect mapping of a correctly understood fact to the eligibility-label schema.
- H2: Confusion between missing clinical evidence and criterion non-applicability.

**Status:** Exploratory observations.

**Next experiment:** Controlled age-based counterfactuals followed by separate clinical-evidence perturbations, with expected labels and primary outcomes specified before evaluation.

**Limitation:** The published TrialGPT study already identifies exclusion-label ambiguity, so a novel contribution requires testing persistence and causal mechanisms in a newer model.

The age-based experiment contains three prespecified synthetic perturbations. The 58-year-old and 16-year-old cases serve as primary age-threshold controls. The missing-age case is retained as an exploratory ambiguity probe pending confirmation of its expected classification under the original TrialGPT rubric. The original and clarified label instructions will be evaluated separately.

### Finding F1 — Prompt-sensitive exclusion applicability classification

**Model:** `gpt-5-mini-2025-08-07`

**Case:** Synthetic patient aged 58 years; exclusion criterion: participant is under 18 years old.

**Method:** Five repeated calls under each of two prompt conditions, with alternating order and the same requested model and JSON response format.

**Results:**
- `clarified_v1`: 5/5 `not excluded` (reference-aligned).
- `trialgpt_adapted_v1`: 5/5 `not applicable` (reference-discordant).
- All 10 responses passed structural validation.

**Mechanistic observation:** Under the adapted prompt, explanations consistently identify the patient's age correctly but treat the unmet age threshold as an absent applicability premise.

**Interpretation:** Evidence of reproducible prompt-conditioned label-boundary confusion on this single synthetic case.

**Limitations:** One unique case; adapted rather than exact historical prompt; no observed failure on the full original annotation 44.

**Next test:** Removing only the applicability-first instruction from the adapted prompt, preserving all other input and model settings, and measure whether the label changes.




### F1 — Applicability-first instruction ablation

**Research question:** Does the initial applicability-check instruction contribute to incorrect exclusion-label selection?

**Model:** `gpt-5-mini-2025-08-07`

**Input:** Synthetic patient aged 58; exclusion criterion: participant is under 18 years old.

**Conditions:**
- `trialgpt_adapted_v1`: applicability-first instruction retained.
- `trialgpt_no_applicability_v1`: only that instruction removed.

**Procedure:** Five repetitions per condition, alternating condition order. Same model snapshot and response format.

**Observed results:**
- Adapted: 5/5 `not applicable`; 0/5 reference-aligned.
- Ablated: 5/5 `not excluded`; 5/5 reference-aligned.
- All outputs passed structural validation.

**Mechanism evidence:** Explanations under the adapted prompt treated a false age condition as an absent applicability premise. Ablated-prompt explanations instead treated the age as direct evidence that the exclusion was not met.

**Interpretation:** The removed instruction contributed to a reproducible label-mapping disagreement on this input.

**Limitations:** Single synthetic case, five responses per condition, adapted rather than exact historical instructions, no independent-case generalization or clinical-outcome assessment.

**Next experiment:** Testing the adapted and ablated prompts across prespecified exclusion criteria covering multiple clinical attributes (smoking stauts, diabetes, history of surgery). Include a genuinely inapplicable criterion to assess possible tradeoffs.

### F2 — Unsupported clinical inference from missing documentation

**Model:** `gpt-5-mini-2025-08-07`

**Clinical question:** Whether the supplied patient note establishes septic shock.

**Method:** Three synthetic evidence conditions (explicitly present, explicitly absent, and unknown), evaluated three times each under a neutral prompt and a prompt containing a TrialGPT-inspired missing-information inference instruction.

**Observed findings:**
- Explicit positive: 3/3 `present` under each prompt.
- Explicit negative: 3/3 `absent` under each prompt.
- Unknown evidence: neutral prompt produced 3/3 `unknown`; inference prompt produced 3/3 `absent`.
- All 18 responses passed structural validation.

**Candidate mechanism:** The inference instruction encourages treating the absence of documentation of a clinically significant condition as evidence that the condition is absent.

**Potential clinical consequence:** A falsely definitive negative assessment could reduce attention to a potentially important exclusion or safety condition in trial screening.

**Limitations:** One unique unknown-evidence synthetic case, explicit wording about missing assessments, three calls per condition, and adapted rather than exact historical instructions.

**Next test:** Evaluate new naturalistic unknown-evidence cases and assess whether the inference-driven overcommitment persists.
