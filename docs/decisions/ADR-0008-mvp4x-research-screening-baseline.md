# ADR-0008: MVP-4X Research Screening Baseline

Status: proposed research-only

Scope: research_only, exploratory_only, weak_label_target, no_final_labels,
no_ground_truth_claim, no_production_claim, not_validated_for_deployment.

## Context

MVP-4X RP consolidated a bounded screening/ranking signal. The evidence supports
manual-review prioritization inside selected exploratory cohorts, while R2 and
calibration remain insufficient for absolute channel-fraction prediction.

## Decision

Freeze a research-only screening baseline:

- target: receiver_mean
- model: Ridge
- feature set: existing_features_only
- use: anomaly ranking, candidate screening, manual-review prioritization

Supported screening cohorts:

- pooled_bc_all
- pooled_bc_high_orientation
- regime_b_high_orientation
- regime_c_all
- regime_c_high_orientation

Unsupported or audit-only cohorts:

- Regime A
- regime_b_all
- low-orientation intervals outside supported cohorts

## Boundaries

The baseline is not an absolute fraction predictor, not a production model, not
validated for deployment, and not validated outside supported cohorts. It does
not approve a production regime split, formal orientation filter, final labels,
ground-truth claim, cross-well generalization, STC, APES, or deep learning.

## Consequences

The next allowed step is offline leakage-safe out-of-fold scoring and a targeted
human review pack. Any formal regime policy, target-view switch, morphology
label redesign, advanced signal processing, final labels, or deployment claim
requires explicit human approval.
