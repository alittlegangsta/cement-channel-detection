# ADR-0010: Retain v1 Screening and Audit Morphology-v2

Date: 2026-06-05

Scope: research_only, exploratory_only, weak_label_target, no_final_labels,
no_ground_truth_claim, no_production_claim, not_validated_for_deployment.

## Status

Accepted for MVP-4X research audit.

## Context

The current MVP-4X v1 research screening baseline uses existing XSI features only,
`receiver_mean` as the weak target view, Ridge as the classical model, and the
`triangular_midpoint_weighted` target kernel. It produces useful bounded
research-only ranking on supported OOF rows, but remains limited by domain shift
and insufficient calibration.

Morphology candidate v2 was generated to explain false-positive-like and
false-negative-like patterns. Under the same existing-feature Ridge comparison,
it did not improve ranking over v1:

- v1 Spearman: about 0.1714
- morphology-v2 Spearman: about 0.0624
- v1 top-10 lift: about 1.50
- morphology-v2 top-10 lift: about 0.86

## Decision

Retain v1 as the current research screening baseline.

Morphology-v2 must not replace v1. Morphology-v2 remains audit-only and
parallel-comparison-only. Its value is error interpretation and future
label-semantics review, not automatic label replacement.

Final labels and production claims remain forbidden.

## Consequences

- Future morphology semantics must be evaluated as pre-registered parallel
  candidates.
- Any morphology candidate that improves ranking still requires human review
  before label-policy adoption.
- XSI-only leakage boundaries remain unchanged; CAST-derived morphology fields
  are weak-label audit fields, not production model inputs.
- No deployment, cross-well, ground-truth, or final-label claim is authorized by
  this ADR.
