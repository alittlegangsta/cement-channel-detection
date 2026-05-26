# ADR-0004: MVP-4B No-Go And Next Directions

Date: 2026-05-26

## Status

Accepted

## Context

MVP-4B tested whether side-depth XSI basic features, side/depth normalized
features, and receiver-derived basic features could weakly agree with
`cast_weak_label_candidates_v001` under a simple, auditable baseline workflow.

The project has completed:

```text
MVP-4B Stage 2 simple baseline
MVP-4B no-go diagnostics
MVP-4B-R sample weighting and side-normalized remediation
MVP-4B-R2 receiver-derived feature remediation
MVP-4B-R3 label-quality subset diagnostics
```

All gates remain `no_go`.

Key observations:

```text
Initial simple baseline degenerated to predicted_positive_rate = 1.0.
Permutation labels matched or exceeded the real weak-label sanity result.
Class-balanced weighting fixed the candidate effective-weight artifact.
Best non-degenerate remediation margin ~= 0.019996 < 0.03.
Best receiver-derived margin ~= 0.0163 < 0.03.
Label-quality subsets did not strengthen feature separation.
```

The CAST labels remain weak-label candidates, not final labels or ground truth.

## Decision

Do not proceed to MVP-4C.

Do not implement STC, APES, deep learning, production training, production
inference, or final-label workflows from the current MVP-4B evidence.

Treat the current side-depth weak-label classification formulation as no-go
for the existing shallow feature family.

The next approved research direction is to reconsider the target definition
before increasing model or feature complexity:

1. Prefer depth-level / interval-level anomaly validation review.
2. Alternatively build a manual/expert label review pack.
3. Consider controlled time-frequency feasibility only after explicit approval.
4. Revisit alignment and depth-window label aggregation before any advanced
   modeling.

## Consequences

The project remains in label/target-definition review, not advanced feature
engineering.

Allowed next branches:

```text
feature/mvp4b-depth-level-target-review
feature/manual-label-review-pack
```

Disallowed without a new gate:

```text
MVP-4C
STC
APES
deep learning
production model
final labels
ground-truth claims for CAST weak-label candidates
```

This decision can be revisited only after a new review shows that the target
definition or expert-reviewed labels provide a stronger and physically
interpretable validation basis.

