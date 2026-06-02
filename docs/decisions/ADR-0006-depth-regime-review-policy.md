# ADR-0006: Depth-Regime Review Policy For Geometry Regression

Date: 2026-06-02

## Status

Accepted for manual review organization and descriptive statistics only.

## Context

MVP-4B-GR-R2 found a clear review-only depth-regime shift across target,
feature, and morphology summaries. The strongest priority review zones were:

```text
2582.79 ft: target / feature / morphology / near-far support
4219.52 ft: morphology / orientation / inclination support
5680.00 ft: feature / 5700 band support
```

Secondary review zones were recorded at 2826.85 ft, 3534.31 ft, 3984.03 ft,
and 4678.49 ft. Depth-match error covariance was unavailable and must remain a
warning rather than an inferred explanation.

## Decision

The project records `configs/geometry_regression_depth_regime_review.example.yaml`
and `cement_channel.evaluation.depth_regime_review_policy` as the formal
review-only policy for geometry-regression manual review packs.

The broad descriptive review regimes are:

```text
Regime A: approximately 2350-3534 ft
Regime B: approximately 3534-4678 ft
Regime C: approximately 4678-5760 ft
```

These regimes may be used only for descriptive statistics, visualization
grouping, manual review organization, physical interpretation, and review
prioritization. The priority boundary review zones use a conservative
configurable half-width and are manual-review windows, not validated physical
layer boundaries.

Target-view policy is:

```text
receiver_mean: conservative reference
receiver_p90: robust candidate for human review discussion
receiver_max: sensitive audit only
full_360_fraction: auxiliary coverage view
receiver_std: receiver disagreement / heterogeneity audit
```

Existing morphology arrays may be shown in the manual review pack, but they
are not approved as new label targets.

## Explicit Non-Decisions

This ADR does not approve:

```text
formal CV split changes
regime-specific model training
production claims
receiver_p90 or receiver_mean as the primary target view
new morphology targets
label definition changes
new XSI features
near/far preprocessing changes
waveform reads
MVP-4C
STC
APES
deep learning
final labels
ground-truth claims
```

## Consequences

Stage 10 stop remains valid. Stage 11 and Stage 12 remain blocked. The manual
review pack can organize intervals by broad regime and priority boundary zone,
but any move toward formal CV stratification, target-view adoption, morphology-
aware target redesign, preprocessing review, or controlled feature review
requires a separate human approval.

