# ADR-0007: MVP-4X Research-Only Regime Policy

Status: accepted for research-only exploratory analysis.

Scope: research_only, exploratory_only, weak_label_target, no_final_labels, no_ground_truth_claim, no_production_claim.

## Context

MVP-4X confounding and stratified exploratory studies show that Regime B and
Regime C contain learnable weak-label-related XSI signal under classical
models, while the whole-well model remains unstable. The evidence is not a
ground-truth claim. CAST-derived continuous targets remain weak-label
candidates.

Regime A has no high-orientation support in the current well. Regime B has
low/high orientation support, but the high-orientation effect does not survive
depth/target matched controls as a formal independent orientation effect.
Regime C has no low-orientation support. B to C and C to B transfer remain
weak, and domain shift is present.

## Decision

Regime A, Regime B, Regime C, and pooled B+C are approved only for research
evaluation:

- Regime A is audit_only because it has no high-orientation support. No
  generalization claim is allowed for Regime A high-orientation behavior.
- Regime B is an exploratory_modeling_domain. A low/high orientation
  common-support audit is allowed, but orientation is not a model input.
- Regime C is an exploratory_modeling_domain. Low-orientation support is
  unavailable in the current well.
- Pooled B+C is exploratory_reference_only and is not automatically preferred
  over regime-specific research baselines.

This policy is not a production split and does not approve a production regime
policy. The high-orientation cohort remains an exploratory cohort only; an
orientation filter is not approved.

Target-view policy:

- receiver_mean is the exploratory primary target.
- receiver_p90 is the robust reference target.
- receiver_max is sensitive audit-only.
- full_360_fraction is auxiliary-only.
- receiver_std is heterogeneity audit-only.

Model use is restricted to research-only anomaly ranking / screening. The
current evidence does not support absolute channel-fraction prediction.

## Forbidden Claims

The following remain forbidden:

- production split;
- production model;
- final labels;
- ground-truth claim;
- absolute channel-fraction prediction claim;
- formal orientation filter;
- using regime_id, depth, orientation confidence, CAST-derived arrays, or
  morphology arrays as candidate production model inputs.

## Consequences

MVP-4X-RP can run repeated blocked validation, ranking audits, ordinal audit
views, calibration/error analysis, and screening-baseline review packs. These
outputs must remain research-only and must not be used to approve STC, APES,
deep learning, final labels, or production deployment without a separate human
scientific decision.
