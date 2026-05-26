# MVP-4B No-Go Research Review

Date: 2026-05-26

This document summarizes the MVP-4B no-go result and the follow-up remediation
diagnostics. It is a research review only. It does not add experiments, train
models, generate final labels, implement STC/APES, or authorize MVP-4C.

## Scope

MVP-4B tested whether side-depth XSI basic, normalized, and receiver-derived
features have enough weak-label agreement with CAST weak-label candidates to
justify moving into advanced feature engineering.

The label source remains:

```text
cast_weak_label_candidates_v001
status = human_reviewed_candidate_v001
no_final_labels = true
plus = primary
minus = audit only
```

The CAST weak-label candidates are not ground truth.

## Completed Stages

1. MVP-4B Stage 1: baseline sample table and robust preprocessing.
2. MVP-4B Stage 2: simple baseline sanity model.
3. MVP-4B no-go diagnostics.
4. MVP-4B-R: sample weighting remediation and side/depth normalized features.
5. MVP-4B-R2: receiver-derived XSI basic feature remediation.
6. MVP-4B-R3: label-quality subset diagnostics.

All gates remain `no_go`.

## Core Results

### Stage 2 Simple Baseline

The first simple baseline did not beat the permutation-label sanity check.

```text
logistic real balanced_accuracy = 0.5000
logistic permutation balanced_accuracy = 0.5000
linear real balanced_accuracy = 0.5000
linear permutation balanced_accuracy = 0.5002
predicted_positive_rate = 1.0
```

Direct interpretation:

```text
The model degenerated into single-class candidate prediction.
The permutation-label result was not lower than the real weak-label result.
The Stage 2 gate must remain no_go.
```

### No-Go Diagnostics

Diagnostics showed that the initial failure was strongly affected by sample
weight imbalance:

```text
candidate count fraction = 0.3400
candidate weight fraction = 0.7915
high-confidence plus/minus disagreement fraction = 0.3618
late_over_early_ratio single-feature threshold balanced_accuracy = 0.5663
```

This identified a weighting artifact, but did not prove that the task was
learnable once remediated.

### MVP-4B-R Weighting And Normalized Features

Sample weighting remediation corrected the candidate effective-weight imbalance:

```text
confidence_only candidate weight fraction = 0.7915
capped_class_balanced_confidence candidate weight fraction = 0.4998
```

The best non-degenerate remediation result was still below the gate margin:

```text
best non-degenerate margin ~= 0.019996
gate margin = 0.03
```

Direct interpretation:

```text
The initial sample_weight artifact was corrected, but class-balanced,
non-degenerate side-level features still did not provide enough margin over
permutation.
```

### MVP-4B-R2 Receiver-Derived Features

Receiver-derived features were added from existing XSI basic features:

```text
raw receiver features = 90
transformed receiver features = 147
finite ratio = 1.0
best receiver-derived margin ~= 0.0163
```

Direct interpretation:

```text
The 13-receiver dimension, as represented by basic receiver-derived summaries,
did not provide enough incremental weak-label signal.
```

### MVP-4B-R3 Label-Quality Subsets

High-quality weak-label subsets were built to test whether simple label cleanup
reveals stronger separation:

```text
strong_positive = 2203
clear_negative = 24814
strong_clear_quality = 14134
quality subset best abs effect size = 0.2838
all weak-label best abs effect size = 0.3735
label_noise_likely = false
5700 ft review exclusion did not flip result
```

Direct interpretation:

```text
The high-quality subset did not strengthen feature separation. Simple label
cleanup does not explain the no-go result.
```

## Direct No-Go Cause

The direct reason for MVP-4B no-go is:

```text
Existing side-depth XSI basic, side-normalized, and receiver-derived features
do not produce a class-balanced, non-degenerate, permutation-safe weak-label
sanity result with margin >= 0.03.
```

This is not a final statement about whether XSI can detect channeling. It is a
statement about the current side-depth weak-label classification formulation
and the current shallow feature set.

## Explanations Already Tested

### Sample-Weight Artifact

Status: mostly ruled out as the remaining primary cause.

The original model degenerated because candidate samples dominated effective
weight. Capped class-balanced weighting corrected the effective candidate
weight fraction to about 0.4998. The model no longer collapsed completely, but
the best margin remained below gate.

### Receiver-Derived Basic Features

Status: insufficient gain.

Receiver-level summary features were finite and auditable, but the best
receiver-derived margin was about 0.0163, below the 0.03 gate threshold.

### Simple Label Cleanup

Status: not supported as the main fix.

The strong-positive / clear-negative subset was large enough, and the 5700 ft
review exclusion did not flip the result. However, feature separation did not
increase relative to the all weak-label comparison.

## Remaining Plausible Explanations

The no-go result does not rule out these possibilities:

1. Advanced XSI waveform, time-frequency, or physics features may carry signal
   that the current basic energy features miss.
2. The side-depth classification target may be too fine for the physical scale
   of the XSI response.
3. A depth-level or interval-level target may better match XSI vertical
   resolution and receiver aperture.
4. CAST weak-label candidate geometry and XSI acoustic response may be
   mismatched in azimuth or depth scale even after current alignment.
5. CAST low-impedance weak labels may capture material or imaging effects that
   do not map one-to-one onto simple XSI energy summaries.

## Recommended Next Routes

Priority A: redefine the task as depth-level / interval-level anomaly
validation.

```text
Goal: test whether XSI supports interval-level weak anomaly evidence instead of
per-side side-depth classification.
Suggested branch: feature/mvp4b-depth-level-target-review
```

Priority B: build a small manual/expert label review pack.

```text
Goal: inspect a small number of intervals with CAST images, weak-label masks,
XSI summaries, orientation confidence, and disagreement overlays.
Suggested branch: feature/manual-label-review-pack
```

Priority C: controlled time-frequency feature feasibility, only after explicit
approval.

```text
Goal: evaluate whether simple time-window/frequency-window descriptors could be
worth a future physics-feature stage.
Constraint: this does not authorize STC, APES, deep learning, or MVP-4C.
```

Priority D: more detailed alignment and depth-window label aggregation.

```text
Goal: aggregate weak labels over physically meaningful depth windows and test
whether side-depth label granularity is the bottleneck.
```

## Explicit Non-Authorizations

The current evidence does not allow:

```text
MVP-4C
STC implementation
APES implementation
deep learning
production training
production inference
final label claims
ground-truth claims for CAST weak-label candidates
```

## Addendum: MVP-4B-R4 Depth-Level Direction

After this side-depth no-go review, MVP-4B-R4 redefined the sanity target as a
depth-level CAST weak-label candidate anomaly. Side-level azimuth labels remain
audit-only and are not the primary target.

Depth-level target review produced a `conditional_go` for a depth-level baseline
sanity check:

```text
depth-level positive fraction = 0.8314575
strong positive depths = 10
clear negative depths = 573
5700 ft band impact = 37 / 5910 positives, not dominant
depth-level best abs effect size = 0.8488
prior side-level best abs effect size = 0.3735
```

Depth-level baseline sanity then found one usable target variant:

```text
target_variant = high_confidence_positive_vs_clear_negative
balanced_accuracy = 0.58797
permutation balanced_accuracy = 0.50250
permutation margin = 0.08547
predicted_positive_rate = 0.57084
degenerate prediction = false
```

MVP-4B-R4c controlled depth-level refinement tested feature-group, confidence
threshold, split-count, repeated permutation, and ~5700 ft exclusion
robustness. The refinement gate returned `go` for preparing a human decision
pack only:

```text
best feature group = robust_top_features_from_baseline
best confidence threshold = 0.6
best exclude_5700_band = true
real balanced_accuracy mean = 0.59192
permutation balanced_accuracy mean = 0.49179
margin mean = 0.10013
predicted_positive_rate = 0.59532
folds above permutation fraction = 1.0
passing feature groups = all_depth_features, energy_window_features,
                         receiver_summary_features,
                         robust_top_features_from_baseline
passing confidence thresholds = 0.4, 0.5, 0.6
passing depth-block splits = 3, 5
passing exclude_5700 values = false, true
suspicious leakage = false
```

This addendum does not rewrite the side-depth MVP-4B no-go conclusion. It only
supports continued depth-level target/refinement review and a human decision
pack. It still does not authorize MVP-4C, STC, APES, deep learning, production
training, production inference, final labels, or ground-truth claims for CAST
weak-label candidates.

## Recommended Decision

Keep MVP-4B in `no_go`.

Recommended next branch:

```text
feature/mvp4b-depth-level-target-review
```

Alternative branch:

```text
feature/manual-label-review-pack
```
