# MVP-4B-R4c Depth-Level Refinement Decision Pack

Date: 2026-05-26

This decision pack summarizes controlled depth-level feature refinement results
for human review. It does not authorize MVP-4C, STC, APES, deep learning,
production modeling, final labels, or ground-truth claims for CAST weak-label
candidates.

## Scope

The reviewed target is:

```text
target_variant = high_confidence_positive_vs_clear_negative
label_status = weak_label_candidate
no_final_labels = true
```

Side-level azimuth labels remain audit-only. The prior side-depth MVP-4B result
remains no-go for the existing shallow side-depth target formulation.

## Key Result

Controlled refinement found a robust depth-level weak-label sanity signal:

```text
gate decision = go
manual_confirmation_required = false
next_branch_requires_human_approval = true
best feature group = robust_top_features_from_baseline
best confidence threshold = 0.6
best exclude_5700_band = true
real balanced_accuracy mean = 0.59192
permutation balanced_accuracy mean = 0.49179
margin mean = 0.10013
margin std = 0.03601
predicted_positive_rate = 0.59532
folds above permutation fraction = 1.0
suspicious leakage = false
```

Robustness evidence:

```text
passing scenario count = 33
passing feature groups =
  - all_depth_features
  - energy_window_features
  - receiver_summary_features
  - robust_top_features_from_baseline
passing confidence thresholds = 0.4, 0.5, 0.6
passing depth-block splits = 3, 5
passing exclude_5700 values = false, true
```

Interpretation:

```text
The depth-level high-confidence weak-label candidate target is more robust than
the previous side-depth target under simple, permutation-safe sanity baselines.
The result is not dependent on the 5700 ft review band, a single split setting,
or one confidence threshold.
```

## Recommended Next Step

Recommended next action:

```text
manual decision review before opening any next branch
```

Allowed topics for that review:

```text
1. controlled depth-level feature refinement v2
2. manual label review pack
3. interval-level target review
4. controlled time-frequency feasibility review
```

Not allowed without separate explicit approval:

```text
MVP-4C
STC
APES
deep learning
production model
final labels
ground-truth claims for CAST weak-label candidates
```

## Human Questions

Please confirm before any next branch:

```text
1. Accept depth-level high-confidence target as the next review target?
2. Prefer controlled depth-level feature refinement v2 or manual label review pack?
3. Keep confidence thresholds 0.4/0.5/0.6 for sensitivity, or choose a narrower policy?
4. Treat 5700 ft as an audit sensitivity only, since exclusion still passes?
5. Allow a controlled time-frequency feasibility review, or defer it?
```

## Files To Review

Primary reports:

```text
/home/xiaoj/cement-channel-data/reports/depth_level_refinement_report_v001.md
/home/xiaoj/cement-channel-data/reports/depth_level_refinement_report_v001.json
/home/xiaoj/cement-channel-data/reports/depth_level_refinement_gate_report.md
/home/xiaoj/cement-channel-data/reports/depth_level_refinement_gate_report.json
```

Review figures:

```text
/home/xiaoj/cement-channel-data/reports/depth_level_refinement_review_v001/
```

Most useful figures:

```text
04_robustness_margin_heatmap.png
05_permutation_margin_distribution.png
06_confidence_threshold_comparison.png
07_depth_block_split_comparison.png
08_exclude_5700_sensitivity.png
```

## Final Boundary

This pack supports human review of the depth-level direction only. It does not
convert weak-label candidates into final labels and does not approve any
advanced modeling stage.
