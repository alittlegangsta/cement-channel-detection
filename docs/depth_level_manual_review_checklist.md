# Depth-Level Manual Review Checklist

Scope: MVP-4B-R4c+ depth-level manual review pack. This checklist is for
human review of depth-level CAST weak-label candidate anomaly plausibility. It
does not approve MVP-4C, STC/APES, deep learning, production modeling, or final
labels.

Review directory:

```text
/home/xiaoj/cement-channel-data/reports/depth_level_manual_review_v001/
```

Primary files to inspect:

```text
review_summary.md
review_intervals.csv
review_intervals.json
overview_depth_label_score_confidence.png
selected_intervals_overview.png
5700_band_sensitivity.png
confidence_and_disagreement_panels.png
interval_xsi_feature_summary_table.csv
interval_xsi_feature_summary_table.json
interval_cast_evidence_summary_table.csv
interval_cast_evidence_summary_table.json
interval_cast_panels/
interval_cast_heatmaps/
interval_xsi_feature_panels/
```

Geometry-aware supplement, if present:

```text
/home/xiaoj/cement-channel-data/reports/geometry_aware_manual_review_v001/
review_summary.md
geometry_mode_comparison.csv
geometry_mode_comparison.json
interval_geometry_comparison.csv
interval_geometry_comparison.json
```

Geometry-aware regression supplement, if present:

```text
/home/xiaoj/cement-channel-data/reports/geometry_regression_manual_review_v001/
review_summary.md
kernel_comparison.csv
kernel_comparison.json
interval_regression_target_summary.csv
interval_regression_target_summary.json
selected_interval_review_list.csv
selected_interval_review_list.json
figures/
```

## Visualization Caveats

- XSI raw features have very different physical scales. Use the raw small
  multiples only to inspect each feature trace on its own y-axis.
- Use the XSI normalized robust-z panel for cross-feature comparison within an
  interval.
- CAST summary metrics are mixed units. Presence fraction, candidate
  confidence, relative drop, and low-inclination fraction are plotted separately
  from severity and Zc.
- Do not compare CAST severity, Zc, and 0-1 summary metrics by bar height.
- CAST evidence categories are review-only categories, not final labels.
- Geometry-aware CAST evidence may change because XSI represents a
  source-to-receiver interval response while CAST is local depth by azimuth
  evidence. Treat changed categories as review prompts, not final labels.
- Geometry-aware regression targets are continuous weak-label candidates. Do
  not collapse them back to final positive/negative decisions during review.
- Review the geometry-aware supplement before approving any later
  interval-level target. Pay special attention to DLR intervals where
  `review_decision_should_be_revisited=true`.
- Review the regression supplement intervals by type:
  `high_fraction_interval`, `low_fraction_interval`, `local_only_anomaly`,
  `kernel_sensitive_interval`, `xsi_high_cast_fraction_low`,
  `xsi_low_cast_fraction_high`, `5700_band_review`, and
  `uncertainty_review`.

## Interval Review Order

1. Inspect the CAST evidence category and the separated CAST panels.
2. Inspect CAST heatmaps when candidate mask, severity, relative drop, or raw Zc
   arrays are available.
3. Inspect XSI normalized robust-z features for relative anomalies.
4. Inspect XSI raw small multiples only for per-feature trace shape and local
   min/mean/max.
5. Compare the score audit with the weak-label candidate class.
6. Record accept, reject, uncertain, or special handling.

## Required Questions

1. Do high-score positive intervals correspond to clear CAST weak-label
   candidate anomaly evidence?
2. Do clear negative intervals look physically normal in CAST summaries and XSI
   feature summaries?
3. For false-positive-like intervals, are they likely weak-label misses, XSI
   noise, or score artifacts?
4. For false-negative-like intervals, are they likely weak-label noise, weak XSI
   sensitivity, or physically subtle anomalies?
5. Should the 5700 ft review band be retained, excluded, or handled separately?
6. Should low-confidence or plus/minus-disagreement intervals be excluded from
   later depth-level review, down-weighted, or handled as uncertain?
7. Is controlled depth-level feature refinement v2 approved as a separate next
   step?
8. Should MVP-4C, STC/APES, deep learning, production modeling, and final labels
   remain blocked?
9. Does the geometry-aware supplement require confirming depth-axis sign before
   any later interval-level target review?
10. Do any DLR intervals need re-review because source-receiver interval
    aggregation changes the CAST evidence category?
11. Do continuous regression fractions look healthier than the old binary
    any-cell aggregation?
12. Which regression kernel, if any, should be used as the next manual-review
    primary candidate?
13. Which regression intervals from `selected_interval_review_list.csv` require
    manual inspection before any further experiment?

## Review Decision Template

```text
reviewer:
review_date:

high_score_positive_intervals_physically_plausible: yes/no/uncertain
clear_negative_intervals_physically_normal: yes/no/uncertain
false_positive_like_primary_explanation: label_miss/xsi_noise/score_artifact/uncertain
false_negative_like_primary_explanation: label_noise/xsi_insensitive/subtle_anomaly/uncertain
5700_band_decision: retain/exclude/handle_separately/uncertain
low_confidence_disagreement_policy: keep/down_weight/exclude/uncertain
approve_controlled_depth_level_feature_refinement_v2: yes/no/conditional
mvp4c_stc_apes_deep_learning_final_labels_still_blocked: yes/no
depth_axis_sign_requires_confirmation: yes/no/uncertain
dlr_intervals_requiring_geometry_re_review:
regression_targets_healthier_than_old_binary: yes/no/uncertain
regression_primary_kernel_candidate: r7_reference_point/midpoint_window/uniform_source_receiver_interval/triangular_midpoint_weighted/none/uncertain
regression_intervals_requiring_review:

required_notes:
```

## Guardrails

- Treat all labels as weak-label candidates, not ground truth.
- Do not generate or approve final labels from this pack.
- Do not infer production readiness from the review figures.
- Do not enter MVP-4C or any advanced feature branch without separate approval.
- Do not run STC/APES or deep learning under this checklist.
- Treat `geometry_alignment_gate_report.json` as a review gate only. A
  `conditional_go` still requires human confirmation before any next branch.
