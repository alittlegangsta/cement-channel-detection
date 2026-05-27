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
interval_cast_panels/
interval_xsi_feature_panels/
```

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

required_notes:
```

## Guardrails

- Treat all labels as weak-label candidates, not ground truth.
- Do not generate or approve final labels from this pack.
- Do not infer production readiness from the review figures.
- Do not enter MVP-4C or any advanced feature branch without separate approval.
- Do not run STC/APES or deep learning under this checklist.
