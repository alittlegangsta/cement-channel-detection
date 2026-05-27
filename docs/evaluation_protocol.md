## MVP-4A Evaluation Gate

MVP-4A is a sanity correlation validation, not model evaluation.

Allowed outputs:

```text
xsi_label_samples_v001.npz
xsi_basic_features_v001.npz
xsi_cast_correlation_report_v001.json
xsi_cast_correlation_v001.csv
mvp4a_review_v001/
mvp4a_gate_report.json
```

Required gate conditions:

```text
label sample index valid
XSI basic feature finite ratio acceptable
high-confidence subset exists
candidate/non-candidate feature separation is interpretable
low-confidence intervals are not used as strong azimuthal supervision
no model training
no final labels
```

MVP-4A metrics are descriptive statistics only: mean/median differences,
confidence-weighted differences, point-biserial style effect size, Spearman/rank
correlation when applicable, and severity trend summaries. Do not report AUC,
accuracy, train/test split, or model performance in MVP-4A.

Gate decisions:

```text
go
conditional_go
no_go
```

`go` or `conditional_go` permits only MVP-4B feature-engineering sanity work. It
does not permit MVP-5 baseline modeling, deep learning, or final label claims.

## MVP-4B Simple Baseline Sanity Evaluation

MVP-4B Stage 2 is a weak-label sanity model check, not production model
evaluation. Metrics compare XSI-derived transformed features with CAST
weak-label candidates and must not be described as ground-truth performance.

Required evaluation constraints:

```text
depth-block group split only
high-confidence azimuthal subset only
sample_weight enabled
plus primary labels
minus audit comparison documented
permutation label sanity check enabled
no final labels
no deep learning
no STC
no APES
no production model export
```

Allowed descriptive metrics:

```text
weighted_accuracy
balanced_accuracy
f1
precision
recall
calibration_summary
```

No-Go conditions:

```text
permutation labels match or exceed real-label sanity metrics
depth-block split is invalid
fold class balance is insufficient
sample weights are invalid or all zero
suspiciously high metrics suggest leakage
low-confidence azimuth labels are used as strong labels
any report claims final labels or a production model
```

Passing this gate only permits MVP-4C advanced feature engineering. It does not
permit MVP-5, deep learning, STC, APES, or final label claims.

## MVP-4B Stage 2 Gate

The MVP-4B Stage 2 gate consumes the simple baseline report, review summary, and
Stage 1 gate report. A `go` or `conditional_go` decision only permits MVP-4C
advanced feature engineering.

Required gate evidence:

```text
valid depth-block split
sufficient high-confidence candidate/non-candidate samples
real weak-label metrics exceed permutation-label metrics
no leakage suspicion
interpretable coefficient summary exists
plus primary vs minus audit comparison exists
no final labels
no production model
no deep learning / STC / APES
```

The gate must be `no_go` when permutation labels match or outperform real
weak-label candidates, when split validity fails, or when metrics are
suspiciously high enough to require leakage review.

## MVP-4B No-Go Diagnostics

When the simple baseline does not exceed the permutation-label sanity baseline,
the next step is diagnosis, not more complex modeling. The diagnostics may
inspect sample weights, fold distributions, prediction degeneracy, disagreement
subsets, confidence thresholds, and feature effect sizes.

The diagnostics must not:

```text
train deep learning models
run STC or APES
enter MVP-4C
generate final labels
call weak-label candidates ground truth
claim production model performance
```

Recommended no-go reason classes:

```text
label_noise
feature_weakness
split_distribution_shift
class_weight_failure
sample_weight_failure
depth_leakage_or_block_issue
insufficient_high_confidence_signal
```

Outputs are review artifacts only and do not change the Stage 2 no-go decision.

## MVP-4B-R Sample Weight Remediation

When no-go diagnostics show single-class prediction caused by effective class
weight imbalance, the next evaluation step is sample-weight remediation rather
than a more complex model. Remediation weights are still weak-label sanity
weights and must not be interpreted as final-label confidence.

Required policy checks:

```text
confidence_only retained as old-policy control
class_balanced_confidence reported
capped_class_balanced_confidence reported and used by default
unweighted control reported
candidate effective weight fraction capped by config
low-confidence azimuthal samples have zero azimuthal weight
large depth-match-error samples have zero azimuthal weight by default
plus/minus disagreement samples are downweighted or explicitly excluded
per-fold effective weight balance is reported
```

The default candidate effective weight fraction should not exceed `0.60`
without an explicit configuration change. If class-balanced weights cannot be
constructed because either high-confidence candidate or non-candidate samples
are absent, the remediation remains `no_go`.

## MVP-4B-R Enhanced Feature Diagnostics

Enhanced remediation features are descriptive transformations of existing MVP-4B
basic features. They may be evaluated in controlled ablations but are not
formal feature engineering approval for MVP-4C.

Required checks:

```text
raw XSI waveform is not read
labels are not used to construct features
all enhanced transformed features are finite
original transformed features are preserved
side/depth normalization metadata is written
receiver-level side-normalized metrics are skipped and reported if unavailable
```

If enhanced transformed features produce substantial non-finite values, the
remediation remains `no_go` and must return to feature preprocessing review.

## MVP-4B-R Remediation Ablation Evaluation

Controlled remediation ablations must report:

```text
balanced_accuracy
weighted_accuracy
precision / recall / f1
predicted_positive_rate
candidate effective weight fraction
permutation balanced_accuracy
real - permutation margin
per-fold metrics
single-class prediction degeneracy
leakage warning
```

If all non-degenerate class-balanced configurations fail to exceed the
permutation sanity baseline, the remediation remains `no_go`. If only
`confidence_only` improves while class-balanced policies do not, the result is
treated as a sample-weight artifact rather than evidence to proceed.

## MVP-4B-R Remediation Gate

The remediation gate must require:

```text
class-balanced non-degenerate baseline above permutation
real - permutation balanced_accuracy margin >= 0.03
predicted_positive_rate not near 0 or 1
support from at least two depth-block folds
candidate effective weight fraction within cap
enhanced transformed feature finite ratio = 1.0
plus/minus disagreement strategy documented
no final labels
no STC / APES / deep learning / MVP-4C implementation already performed
```

Failure of these checks keeps the decision at `no_go` and recommends returning
to label sampling or feature design instead of escalating model complexity.

## MVP-4B-R2 Receiver Feature Diagnostics

Receiver-derived remediation features may be built only from
`xsi_basic_features_v001.npz` and existing MVP-4B sample tables. The diagnostic
goal is to determine whether the 13-receiver XSI array dimension carries
stronger weak-label sanity signal than side-level aggregates.

Required checks:

```text
raw XSI waveform is not read
labels are not used to construct receiver-derived features
all receiver-derived transformed features are finite
feature ranges and finite ratios are reported
top standardized differences are reported as diagnostics only
no final labels
no STC / APES / deep learning / MVP-4C implementation
```

Receiver feature ablations must still use depth-block splits, capped
class-balanced confidence weighting, and permutation checks. If the best
non-degenerate receiver-derived result fails the configured permutation margin,
the decision remains `no_go`.

Required receiver ablation comparisons:

```text
side-level enhanced features only
receiver-derived features only
side-level + receiver-derived features
receiver-derived late_over_early subset
receiver-derived far/near subset
include plus/minus disagreement
exclude plus/minus disagreement
```

The receiver ablation report may suggest a gate decision, but it must not enter
MVP-4C by itself.

## MVP-4B-R2 Receiver Feature Gate

The receiver feature gate consumes the receiver-derived feature report,
receiver ablation report, and prior MVP-4B-R gate report. It must evaluate the
best receiver-derived scenario separately from side-level-only scenarios.

Required gate conditions:

```text
receiver-derived transformed feature finite ratio = 1.0
label fields were not used to construct receiver-derived features
best receiver-derived scenario is class-balanced and non-degenerate
best receiver-derived real - permutation margin >= 0.03
best receiver-derived result is supported by at least two depth-block folds
no leakage suspicion
no final labels
no STC / APES / deep learning / MVP-4C implementation
```

If these checks fail, the recommended next step is label refinement or
controlled time-frequency feature sanity, not model escalation.

## MVP-4B-R3 Label-Quality Subset Gate

Label-quality subset diagnostics are weak-label noise diagnostics after the
receiver-feature gate remains `no_go`. They may build strong-positive,
clear-negative, disagreement-free, high-orientation, connected-object, and
review-exclusion masks over existing sample tables.

The gate requires:

```text
quality strong-positive and clear-negative subsets both non-empty and above the configured minimum
feature separation on quality subsets improves over the all-candidate baseline by the configured delta
quality subset best effect size reaches the configured sanity threshold
the ~5700 ft review exclusion does not flip the conclusion
no final labels
no STC / APES / deep learning / MVP-4C implementation
```

If these checks pass, the only allowed next recommendation is controlled
time-frequency sanity. If they fail, the project remains `no_go` and should
return to label definition review or manual annotation before adding model
complexity.

## MVP-4B-R4 Depth-Level Separation Audit

Depth-level target review evaluates whether XSI depth-level aggregates separate
CAST weak-label candidate depths more clearly than the previous side-depth
target. It is not a production model evaluation.

Required comparisons:

```text
depth_has_channel_any vs no_channel
strong_positive depths vs clear_negative depths
high-confidence depth only
exclude ~5700 ft review band sensitivity
low vs high plus/minus disagreement
```

Allowed metrics are feature distribution summaries, standardized effect size,
single-feature threshold sanity, and depth-shift sanity checks. The audit must
not train a model, run STC/APES, enter MVP-4C, or claim final labels.

Depth-level separation is considered enhanced only if its best absolute effect
size exceeds the prior side-level audit by the configured delta and reaches the
configured sanity threshold. A passing audit may only recommend Stage 5
depth-level gate review.

## MVP-4B-R4 Depth-Level Target Gate

The depth-level target gate consumes the depth-level label report, depth-level
XSI feature report, and depth-level separation audit. It must verify:

```text
strong positive depth subset is non-empty
clear negative depth subset is non-empty
~5700 ft review band does not dominate positives
depth-level separation improves over side-level audit
no final labels
no STC / APES / deep learning / MVP-4C
```

A `conditional_go` may allow only a depth-level baseline sanity model. It does
not allow side-level MVP-4C, STC/APES, deep learning, production model claims,
or final-label claims.

## MVP-4B-R4b Depth-Level Baseline Sanity

Depth-level baseline sanity may use only simple logistic regression or linear
probe baselines over `depth_level_xsi_features_v001`. It evaluates weak-label
agreement, not formal model performance.

Required target variants:

```text
all_positive_vs_negative
strong_positive_vs_clear_negative
high_confidence_positive_vs_clear_negative
```

Required checks:

```text
depth-block split
fold class balance
balanced_accuracy / precision / recall / f1
permutation check
real minus permutation margin
predicted positive rate
degenerate single-class prediction flag
top feature coefficients
```

If permutation metrics are not lower than real-label metrics, or if predictions
degenerate to a single class, the baseline remains `no_go` for that variant.
The gate also requires stable fold evidence: at least the configured minimum
number of depth-block folds must individually show a non-degenerate real minus
permutation margin. Variants with too few strong positives or fold-level class
balance failures are reported as skipped rather than forced.
Passing this sanity check can only feed a depth-level baseline gate; it does not
authorize MVP-4C, STC/APES, deep learning, production claims, or final labels.

## MVP-4B-R4b Depth-Level Baseline Gate

The depth-level baseline gate emits `go`, `conditional_go`, or `no_go`.
It may permit only controlled depth-level feature refinement when a simple
logistic-regression or linear-probe baseline beats its permutation baseline,
has non-degenerate predictions, and is supported by stable depth-block folds.

The gate must remain `no_go` when no target variant is usable, when the best
real-minus-permutation margin is below the configured threshold, when the
prediction collapses to a single class, or when any guardrail permits final
labels, STC, APES, deep learning, production claims, or MVP-4C. A
`conditional_go` caused by review warnings does not permit side-level MVP-4C.

## MVP-4B-R4c Controlled Depth-Level Refinement

Controlled depth-level refinement evaluates robustness, not formal model
performance. It may use only logistic regression or linear probe baselines over
`depth_level_xsi_features_v001` and only for
`high_confidence_positive_vs_clear_negative` weak-label candidate sanity checks.

Required robustness axes:

```text
feature group ablation
include vs exclude ~5700 ft review band
confidence thresholds
3-fold vs 5-fold depth-block split
repeated permutation checks
per-fold margin stability
```

The refinement recommendation must remain `no_go` when the real-label result
does not exceed permutation, when predictions are degenerate, when the result is
supported by only one fold, when it works only with the ~5700 ft band included,
or when leakage is suspected. A `conditional_go` is required when results depend
on a small number of feature groups, a specific confidence threshold, a depth
interval decision, or another scientific judgment that needs human review.

No refinement result authorizes MVP-4C, STC/APES, deep learning, production
claims, ground-truth claims for CAST weak-label candidates, or final labels.

## MVP-4B-R4c Depth-Level Refinement Gate

The refinement gate consumes the refinement robustness report, prior
depth-level baseline report, and review-figure summary. It returns `go`,
`conditional_go`, or `no_go`.

`go` requires multiple feature groups or robustness axes to remain above
permutation, mean margin at least `0.05`, stable fold support, non-degenerate
predicted-positive rate, robustness to excluding the ~5700 ft review band, and
no leakage warning.

`conditional_go` is required when the result exceeds permutation but depends on
a small number of feature groups, a narrow confidence threshold, a split choice,
or a depth interval decision requiring human review.

`no_go` is required when the result does not exceed permutation, predictions
degenerate, only one fold supports the result, leakage is suspected, or sample
subsets are insufficient. Regardless of decision, MVP-4C, STC/APES, deep
learning, final labels, production claims, and ground-truth claims remain
blocked.

## Depth-Level Manual Review Pack

When the refinement gate returns `go`, evaluation may proceed only to a manual
review pack. The pack selects depth intervals for expert inspection and reports
weak-label candidate summaries, prediction-score audit summaries, confidence,
plus/minus disagreement, 5700 ft review-band sensitivity, and XSI feature
summaries.

The pack is not a new model evaluation and is not formal performance evidence.
It must not train new models, generate final labels, call weak-label candidates
ground truth, enter MVP-4C, run STC/APES, or use deep learning. The expected
human review outcome is a checklist decision on whether the selected intervals
are physically plausible and whether any later controlled depth-level
refinement v2 requires separate approval.
