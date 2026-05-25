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
