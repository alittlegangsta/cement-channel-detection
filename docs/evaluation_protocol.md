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
