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

## MVP-4B Stage 1 Gate

MVP-4B Stage 1 is a sample-table and preprocessing readiness gate. It is not
model evaluation and must not report accuracy, AUC, train/test split results, or
model performance.

Allowed inputs:

```text
baseline_sample_table_report_v001.json
feature_preprocessing_diagnostics_v001.json
mvp4a_gate_report.json
```

Required gate conditions:

```text
MVP-4A permits MVP-4B
baseline sample table built successfully
high-confidence candidate and non-candidate subsets both exist
transformed features are finite within the configured tolerance
sample_weight is finite, bounded, and not all zero
depth_match_error exclusion policy is reported
plus/minus disagreement is preserved as an audit flag
no model training
no final labels
```

Gate decisions:

```text
go
conditional_go
no_go
```

`go` or `conditional_go` permits only MVP-4B Stage 2 simple baseline sanity
modeling. It does not permit MVP-4C, MVP-5, deep learning, STC, APES,
production inference, or final label claims.
