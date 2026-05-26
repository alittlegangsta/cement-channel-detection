# Labeling Protocol

本文件定义 MVP-3 及后续弱标签阶段的最小可审计规则。MVP-3 当前只允许生成
CAST weak-label candidates，不允许生成 final labels，也不允许进入特征提取、
模型训练或 MVP-4 相关性实验。

## MVP-3 CAST Weak-Label Candidate Scope

MVP-3 的标签来源是 CAST `Zc`。允许读取：

- `CAST.Zc`
- `CAST.Depth`
- `Depth_inc` / `Inc` / `RelBearing`
- `orientation_confidence_v001.npz`

禁止读取完整 XSI waveform，禁止做 XSI feature extraction、STFT、STC、APES 或模型训练。

## Label Version

当前候选标签版本：

```text
cast_weak_v001
```

该版本必须保留：

- `label_source = cast_weak_plus`
- `label_source = cast_weak_minus_ablation`
- `convention_status = specification_preferred_plus_data_unresolved`
- `no_final_labels = true`

## Candidate Encoding

`presence` 编码：

```text
-1 = unknown / invalid
 0 = no-channel candidate
 1 = channel candidate
```

`severity` 编码：

```text
-1 = unknown / invalid
 0 = none
 1 = mild
 2 = moderate
 3 = severe
```

`label_confidence` 必须是 `[0, 1]` 内的浮点数。

## Adaptive CAST Zc Baseline

MVP-3 不允许只使用固定 `Zc < 2.5 MRayl` 阈值。必须先沿 depth 方向建立
局部背景基线：

```text
rolling median 或 rolling quantile
window_m = 50-100
默认 quantile = 0.90
```

基线只能沿 depth 方向平滑，不得沿 azimuth 方向平滑，以免抹掉方位异常。

## Candidate Rule

MVP-3 的候选规则为：

```text
channel_candidate =
  (Zc < Zc_base * (1 - alpha))
  OR
  (Zc < zc_min_limit)
```

其中 `alpha` 和 `zc_min_limit` 必须来自配置。若 `zc_min_limit` 尚未人工确认，
脚本只能使用配置中声明的 conservative fallback，并在报告中明确标记需要人工确认。

## Confidence

`label_confidence` 至少融合：

- relative drop strength
- orientation confidence
- baseline validity
- RelBearing finite/valid status
- bad-data confidence
- environment / outlier warnings

低井斜或 RelBearing 不可靠位置只降低 confidence，不直接删除。

MVP-3R 必须拆解并输出以下分量，报告中应能解释低 confidence 的主因：

```text
zc_strength_confidence
baseline_confidence
orientation_confidence_on_cast_depth
relbearing_valid_confidence
bad_data_confidence
final_label_confidence
```

## Bad-Data Mask

候选生成前必须标记以下 bad-data 单元：

- non-finite `Zc`
- `Zc <= 0`
- extreme `relative_drop > 0.95`

bad-data 单元不得直接触发 severe label。isolated extreme outlier 需要在报告和
review figure 中单独标出；若保留为候选上下文，只能降低 confidence，不能提高
severity。

## Threshold Sensitivity

MVP-3R 必须运行阈值敏感性网格：

```text
alpha: [0.30, 0.35, 0.40]
zc_min_limit: [2.0, 2.5, 3.0]
severity threshold sets:
  default: [0.30, 0.45, 0.60]
  conservative: [0.35, 0.50, 0.65]
  aggressive: [0.25, 0.40, 0.55]
```

每组至少统计 plus/minus coverage、plus/minus disagreement、规则触发来源、
mean label confidence、low-confidence fraction、severity distribution、
connected component count、isolated speckle ratio、`relative_drop > 0.95`
outlier fraction 和 invalid/bad `Zc` fraction。该报告用于人工阈值复核，不
生成 final labels。

MVP-3H 人工审查接受以下参数作为 `candidate_v001` weak-label candidate 参数，
而不是 final label 参数：

```yaml
recommended_parameter_set:
  alpha: 0.35
  zc_min_limit: 2.5
  severity_thresholds: [0.30, 0.45, 0.60]
  status: human_reviewed_candidate_v001
  final_label: false
```

`zc_min_limit = 2.5 MRayl` 接受为 provisional `candidate_v001` 参数，但仍保留
domain confirmation note。

记录依据：

- alpha 增大时 coverage 下降；
- zc_min_limit 增大时 coverage 上升；
- 当前中心组 coverage 约 0.1797，处于本轮 sensitivity 中间范围；
- relative_drop image 和 candidate overlays 呈成片连续，不是纯随机噪声；
- speckle ratio 约 0.018-0.023，不是主要问题；
- `relative_drop > 0.95` outlier fraction 约 `4.04e-05`，数量很少；
- 低 confidence 主要来自 orientation confidence，尤其 4200-5700 深度段；
- baseline confidence、relbearing_valid_confidence 基本为 1；
- bad_data_confidence 只有少量异常；
- plus/minus disagreement 仍约 0.20，必须保留 plus primary / minus ablation。

RelBearing label policy:

```yaml
relbearing_label_policy:
  primary: plus
  primary_status: human_specification_approved
  data_driven_validation: insufficient_evidence
  minus_ablation_retained: true
  minus_usage: audit_only
  single_sign_final_label_approved: false
```

## Review And Gate

MVP-3 完成后必须输出 candidate audit 和 review figures。MVP-3R review figures
必须使用无 GUI 可运行的 matplotlib Agg backend，并包含 colorbar、depth axis、
azimuth axis、标题、confidence 拆解图、bad-data overlay、relative-drop outlier
overlay、plus/minus disagreement map，以及只在 candidate 区域显示的 severity map。

当前 MVP-3H gate 必须保持 `conditional_go` 且 `mvp4_allowed=false`，原因是 weak
labels 是 human-reviewed candidates，但不是 final labels；MVP-4 需要单独明确批准。

## MVP-4B-R3 Label-Quality Subsets

MVP-4B-R3 只允许在 simple baseline 和 receiver-derived remediation 仍为 `no_go`
之后构建 CAST weak-label candidate 的高质量诊断 subset。该阶段不得生成 final
labels，也不得把 subset 当作 ground truth。

必须保留：

```text
primary_label = plus
audit_label = minus_ablation
label_status = human_reviewed_candidate_v001
no_final_labels = true
```

允许的 subset 仅用于诊断 label noise / label mapping noise：

```text
strong_positive: plus candidate, moderate/severe, confidence threshold, no plus/minus disagreement
clear_negative: plus non-candidate, high confidence, no plus/minus disagreement
high_confidence_orientation: orientation_confidence threshold
connected_object_only: candidate connected objects above area/depth-length threshold
exclude_review_intervals: includes the ~5700 ft horizontal severe band as review exclusion
```

`connected_object_only` 和 review exclusion 都是 weak-label quality filters，不是人工批准的
final label 操作。若这些 filters 使 feature separation 明显增强，只能说明 label
noise 或 mapping noise 可能是 MVP-4B no-go 的主因。

## MVP-4B-R4 Depth-Level Target Review

MVP-4B-R4 只允许把 CAST weak-label candidates 聚合成 depth-level review target，
用于检查“每个深度是否存在明显 CAST channel candidate”。它不是 final label 生成，
也不得把 CAST candidate 称为 ground truth。

主任务字段必须来自 `configs/depth_level_label.example.yaml`，至少包含：

```text
depth_has_channel_any
depth_candidate_fraction
depth_max_severity
depth_max_confidence
depth_min_zc
depth_p05_zc
depth_p10_zc
depth_max_relative_drop
depth_largest_azimuth_object_width
depth_plus_minus_disagreement_fraction
depth_orientation_confidence
depth_label_confidence
```

聚合策略必须保留 `any`、`max`、`percentile`、`fraction` 等信息，不允许只用
mean。side-level 方位标签只能作为 audit 字段，不能作为主训练目标。

必须保留：

```text
primary_label = plus
audit_label = minus_ablation
label_status = human_reviewed_candidate_v001
side_level_labels.usage = audit_only
side_level_labels.train_target = false
no_final_labels = true
```

MVP-4B-R4 可构建 strong positive / clear negative depth 子集做 feature separation
sanity，但这些子集仍只是 weak-label candidate review masks。若 positive 或 negative
depth subset 为空，或 positive depth 主要由约 5700 ft horizontal severe band 主导，
必须停止并返回人工标签审查。

Depth-level candidate table 构建脚本为：

```bash
python scripts/06r_build_depth_level_labels.py --config configs/paths.local.yaml
```

输出 `depth_level_labels_v001.npz` 只允许作为 target review artifact；其中
`depth_strong_positive_mask` 和 `depth_clear_negative_mask` 是 sanity subset，不是
ground truth，也不是 production label。
