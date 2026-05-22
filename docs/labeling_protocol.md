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

人工审查当前接受以下 provisional weak-label 参数组作为后续复核中心点，而不是
final label 参数：

```yaml
recommended_parameter_set:
  alpha: 0.35
  zc_min_limit: 2.5
  severity_thresholds: [0.30, 0.45, 0.60]
  status: provisional_after_sensitivity
  requires_human_review: true
  no_final_labels: true
```

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

## Review And Gate

MVP-3 完成后必须输出 candidate audit 和 review figures。MVP-3R review figures
必须使用无 GUI 可运行的 matplotlib Agg backend，并包含 colorbar、depth axis、
azimuth axis、标题、confidence 拆解图、bad-data overlay、relative-drop outlier
overlay、plus/minus disagreement map，以及只在 candidate 区域显示的 severity map。

当前 MVP-3R gate 必须保持 `conditional_go` 且 `mvp4_allowed=false`，原因是
阈值仍是 provisional，且 plus/minus disagreement 仍不可忽略。只有 gate report
明确 `go`，且未声称 final labels，才允许进入 MVP-4。
