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
- environment / outlier warnings

低井斜或 RelBearing 不可靠位置只降低 confidence，不直接删除。

## Review And Gate

MVP-3 完成后必须输出 candidate audit 和 review figures。只有 gate report 明确
`go` 或 `conditional_go`，且未声称 final labels，才允许进入 MVP-4。
