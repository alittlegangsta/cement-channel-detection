# Experiment Protocol  
## 固井窜槽智能检测项目实验协议

本文件定义本项目所有实验的命名、前置条件、运行流程、记录规范、结果归档、质量门槛和 go/no-go 判断标准。

本文件必须与以下文件保持一致：

```text
README.md
AGENTS.md
docs/report.md
docs/data_contract.md
docs/development_workflow.md
configs/*.yaml
```

其中：

- `README.md`：项目入口和开发流程；
- `AGENTS.md`：AI agent 与开发行为规则；
- `docs/report.md`：完整技术方案、物理原则、公式、标签规则、模型路线；
- `docs/data_contract.md`：数据结构、HDF5 Schema、字段、单位和版本约束；
- `docs/development_workflow.md`：日常开发流程；
- `docs/experiment_protocol.md`：本文件，负责实验规范。

如果本文件与 `docs/report.md` 冲突，以 `docs/report.md` 为准，并同步修订本文件。

---

## 1. 实验总原则

本项目的实验不是简单跑训练脚本，而是围绕数据可信、标签可信、模型可信和解释可信逐级推进。

核心原则：

```text
先验证数据，再生成标签
先审计标签，再提取特征
先做相关性验证，再训练模型
先做物理 baseline，再做深度模型
先做 XSI-only，再做 XSI+CAST 融合解释
先做可解释性检查，再输出工程结论
```

任何实验如果缺少以下内容，均视为不可复现：

```text
experiment_id
git_commit
branch
config files
config hash
data_version
label_version
feature_version
schema_version
random seed
host
start_time
end_time
metrics
logs
figures
notes
failure_cases
```

一句话原则：

> 没有 manifest 的实验，不算完成；没有图的实验，不可信；没有负/正对照的模型结果，不可作为科学结论。

---

## 2. 实验阶段划分

实验必须按项目 MVP 顺序推进，不得跳过前序阶段直接训练深度模型。

| 阶段 | 名称 | 目标 | 允许产出 | 禁止事项 |
|---|---|---|---|---|
| EXP-0 | 环境与工程骨架检查 | 确认项目可运行 | smoke test、lint、env check | 不处理真实数据 |
| EXP-1 | 数据契约与 manifest | 识别原始数据结构 | manifest、schema report | 不生成标签 |
| EXP-2 | XSI / CAST QC | 评估原始数据质量 | QC 报告、质量掩码、预览图 | 不训练模型 |
| EXP-3 | 深度与方位对齐 | 深度插值、高边归一化、RelBearing 验证 | alignment HDF5、旋转图、lag 图 | 不直接训练 |
| EXP-4 | 强弱侧与有效窗口 | 建模偏心、强弱侧、质量窗口、异常窗口 | side weights、effective window | 不丢弃全方位 Side |
| EXP-5 | 弱标签生成与审计 | 生成弱标签、置信度、对象级标签 | labels、label report、对象表 | 不把标签当绝对真值 |
| EXP-6 | 跨模态相关性验证 | 验证 XSI 与 CAST 是否存在稳定关系 | 消融报告、go/no-go | 不上深度模型 |
| EXP-7 | 物理 baseline | 用物理特征检验可学习性 | baseline metrics、feature importance | 不追求复杂模型 |
| EXP-8 | XSI-only 模型 | 验证仅凭 XSI 的预测能力 | XSI-only model、metrics、XAI | 禁止输入 CAST |
| EXP-9 | XSI+CAST 融合解释 | 辅助专家多模态解释 | fusion report、uncertainty | 不用于证明 XSI-only 能力 |
| EXP-10 | 专家复核与发布门槛 | 检查高风险样本和上线指标 | expert review、release report | 不自动替代人工决策 |

---

## 3. 实验命名规范

### 3.1 experiment_id 格式

推荐格式：

```text
<stage>_<task>_<date>_<run>
```

示例：

```text
exp01_manifest_20260518_r001
exp02_xsi_qc_20260518_r001
exp03_relbearing_alignment_20260518_r002
exp05_label_v001_20260518_r001
exp06_correlation_ablation_20260518_r001
exp07_physics_baseline_20260518_r001
exp08_xsi_only_stft_stc_20260518_r001
```

日期格式：

```text
YYYYMMDD
```

run 编号：

```text
r001, r002, r003, ...
```

---

### 3.2 版本命名

数据版本：

```text
data_v001
data_v002
```

标签版本：

```text
label_v001
label_v002
```

特征版本：

```text
feature_physics_v001
feature_stft_v001
feature_stc_v001
feature_apes_v001
```

schema 版本：

```text
schema_v001
```

模型版本：

```text
model_baseline_v001
model_xsi_only_v001
model_fusion_v001
```

---

### 3.3 输出目录命名

推荐输出结构：

```text
/home/xiaoj/cement-channel-data/
├─ manifests/
├─ processed/
├─ features/
├─ reports/
├─ logs/
└─ tmp/
```

实验输出建议：

```text
reports/experiments/<experiment_id>/
logs/<experiment_id>.log
manifests/<experiment_id>.json
```

服务器上建议：

```text
/home/你的用户名/logs/<experiment_id>.log
/home/你的用户名/models/<experiment_id>/
/home/你的用户名/project/cement-channel-detection/experiments/manifests/<experiment_id>.json
```

---

## 4. 实验前置条件

任何实验运行前必须确认：

```text
当前分支正确
git status 干净或已记录未提交修改
Python 环境正确
配置文件存在
输入文件存在
输出目录存在
不会覆盖旧结果
随机种子已设置
日志路径已设置
manifest 路径已设置
```

推荐检查命令：

```bash
git status
git branch --show-current
make test-smoke
make test
make lint
python scripts/00_check_env.py
```

---

## 5. 实验 Manifest 规范

每次实验必须生成 manifest。

推荐路径：

```text
experiments/manifests/<experiment_id>.json
```

或本地数据目录：

```text
/home/xiaoj/cement-channel-data/manifests/<experiment_id>.json
```

---

### 5.1 Manifest 顶层字段

```json
{
  "experiment_id": "exp07_physics_baseline_20260518_r001",
  "stage": "EXP-7",
  "task": "physics_baseline",
  "status": "completed",
  "created_at": "2026-05-18T10:00:00-07:00",
  "started_at": "2026-05-18T10:01:00-07:00",
  "ended_at": "2026-05-18T10:20:00-07:00",
  "duration_seconds": 1140,
  "owner": "xiaoj",
  "host": "local-wsl2",
  "notes": "Initial physics baseline on tiny sample."
}
```

---

### 5.2 Git 信息

```json
{
  "git": {
    "repo": "cement-channel-detection",
    "branch": "feature/physics-baseline",
    "commit": "abcdef123456",
    "is_dirty": false,
    "remote": "origin"
  }
}
```

如果 `is_dirty = true`，必须记录未提交文件列表。

---

### 5.3 环境信息

```json
{
  "environment": {
    "os": "WSL2 Ubuntu",
    "python": "3.10",
    "interpreter": "/home/dministrator/micromamba/envs/cement_env/bin/python",
    "env_manager": "micromamba",
    "env_name": "cement_env",
    "cuda_visible_devices": null,
    "gpu": null
  }
}
```

服务器实验应记录：

```json
{
  "environment": {
    "os": "Ubuntu 18.04.6",
    "cuda_visible_devices": "1,2",
    "gpu": ["A100-40GB", "A100-40GB"],
    "driver_version": "...",
    "cuda_version": "...",
    "conda_env": "cement-channel"
  }
}
```

---

### 5.4 输入数据版本

```json
{
  "inputs": {
    "data_version": "data_v001",
    "schema_version": "schema_v001",
    "label_version": "label_v001",
    "feature_version": "feature_physics_v001",
    "raw_files": [
      "/home/xiaoj/cement-channel-data/raw/well_001/XSI.mat",
      "/home/xiaoj/cement-channel-data/raw/well_001/CAST.mat"
    ],
    "input_hdf5": "/home/xiaoj/cement-channel-data/processed/aligned_data_v001.h5"
  }
}
```

---

### 5.5 配置文件与 hash

```json
{
  "configs": {
    "paths": "configs/paths.local.yaml",
    "preprocess": "configs/preprocess.yaml",
    "alignment": "configs/alignment.yaml",
    "label": "configs/label_v001.yaml",
    "feature": "configs/feature_stft.yaml",
    "train": "configs/train_baseline.yaml",
    "eval": "configs/eval.yaml"
  },
  "config_hash": {
    "paths": "sha256:...",
    "preprocess": "sha256:...",
    "alignment": "sha256:...",
    "label": "sha256:...",
    "feature": "sha256:...",
    "train": "sha256:...",
    "eval": "sha256:..."
  }
}
```

---

### 5.6 输出文件

```json
{
  "outputs": {
    "hdf5": [
      "/home/xiaoj/cement-channel-data/processed/aligned_data_v001.h5"
    ],
    "figures": [
      "/home/xiaoj/cement-channel-data/reports/experiments/exp03/rotation_ablation.png"
    ],
    "reports": [
      "/home/xiaoj/cement-channel-data/reports/experiments/exp03/alignment_report.md"
    ],
    "logs": [
      "/home/xiaoj/cement-channel-data/logs/exp03_relbearing_alignment_20260518_r001.log"
    ],
    "models": []
  }
}
```

---

### 5.7 指标记录

```json
{
  "metrics": {
    "recall": null,
    "precision": null,
    "f1": null,
    "aucpr": null,
    "azimuth_iou": null,
    "depth_iou": null,
    "ece": null,
    "brier": null,
    "review_segments_per_100m": null
  }
}
```

不适用的指标可以为 `null`，但字段应尽量保留。

---

### 5.8 Go / No-Go 结论

```json
{
  "go_no_go": {
    "decision": "go",
    "reason": "RelBearing plus sign improves circular correlation and physical baseline beats random control.",
    "blocking_issues": [],
    "warnings": [
      "Low inclination interval has high uncertain ratio."
    ],
    "next_stage_allowed": "EXP-8"
  }
}
```

允许取值：

```text
go
no_go
conditional_go
```

---

## 6. 配置管理协议

实验必须由配置驱动，不得在代码中硬编码路径、阈值、窗口、采样率和模型参数。

配置文件位于：

```text
configs/
```

常用配置：

```text
configs/paths.local.yaml
configs/paths.server.yaml
configs/preprocess.yaml
configs/alignment.yaml
configs/label_v001.yaml
configs/feature_stft.yaml
configs/feature_stc.yaml
configs/train_baseline.yaml
configs/train_xsi_only.yaml
configs/train_fusion.yaml
configs/eval.yaml
```

提交到 Git 的是：

```text
*.example.yaml
公共默认配置
非敏感配置
```

不提交到 Git 的是：

```text
configs/paths.local.yaml
configs/paths.server.yaml
包含真实路径、服务器路径、用户名、密钥的配置
```

---

## 7. 随机性与可复现性

所有涉及随机性的实验必须设置随机种子。

应记录：

```text
python random seed
numpy random seed
torch random seed
data split seed
augmentation seed
```

如果使用 PyTorch，应尽量记录：

```text
torch version
cuda version
cudnn deterministic setting
cudnn benchmark setting
```

示例：

```json
{
  "random_seed": {
    "python": 42,
    "numpy": 42,
    "torch": 42,
    "split": 42
  }
}
```

注意：

- 完全确定性可能降低 GPU 性能；
- 如果无法保证完全确定性，必须在 manifest 中说明。

---

## 8. 数据实验协议

### 8.1 EXP-1：Manifest 实验

目标：

```text
读取原始 .mat 文件结构
记录变量名、shape、dtype、深度范围
生成 raw_file_inventory.csv
生成 data_manifest.json
```

必须输出：

```text
raw_file_inventory.csv
data_manifest.json
manifest_report.md
```

必须记录：

```text
文件路径
文件大小
变量名
shape
dtype
depth range
time sample count
receiver count
side count
cast azimuth count
Inc / RelBearing 是否存在
```

Go 条件：

```text
能识别 XSI
能识别 CAST
能识别 depth
能识别 Inc / RelBearing 或明确缺失
数据 shape 可解释
```

No-Go 条件：

```text
无法读取 .mat
缺少关键变量
维度无法解释
深度轴缺失且无法推断
```

---

### 8.2 EXP-2：QC 实验

目标：

```text
检查 XSI 原始波形质量
检查 CAST 成像质量
生成质量权重和质量掩码
```

必须输出：

```text
xsi_qc_report.md
cast_qc_report.md
quality_masks.h5
qc_summary.png
low_confidence_intervals.csv
```

XSI QC 至少检查：

```text
dead receiver
dead side
saturation ratio
clipping ratio
noise floor
SNR
first arrival time
receiver consistency
gain change flag
```

CAST QC 至少检查：

```text
missing sector ratio
sector quality weight
low confidence mask
collar / geometry mask if available
```

Go 条件：

```text
关键数据质量可评估
质量权重可生成
低质量区域可标记
```

No-Go 条件：

```text
大面积缺失但未标记
XSI / CAST 质量无法评估
QC 输出无法与深度和方位对齐
```

---

### 8.3 EXP-3：Alignment 实验

目标：

```text
完成深度对齐
完成 RelBearing 高边坐标归一化
验证 RelBearing 正负号
处理低井斜段
估计局部 depth lag
```

MVP-2 的第一步必须是 depth axis audit。该 audit 只允许读取
`CAST.Depth`、`XSILMR{receiver}.Depth` 和 `Depth_inc`，输出：

```text
depth_axis_audit_report.md
depth_axis_audit_report.json
```

Depth audit Go 条件：

```text
CAST / XSI / pose 三套 depth 轴存在共同 overlap
depth 基本递增
无严重 NaN / Inf
XSI receiver depth 轴大体一致
```

若只有 depth unit 未确认，可为 `conditional_go`，但必须记录 warning。

必须输出：

```text
aligned_coordinates.h5
alignment_report.md
rotation_ablation.png
local_depth_lag.png
orientation_confidence.png
```

必须比较：

```text
+RelBearing
-RelBearing
no rotation
random rotation
```

MVP-2 RelBearing sign validation 必须输出：

```text
relbearing_sign_validation_report.md
relbearing_sign_validation_report.json
configs/alignment.relbearing.example.yaml
```

若 small-slice 中没有可用的 CAST/XSI 共同方位证据，或 plus/minus 与
no-rotation/random-rotation 无法区分，validation decision 必须为
`insufficient_evidence`，不得硬选符号，不得进入正式 alignment 或 MVP-3，除非人工确认
或后续明确批准 dual-sign / ablation 协议。

若初始 small-slice 与 proposed depth grid 没有共同覆盖，可执行 Stage 6b：
overlap-targeted small-slice + RelBearing evidence augmentation。该阶段只能在
`depth_grid_proposal.json` 的共同 overlap 内读取不超过小片段规模的数据，默认窗口
不超过 2.0 m，并输出：

```text
small_slice_overlap_v001.npz
small_slice_overlap_summary_v001.json
depth_resample_overlap_preview_v001.npz
depth_resample_overlap_preview_report.md/json
relbearing_sign_validation_overlap_report.md/json
```

Stage 6b 仍不得选择最终 RelBearing sign；如果 plus/minus 仍无法区分，必须继续
`insufficient_evidence` 并要求人工确认或 dual-sign / ablation。

Go 条件：

```text
至少一个 RelBearing 符号明显优于未旋转
低井斜段被标记 uncertain
局部 depth lag 有记录
方位坐标在 [0, 360)
```

No-Go 条件：

```text
RelBearing 正负号无法区分
对齐后相关性不提升
低井斜段未处理
深度错位严重且无置信度标记
```

---

### 8.4 EXP-4：强弱侧与有效窗口实验

目标：

```text
估计强侧 / 弱侧
生成 side_quality_weight
生成 W_quality
生成 W_anomaly
生成 W_eff
```

必须输出：

```text
strong_weak_side_report.md
strong_weak_polar.png
effective_window_overlay.png
side_quality_weights.h5
```

Go 条件：

```text
theta_strong_deg 可生成
side_quality_weight 合理
W_quality 与 W_anomaly 可区分
W_eff 不等于只看强侧
全方位 Side 被保留
```

No-Go 条件：

```text
强侧估计与异常方位混淆严重
只保留强侧导致弱侧异常丢失
质量权重全为常数且无解释
```

---

### 8.5 EXP-5：弱标签实验

目标：

```text
基于 CAST 生成弱标签
计算 label_confidence
生成 uncertain_mask
生成对象级窜槽标签
```

必须输出：

```text
labels_v001.h5
label_report.md
label_preview.png
label_confidence.png
channel_objects.csv
```

必须包含：

```text
absolute Zc threshold
adaptive baseline
relative drop
azimuth gradient
depth continuity
HardQualityMask
label_confidence
object-level labels
```

Go 条件：

```text
presence / severity 编码合法
uncertain 样本被标记
label_confidence 在 [0, 1]
对象级标签可解释
标签图与 CAST 图大体一致
```

No-Go 条件：

```text
只用固定阈值
方位梯度单独触发高置信标签
无 uncertain mask
label_confidence 缺失
标签与 QC 冲突
```

---

## 9. 相关性验证协议

### 9.1 必做实验

训练模型前必须完成：

```text
方位循环互相关
强弱侧加权消融
CAST 有效窗口消融
随机方位打乱
深度错位检验
负对照
正对照
物理 baseline
```

---

### 9.2 方位循环互相关

目标：

```text
验证 XSI 方位特征与 CAST 方位异常是否在正确 RelBearing 后对齐
```

比较对象：

```text
no rotation
+RelBearing
-RelBearing
random rotation
```

Go 条件：

```text
正确 RelBearing 后 rho(k) 峰值接近 k=0
正确旋转显著优于未旋转和随机旋转
```

---

### 9.3 随机方位打乱

目标：

```text
检查模型是否真的利用方位对应关系
```

方法：

```text
对 CAST 方位随机循环平移
重新计算相关性或训练 baseline
```

Go 条件：

```text
随机打乱后性能明显下降
```

No-Go 条件：

```text
打乱后性能仍然很高
```

说明：

```text
若打乱后性能不下降，模型可能学习的是深度趋势、全局背景或数据泄漏。
```

---

### 9.4 深度错位检验

目标：

```text
检查模型是否真的利用局部深度对应关系
```

方法：

```text
人为将 CAST 标签上下平移若干深度采样点
比较性能变化
```

Go 条件：

```text
错位后性能明显下降
```

No-Go 条件：

```text
错位后性能基本不变
```

---

### 9.5 负对照实验

负对照输入包括：

```text
depth index
logging order
random noise
smoothed global Zc background
Inc / RelBearing only
well_id only
```

Go 条件：

```text
负对照不应显著优于随机
```

No-Go 条件：

```text
负对照模型取得高性能
```

---

### 9.6 正对照实验

正对照可以使用合成或半合成异常：

```text
CAST 中插入低 Zc 方位条带
XSI 中插入套管波增强
XSI 中插入衰减斜率降低
STC 中插入峰值异常
人为设置已知 RelBearing 旋转角
```

Go 条件：

```text
pipeline 能找回已知异常的位置、方位和深度
```

---

## 10. 物理 Baseline 实验协议

物理 baseline 是深度模型前的硬门槛。

### 10.1 推荐输入

```text
套管波时间窗能量
流体波时间窗能量
地层波相关能量
13 接收器衰减斜率
STC 套管波慢度峰
STC 流体波慢度峰
强弱侧能量比
有效窗口能量
姿态置信度
深度对齐置信度
质量权重
```

### 10.2 推荐模型

```text
Logistic Regression
Random Forest
Gradient Boosting
XGBoost / LightGBM
```

### 10.3 必须报告

```text
AUCPR
Recall
Precision
F1
IoU
Azimuth IoU
object-level recall
feature importance
per-well metrics
failure cases
```

### 10.4 Go 条件

```text
物理 baseline 显著优于随机
物理 baseline 优于负对照
错位和打乱后性能下降
重要特征具有物理解释
```

### 10.5 No-Go 条件

```text
baseline 接近随机
负对照同样高分
特征重要性全落在无物理意义变量
跨井泛化严重失败
```

如果 baseline 不通过，不允许进入 XSI-only 深度模型。

---

## 11. XSI-only 模型实验协议

### 11.1 输入规则

允许输入：

```text
XSI waveform
STFT features
STC / physics features
Inc
RelBearing
orientation_confidence
side_quality_weight
effective_window_mask
quality_flags
```

禁止输入：

```text
CAST Zc
CAST label map
CAST-derived direct feature maps
future labels
test labels
```

### 11.2 必须有 input guard

训练代码必须在 `model_mode = xsi_only` 时阻断 CAST 输入。

如果发现以下字段进入模型输入，应直接报错：

```text
/aligned/cast_zc
/aligned/cast_zc_to_xsi8
/label/presence as input
/label/severity as input
CAST direct feature map
```

### 11.3 必须报告

```text
AUCPR
Recall
Precision
F1
Azimuth IoU
Depth IoU
object-level recall
object-level precision
high_severity_recall
ECE
Brier Score
review_segments_per_100m
failure cases
XAI report
```

### 11.4 Go 条件

```text
优于物理 baseline 或在特定指标上有明确提升
在跨井测试中保持稳定
负对照低分
错位 / 打乱后性能下降
解释落在合理 XSI 时间窗和 STC 模态
```

### 11.5 No-Go 条件

```text
只在随机划分上高分
跨井测试失败
疑似标签泄漏
解释落在无意义区域
负对照也高分
```

---

## 12. XSI+CAST 融合解释实验协议

XSI+CAST 融合模型只能用于：

```text
专家辅助解释
多模态一致性验证
标签质量审计
错误案例分析
```

不得用于证明 XSI-only 能力。

### 12.1 允许输入

```text
XSI
CAST
pose
quality weights
effective windows
QC masks
```

### 12.2 必须输出

```text
XSI evidence strength
CAST evidence strength
cross-modal consistency
uncertainty score
quality flags
whether CAST dominates
whether XSI supports
expert review required flag
```

### 12.3 消融要求

必须比较：

```text
simple concat
global average fusion
symmetric cross-attention
asymmetric cross-attention
without quality gate
with random CAST azimuth shuffle
with wrong RelBearing sign
```

---

## 13. 可解释性实验协议

模型解释必须回到：

```text
depth
azimuth
XSI side
millisecond time window
frequency band
STC slowness peak
CAST low-Zc azimuth
strong / weak side
quality flags
uncertainty
```

必须执行解释可信度检查：

```text
random label test
model parameter randomization
input perturbation
side perturbation
STC consistency check
counterfactual masking
```

解释不可信条件：

```text
归因落在噪声段
归因落在缺失段
归因落在泥浆直达波无关区域
归因落在低质量 CAST 扇区
随机标签训练仍出现稳定物理解释
遮挡关键区域预测不变
```

---

## 14. 指标协议

禁止只看 Accuracy。

必须根据任务报告以下指标。

### 14.1 检出能力

```text
Recall
Precision
F1
AUCPR
```

### 14.2 方位定位

```text
Azimuth IoU
circular angle error
azimuth center error
```

### 14.3 深度定位

```text
Depth IoU
top depth error
bottom depth error
continuous segment recall
```

### 14.4 对象级指标

```text
object-level recall
object-level precision
object-level IoU
channel center azimuth error
channel depth length error
```

### 14.5 严重程度

```text
Macro-F1
Ordinal MAE
high_severity_recall
missed_high_severity_channels
```

### 14.6 校准能力

```text
ECE
Brier Score
Reliability curve
```

### 14.7 工程成本指标

```text
review_segments_per_100m
false_alarm_segments_per_100m
recall_at_fixed_review_budget
precision_at_95_recall
```

---

## 15. Go / No-Go 标准

### 15.1 进入标签生成前

Go 条件：

```text
manifest 完成
QC 完成
关键 shape 明确
depth / Inc / RelBearing 可用
```

No-Go 条件：

```text
关键变量无法识别
XSI / CAST shape 不可解释
depth 缺失且无法恢复
```

---

### 15.2 进入特征提取前

Go 条件：

```text
深度对齐完成
RelBearing 符号候选已验证
低井斜段已标记
强弱侧和有效窗口可生成
```

No-Go 条件：

```text
RelBearing 方向完全不可信
深度错位严重且无标记
有效窗口退化为无意义常数
```

---

### 15.3 进入物理 baseline 前

Go 条件：

```text
标签规则可解释
label_confidence 可用
uncertain mask 可用
对象级标签可生成
负/正对照 pipeline 可运行
```

No-Go 条件：

```text
标签只是固定阈值
没有置信度
没有 uncertain mask
标签与 QC 冲突
```

---

### 15.4 进入 XSI-only 深度模型前

Go 条件：

```text
物理 baseline 优于随机
负对照低分
正对照通过
错位 / 打乱性能下降
跨模态相关性成立
```

No-Go 条件：

```text
baseline 不优于随机
打乱后性能不降
错位后性能不降
负对照高分
```

---

### 15.5 进入融合解释模型前

Go 条件：

```text
XSI-only 结果可信
XSI-only 防泄漏已验证
XAI 能回到物理时间窗
融合模型目标明确为解释辅助
```

No-Go 条件：

```text
试图用融合模型证明 XSI-only 能力
CAST 无条件主导判定
没有不确定性输出
```

---

## 16. 结果归档规范

每次实验至少保存：

```text
manifest.json
config snapshots 或 config hash
log file
metrics.json
figures/
failure_cases.csv
go_no_go.json
```

推荐结构：

```text
reports/experiments/<experiment_id>/
├─ report.md
├─ metrics.json
├─ go_no_go.json
├─ figures/
│  ├─ qc_summary.png
│  ├─ alignment_plot.png
│  ├─ label_preview.png
│  └─ failure_cases.png
└─ failure_cases.csv
```

模型实验额外保存：

```text
models/<experiment_id>/
├─ model.pt
├─ config.yaml
├─ metrics.json
├─ calibration.json
└─ model_card.md
```

模型权重不得提交到 Git。

---

## 17. 失败实验处理

失败实验也应记录。

失败时必须保存：

```text
manifest
error log
failed config
failure reason
partial outputs if useful
next action
```

失败状态：

```text
failed
cancelled
blocked
no_go
```

不要删除失败实验。失败实验能避免重复踩坑。

---

## 18. 本地与服务器实验边界

### 18.1 本地适合运行

```text
manifest
schema validation
tiny sample
unit test
integration test
small QC
small alignment
label preview
small STFT
small physical baseline
visualization debug
```

### 18.2 服务器适合运行

```text
full-well preprocessing
full STC / APES
large HDF5 / Zarr generation
GPU training
long inference
large XAI batch
formal evaluation
```

### 18.3 不应在本地运行

```text
全井段高分辨率 STC
全量 APES
多 GPU 训练
长时间正式训练
大规模推理
```

---

## 19. 实验安全规则

禁止：

```text
覆盖旧实验
覆盖旧 HDF5
修改原始 raw 数据
把真实数据提交到 Git
把模型权重提交到 Git
把 .env 提交到 Git
用未记录配置运行实验
训练时动态读取大 .mat
DataLoader 中动态计算 STC / APES
XSI-only 输入 CAST
随机逐点划分相邻深度
```

如果确实需要覆盖，必须显式指定：

```bash
--overwrite true
```

并在 manifest 中记录原因。

---

## 20. 实验前 Checklist

每次实验前确认：

```text
[ ] 当前分支正确
[ ] git status 已检查
[ ] 输入数据存在
[ ] 输出目录存在
[ ] 不会覆盖旧结果
[ ] 配置文件存在
[ ] 配置 hash 可记录
[ ] data_version 已知
[ ] label_version 已知或不适用
[ ] feature_version 已知或不适用
[ ] schema_version 已知
[ ] random seed 已设置
[ ] 日志路径已设置
[ ] manifest 路径已设置
[ ] tiny sample 已先跑通
```

---

## 21. 实验后 Checklist

每次实验后确认：

```text
[ ] manifest 已生成
[ ] log 已保存
[ ] metrics 已保存
[ ] figures 已保存
[ ] config hash 已保存
[ ] git_commit 已保存
[ ] data_version 已保存
[ ] label_version 已保存
[ ] feature_version 已保存
[ ] go/no-go 已判断
[ ] failure cases 已记录
[ ] 没有大文件被加入 Git
```

---

## 22. MVP-1 Gate Report

MVP-1 完成前应生成 gate report，汇总 manifest、MAT metadata、struct probe、raw variable
mapping、small slice、tiny HDF5 和 initial QC skeleton 的状态。

Go / No-Go 原则：

- 任一输入缺失、schema validation error、small slice error、tiny HDF5 error 或 QC error
  均为 `no_go`；
- 若只有 depth unit unknown、time unit unknown 或 tiny prototype 未做正式 alignment 等不确定性，
  可为 `conditional_go`；
- `go` 或 `conditional_go` 只允许进入 MVP-2 的对齐与方位归一化阶段，不允许跳到 label 或模型训练。

---

## 22. 当前阶段建议实验顺序

当前项目处于工程骨架与 MVP-1 准备阶段。

建议实验顺序：

```text
EXP-0: environment smoke test
EXP-1: raw manifest on tiny sample
EXP-1b: raw manifest on one local well
EXP-2: XSI / CAST QC on tiny sample
EXP-3: alignment dry-run on tiny sample
EXP-5: label generation on synthetic / tiny sample
EXP-6: correlation sanity checks on synthetic / tiny sample
EXP-7: first physics baseline on tiny sample
```

只有 tiny sample 全流程通过后，才进入真实小井段。

---

## 23. 最终原则

实验协议的目标不是增加流程负担，而是防止以下问题：

```text
模型学到错位数据
模型学到 CAST 泄漏
模型学到井号或深度趋势
模型学到偏心伪影
模型学到低质量成像噪声
实验无法复现
高分不可解释
```

一句话原则：

> 任何实验都必须能回答：用了什么数据、什么配置、什么代码、什么标签、什么特征、得到什么结果、为什么可信、是否允许进入下一阶段。
