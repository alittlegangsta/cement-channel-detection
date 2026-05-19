# AGENTS.md  
## 固井窜槽智能检测与声波波形可解释分析项目 Agent 开发规范

本文件是本仓库中所有 AI coding agent、Codex CLI、Codex IDE 插件、自动化脚本、后续开发者必须遵守的项目级规则。

本项目的最高技术依据是：

```text
docs/report.md
```

README 负责项目入口、开发流程和运行说明；`docs/report.md` 负责完整技术路线、物理原则、公式、标签规则、模型边界和验收标准；本 `AGENTS.md` 负责约束 agent 在本仓库中的行为。

任何 agent 在修改代码、文档、配置或脚本时，都必须优先遵守：

```text
1. 本 AGENTS.md
2. docs/report.md
3. README.md
4. docs/* 中的专门规范
5. configs/* 中的实验配置
```

---

## 1. 项目身份

项目名称：

```text
cement-channel-detection
```

项目目标：

```text
利用 LWD 单极声波全波列数据 XSI 与 CAST 超声井壁成像数据，构建科学、可解释、可审计、可复现的固井窜槽智能检测系统。
```

本项目不是单纯的深度学习分类项目，而是一个强物理约束的数据工程、弱监督标签、声学特征、模型训练、可解释性和专家复核闭环项目。

核心原则：

```text
数据先于模型
验证先于训练
物理解释先于黑盒分数
弱标签必须带置信度
XSI-only 与 XSI+CAST 必须严格区分
没有 QC、对齐、消融、负/正对照和物理 baseline，不允许进入深度模型阶段
```

---

## 2. 当前本地开发环境

当前本地项目路径：

```text
/home/xiaoj/cement-channel-detection
```

本地开发环境：

```text
Windows + WSL2 Ubuntu
VS Code Insiders
micromamba
```

本地 micromamba 环境：

```text
env name: cement_env
python: 3.10
interpreter: /home/dministrator/micromamba/envs/cement_env/bin/python
```

VS Code 配置文件：

```text
/home/xiaoj/cement-channel-detection/.vscode/settings.json
```

VS Code Python 解释器配置：

```json
{
  "python.defaultInterpreterPath": "/home/dministrator/micromamba/envs/cement_env/bin/python",
  "python-envs.defaultEnvManager": "ms-python.python:system"
}
```

注意：

- 不要擅自把 `/home/dministrator/...` 改成 `/home/Administrator/...`。
- 以当前机器真实可执行路径为准。
- 如果需要检查解释器，使用：

```bash
/home/dministrator/micromamba/envs/cement_env/bin/python --version
```

---

## 3. 服务器运行环境

服务器用于全量数据处理和 GPU 训练，不用于日常手工改代码。

服务器约束：

```text
OS: Ubuntu 18.04.6
GPU: 3 × A100 40GB
数据、模型、日志必须保存到 /home 下
默认不要使用 GPU0
训练默认使用 CUDA_VISIBLE_DEVICES=1,2
```

推荐服务器目录：

```text
/home/你的用户名/project
/home/你的用户名/data
/home/你的用户名/models
/home/你的用户名/logs
```

服务器职责：

```text
git pull
激活 conda/micromamba 环境
运行预处理脚本
运行离线特征提取
运行 GPU 训练
运行推理
保存日志
保存模型
保存实验 manifest
```

服务器禁止：

```text
手工乱改代码
直接修改 main 分支代码
把数据写入根目录 /
把模型、日志、HDF5 提交到 Git
在训练循环中动态读取大 .mat 文件
```

---

## 4. Git 分支规范

本项目采用：

```text
main
dev
feature/*
```

分支职责：

| 分支 | 作用 |
|---|---|
| `main` | 稳定版，只放经过验证的阶段成果 |
| `dev` | 日常集成分支 |
| `feature/*` | 每个具体小任务一个分支 |

Agent 必须遵守：

1. 不得直接在 `main` 上开发。
2. 新功能必须从 `dev` 创建 `feature/*` 分支。
3. 每个 feature 分支只做一个明确任务。
4. 合并前必须运行测试、lint 和环境检查。
5. 阶段稳定后，才允许 `dev → main`。

推荐流程：

```bash
git checkout dev
git pull origin dev
git checkout -b feature/<task-name>

# 修改代码 / 文档 / 配置

make test
make lint
python scripts/00_check_env.py

git status
git add .
git commit -m "<type>: <message>"
git push -u origin feature/<task-name>
```

提交信息建议使用：

```text
chore: initialize project scaffold
docs: add data contract draft
config: add preprocessing config
test: add alignment unit tests
feat: implement relbearing rotation
fix: correct circular distance calculation
refactor: split label confidence utilities
```

---

## 5. 仓库目标结构

Agent 应保持或逐步补齐以下结构：

```text
cement-channel-detection/
├─ README.md
├─ AGENTS.md
├─ pyproject.toml
├─ environment.local.yml
├─ environment.server.yml
├─ .gitignore
├─ .gitattributes
├─ .env.example
├─ Makefile
│
├─ .vscode/
│  └─ settings.json
│
├─ docs/
│  ├─ report.md
│  ├─ data_contract.md
│  ├─ development_workflow.md
│  ├─ experiment_protocol.md
│  ├─ server_runbook.md
│  ├─ labeling_protocol.md
│  ├─ model_design.md
│  ├─ evaluation_protocol.md
│  └─ decisions/
│     ├─ ADR-0001-repo-structure.md
│     ├─ ADR-0002-data-format.md
│     └─ ADR-0003-label-versioning.md
│
├─ configs/
│  ├─ paths.local.example.yaml
│  ├─ paths.server.example.yaml
│  ├─ preprocess.yaml
│  ├─ alignment.yaml
│  ├─ label_v001.yaml
│  ├─ feature_stft.yaml
│  ├─ feature_stc.yaml
│  ├─ train_baseline.yaml
│  ├─ train_xsi_only.yaml
│  ├─ train_fusion.yaml
│  └─ eval.yaml
│
├─ src/
│  └─ cement_channel/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ data/
│     ├─ qc/
│     ├─ alignment/
│     ├─ labels/
│     ├─ features/
│     ├─ models/
│     ├─ training/
│     ├─ evaluation/
│     ├─ xai/
│     ├─ visualization/
│     └─ utils/
│
├─ scripts/
│  ├─ 00_check_env.py
│  ├─ 01_build_manifest.py
│  ├─ 02_run_qc.py
│  ├─ 03_align_data.py
│  ├─ 04_generate_labels.py
│  ├─ 05_extract_features.py
│  ├─ 06_train_baseline.py
│  ├─ 07_train_xsi_only.py
│  ├─ 08_evaluate.py
│  └─ server/
│     ├─ pull_and_run.sh
│     ├─ train_baseline.sh
│     ├─ train_xsi_only.sh
│     └─ monitor_gpu.sh
│
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ smoke/
│  └─ fixtures/
│     └─ tiny_sample/
│
├─ notebooks/
│  ├─ 00_data_preview.ipynb
│  ├─ 01_alignment_debug.ipynb
│  └─ 02_label_debug.ipynb
│
├─ experiments/
│  ├─ README.md
│  └─ manifests/
│
├─ reports/
│  └─ figures/
│
└─ .github/
   └─ workflows/
      └─ ci.yml
```

---

## 6. 不允许提交到 Git 的内容

Agent 必须确保以下内容不被提交：

```text
data/
outputs/
artifacts/
checkpoints/
models/
logs/
*.mat
*.h5
*.hdf5
*.npy
*.npz
*.pt
*.pth
*.ckpt
*.onnx
*.log
.env
__pycache__/
.ipynb_checkpoints/
```

真实数据、HDF5 缓存、模型权重、训练日志必须放在 Git 外部。

若发现大文件被误加入暂存区，必须提醒用户并执行：

```bash
git rm --cached <file>
```

不得擅自删除用户真实数据文件，只能从 Git 索引中移除。

---

## 7. 基础命令

Agent 在完成任何代码修改后，应建议或执行以下检查：

```bash
make test-smoke
make test
make lint
python scripts/00_check_env.py
```

当前基础命令应通过：

```text
make test-smoke
make test
make lint
python scripts/00_check_env.py
```

如果 lint 失败，优先使用：

```bash
make fix
make lint
```

或：

```bash
python -m ruff check src scripts tests --fix
python -m ruff format src scripts tests
```

不得提交无法通过 smoke test、pytest、ruff 的代码。

---

## 8. Python 代码规范

### 8.1 基本要求

Python 代码应遵守：

```text
Python >= 3.10
类型注解优先
小函数
清晰命名
无硬编码路径
无隐藏全局状态
配置驱动
可测试
可复现
```

### 8.2 Import 规范

使用 ruff 管理 import 顺序。

推荐：

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
```

不要手动堆叠无序 import。

### 8.3 路径规范

禁止在代码中硬编码：

```python
"/home/xiaoj/cement-channel-detection"
"/home/xxx/data"
"/home/xxx/models"
"C:\\Users\\..."
```

路径必须来自：

```text
configs/paths.local.example.yaml
configs/paths.server.example.yaml
环境变量
命令行参数
```

例外：

- 测试中可使用 `tmp_path`；
- 文档中可写示例路径；
- `scripts/00_check_env.py` 可检查项目相对目录。

### 8.4 配置优先

阈值、窗口、采样率、滤波参数、STC 慢度网格、标签版本、输出路径、GPU 设置不得写死在核心代码中。

必须放到：

```text
configs/*.yaml
```

或命令行参数中。

### 8.5 日志规范

长期运行脚本必须使用 logging，不应只用 print。

至少记录：

```text
start time
config path
git commit
input files
output files
data_version
label_version
feature_version
host
GPU
random seed
warnings
errors
end time
```

---

## 9. 测试规范

测试目录：

```text
tests/
├─ unit/
├─ integration/
├─ smoke/
└─ fixtures/
   └─ tiny_sample/
```

测试分层：

| 类型 | 位置 | 作用 |
|---|---|---|
| smoke | `tests/smoke/` | 快速确认项目结构、import、基础命令 |
| unit | `tests/unit/` | 测试单个函数，如 circular distance、RelBearing 旋转 |
| integration | `tests/integration/` | 测试小样例完整流程 |
| fixtures | `tests/fixtures/` | 存放极小样例数据，不放真实大数据 |

### 9.1 必须优先测试的函数

Agent 实现以下功能时，必须同时添加测试：

```text
circular distance
angle wrap to [0, 360)
RelBearing + / - rotation
CAST 180 → XSI 8 side aggregation
low inclination confidence
depth lag estimation
weighted aggregation
quality mask logic
label candidate generation
label confidence calculation
object-level connected component logic
HDF5 schema validation
```

### 9.2 tiny sample 原则

`tests/fixtures/tiny_sample/` 只允许放极小、可公开或合成的数据。

不得放真实井数据、大 `.mat`、大 HDF5、模型权重。

---

## 10. 数据工程规则

### 10.1 原始数据

原始 `.mat` 文件不得直接进入训练循环。

禁止：

```text
PyTorch Dataset.__getitem__ 中读取大 .mat
DataLoader 动态做 STC / APES
训练时动态生成完整 HDF5
训练时动态做高成本深度对齐
```

正确流程：

```text
raw .mat
→ manifest
→ QC
→ alignment
→ HDF5 / Zarr / memmap
→ feature extraction
→ training
```

### 10.2 HDF5 / Zarr / Memmap

核心数据必须遵守数据契约。

统一维度顺序：

```text
depth → receiver → side → time
```

核心字段应与 `docs/report.md` 和 `docs/data_contract.md` 一致。

必须保存元信息：

```text
data_version
label_version
feature_version
git_commit
preprocess_config_hash
created_at
created_by
source_files
```

### 10.3 数据版本

任何下列变化都必须产生新版本：

```text
深度插值方式变化
RelBearing 符号方向变化
低井斜阈值变化
XSI Side 方位定义变化
CAST 聚合策略变化
标签阈值变化
自适应基线窗口变化
STFT 参数变化
STC 慢度网格变化
QC 规则变化
```

不得覆盖旧版本输出。

---

## 11. 物理与坐标规则

### 11.1 高边坐标归一化

所有 XSI Side 和 CAST 方位匹配前，必须转换到相对于高边的坐标系。

默认候选公式：

```text
theta_aligned = (theta_raw + RelBearing) mod 360
```

但必须同时测试：

```text
theta_aligned_plus  = (theta_raw + RelBearing) mod 360
theta_aligned_minus = (theta_raw - RelBearing) mod 360
```

最终符号必须通过：

```text
循环互相关
可视化
专家抽查
错误符号消融
```

确认。

### 11.2 低井斜处理

井斜过低时，高边方向不稳定。

必须生成：

```text
orientation_confidence
orientation_uncertain
low_inc_mask
```

低井斜段不得生成高置信方位标签。

### 11.3 圆周拓扑

方位是圆周，不是线性序列。

必须使用：

```text
circular distance
circular padding
sin / cos angle encoding
circular IoU
```

Side A 与 Side H 相邻，0° 与 360° 相邻。

---

## 12. 强弱侧规则

强侧通常表示更好的观测质量，但不等于真实窜槽方向。

弱侧信噪比可能低，但不能直接丢弃。

必须保留：

```text
strong side features
weak side features
strong-weak delta
strong-weak ratio
all-side robust statistics
effective-window features
quality weights
uncertainty flags
```

强弱侧估计必须写入：

```text
theta_strong_deg
theta_weak_deg
eccentricity_score
strong_side_confidence
side_quality_weight
cast_sector_quality_weight
```

如果 CAST 只有 Zc 而没有半径、回波幅度、travel time 等质量字段，则强侧估计置信度必须降低。

---

## 13. 有效窗口规则

CAST 180 方位用于识别：

```text
高质量观测方向
低阻抗异常方向
最大方位梯度异常方向
```

有效窗口应拆分为：

```text
W_quality
W_anomaly
W_eff = W_quality ∪ W_anomaly
```

不得把有效窗口误解为只看强侧。

模型最终仍应看到全方位结构：

```text
xsi_all_sides
xsi_eff_window
xsi_strong_weak_delta
quality_masks
```

---

## 14. 标签规则

### 14.1 弱标签原则

CAST 标签不是绝对真值，而是 weak label / teacher label。

标签必须包含：

```text
presence_label
severity_label
azimuth_span_deg
depth_span
label_confidence
uncertain_mask
evidence_type
label_version
channel_object_id
```

`presence_label` 编码：

```text
0  = 无窜槽
1  = 疑似窜槽
-1 = uncertain
```

### 14.2 标签生成

标签不得只依赖固定 `Zc < 2.5 MRayl`。

必须综合：

```text
物理保底阈值
局部自适应背景基线
相对下降幅度
方位动态梯度
纵向连续性
HardQualityMask
label_confidence
```

方位梯度项不能单独触发高置信窜槽标签，必须伴随一定程度低阻抗下降。

### 14.3 标签置信度

标签置信度必须综合：

```text
q_cast
q_xsi
c_orient
c_depth
c_continuity
quality_flags
```

质量一般的数据不应全部删除，应降低置信度；严重无效数据才进入 `HardQualityMask`。

---

## 15. 特征工程规则

### 15.1 XSI QC

必须支持或预留：

```text
xsi_dead_receiver_mask
xsi_dead_side_mask
xsi_saturation_ratio
xsi_clipping_ratio
xsi_noise_floor
xsi_snr
xsi_first_arrival_time
xsi_time_zero_shift
xsi_receiver_consistency
xsi_polarity_check
xsi_gain_change_flag
```

### 15.2 CAST QC

必须支持或预留：

```text
cast_missing_sector_ratio
cast_echo_amplitude_quality
cast_travel_time_quality
cast_radius_variation
cast_ovality
cast_thickness_anomaly
cast_collar_mask
cast_low_confidence_mask
cast_sector_quality_weight
```

### 15.3 STFT / ISTFT

STFT 必须可逆或近似可逆。

必须记录：

```text
window length
overlap
window function
boundary mode
reconstruction error
feature_version
```

重构误差过大时，该 STFT 配置不得作为可逆主分支。

### 15.4 STC / APES

STC、APES 属于高成本物理特征，不得在 DataLoader 中动态计算。

必须：

```text
离线预计算
写入 HDF5 / Zarr / NumPy memmap
保存 feature_version
保存参数 hash
保存代码 commit
```

---

## 16. 模型规则

### 16.1 先 baseline，后深度模型

必须先实现物理 baseline。

允许模型：

```text
Logistic Regression
Random Forest
Gradient Boosting
XGBoost / LightGBM
```

如果 baseline 完全无法优于随机，不得直接上深度模型。

### 16.2 XSI-only 模型

XSI-only 模型允许输入：

```text
XSI waveform
STFT features
STC / physics features
Inc
RelBearing
quality weights
effective window markers
```

XSI-only 模型禁止输入：

```text
CAST Zc
CAST label map
CAST-derived direct feature maps
```

CAST 只能用于：

```text
weak label
label confidence
effective-window generation
teacher signal
training supervision
```

推理阶段不得需要 CAST。

### 16.3 XSI+CAST 融合解释模型

XSI+CAST 融合模型允许输入 CAST，但只能用于：

```text
专家辅助解释
多模态一致性验证
标签质量审计
错误案例分析
```

不得用其高分证明 XSI-only 能力。

### 16.4 非对称交叉注意力

若实现非对称交叉注意力，必须包含：

```text
强弱侧 mask
质量权重
姿态置信度
深度对齐置信度
有效窗口标记
跨模态一致性约束
uncertainty output
```

`M_quality` 必须作为 attention bias：

```text
高质量位置：0 或小惩罚
低质量位置：负值
严重无效位置：-inf
```

不得让 CAST 无条件主导最终判定。

---

## 17. 训练规则

训练前必须确认：

```text
数据契约通过
XSI QC 通过
CAST QC 通过
深度对齐通过
RelBearing 正负号验证完成
强弱侧权重生成完成
有效窗口生成完成
弱标签审计完成
跨模态相关性验证完成
负对照完成
正对照完成
物理 baseline 完成
```

训练脚本必须保存：

```text
experiment_id
git_commit
branch
config files
config hash
data_version
label_version
feature_version
random seed
host
GPU
start_time
end_time
metrics
logs
figures
failure_cases
```

### 17.1 类别不平衡

必须考虑：

```text
Focal Loss
BCE + Dice
Ordinal Loss
Brier / calibration loss
oversampling channel intervals
uncertain mask exclusion
cost-sensitive thresholding
```

### 17.2 损失函数

推荐多任务损失：

```text
L_presence_focal
L_severity_ordinal
L_azimuth_BCE_Dice
L_confidence_Brier
L_consistency
```

不得重复计算 Dice。

---

## 18. 评估规则

禁止只看 Accuracy。

必须报告：

```text
Recall
Precision
F1
AUCPR
IoU
Azimuth IoU
Depth IoU
Macro-F1
Ordinal MAE
ECE
Brier Score
Reliability curve
False Positive per 100m
review_segments_per_100m
object_level_recall
object_level_precision
high_severity_recall
```

必须完成：

```text
RelBearing 旋转消融
强弱侧消融
有效窗口消融
深度错位检验
随机方位打乱
负对照
正对照
按井划分验证
跨井测试
```

如果随机方位打乱或深度错位后性能不下降，说明模型可能存在泄漏或伪相关，必须暂停主模型训练。

---

## 19. 可解释性规则

解释必须回到：

```text
depth
azimuth
XSI side
millisecond time window
STFT frequency band
STC slowness peak
CAST low-Zc azimuth
strong/weak side status
quality flags
uncertainty
```

解释可信度检查必须包括：

```text
random label test
model parameter randomization
input perturbation
side perturbation
STC consistency check
counterfactual masking
```

如果归因区域落在噪声段、缺失段、低质量 CAST 扇区或无物理意义时间窗，该解释不应被接受。

---

## 20. 可视化规则

每个关键阶段必须输出图。

必须支持或预留：

```text
data manifest summary
XSI waveform preview
CAST Zc image
depth alignment plot
local depth lag plot
RelBearing rotation before/after plot
strong/weak side polar plot
effective window overlay
label mask image
label confidence image
STC slowness-time plot
baseline feature importance
prediction vs label plot
failure case report
```

没有图，不相信结果。

---

## 21. 文档规则

Agent 修改项目时，应同步维护相关文档。

文档职责：

| 文档 | 作用 |
|---|---|
| `README.md` | 项目入口、开发流程、运行说明 |
| `AGENTS.md` | AI agent 与开发行为规则 |
| `docs/report.md` | 最高技术方案 |
| `docs/data_contract.md` | HDF5 / Zarr / memmap 数据契约 |
| `docs/development_workflow.md` | 本地、GitHub、服务器协作流程 |
| `docs/experiment_protocol.md` | 实验记录与 manifest 规范 |
| `docs/server_runbook.md` | 服务器运行手册 |
| `docs/labeling_protocol.md` | 弱标签生成规则 |
| `docs/model_design.md` | 模型设计 |
| `docs/evaluation_protocol.md` | 评估与验收 |
| `docs/decisions/ADR-*.md` | 重要技术决策记录 |

重要架构决策必须写 ADR。

---

## 22. 配置规则

配置文件位置：

```text
configs/
```

禁止在代码中硬编码：

```text
路径
阈值
采样率
滤波参数
STFT 参数
STC 网格
标签版本
训练超参数
GPU 编号
输出目录
```

必须通过 YAML、CLI 参数或环境变量传入。

配置文件修改会影响实验可复现性，必须记录到 manifest。

---

## 23. 脚本规则

脚本命名按阶段编号：

```text
00_check_env.py
01_build_manifest.py
02_run_qc.py
03_align_data.py
04_generate_labels.py
05_extract_features.py
06_train_baseline.py
07_train_xsi_only.py
08_evaluate.py
```

所有长期脚本应支持：

```bash
--config configs/xxx.yaml
--dry-run
--limit-depth 100
--output-dir <path>
--overwrite false
```

默认不得覆盖已有结果。

如果需要覆盖，必须显式传入：

```bash
--overwrite true
```

---

## 24. Agent 行为准则

AI agent 在本项目中必须：

1. 先读相关文档，再改代码。
2. 先理解 `docs/report.md`，再实现模型或标签逻辑。
3. 优先做小步修改。
4. 每次修改尽量可测试。
5. 不擅自移动大文件。
6. 不创建真实数据副本。
7. 不引入无法解释的复杂模型。
8. 不绕过 QC、alignment、baseline。
9. 不把 CAST 输入 XSI-only 模型。
10. 不在服务器上手改代码。
11. 不删除用户数据。
12. 不覆盖已有实验结果。
13. 不提交 secret、`.env`、数据、模型、日志。
14. 修改配置时同步更新文档或注释。
15. 添加新模块时同步添加测试。

---

## 25. MVP 路线

Agent 应按 MVP 顺序推进，不得跳跃到深度模型。

| MVP 阶段 | 必须完成 | 明确不做 |
|---|---|---|
| MVP-1 数据契约与 QC | HDF5 schema、XSI QC、CAST QC、版本字段、质量掩码 | 不训练深度模型 |
| MVP-2 对齐与方位归一化 | RelBearing 双符号验证、低井斜 mask、高边坐标图、深度 lag 估计 | 不生成最终训练集 |
| MVP-3 弱标签审计 | 自适应基线、方位梯度、纵向连续性、uncertain mask、对象级标签 | 不做复杂神经网络 |
| MVP-4 跨模态相关性验证 | 旋转消融、强弱侧消融、错位检验、随机方位打乱、负/正对照 | 不追求最高精度 |
| MVP-5 物理 baseline | STC、能量、衰减、强弱侧特征 + 传统模型 | 不上大模型 |
| MVP-6 XSI-only 深度模型 | STFT 可逆支路 + STC 物理支路 + 多任务头 | 不把 CAST 作为输入 |
| MVP-7 XSI+CAST 融合解释 | 非对称交叉注意力、跨模态一致性、不确定性输出 | 不用融合模型证明 XSI-only 能力 |
| MVP-8 专家复核与发布门槛 | 错误案例库、解释报告、对象级评估、工程成本指标 | 不自动替代人工决策 |

---

## 26. 当前阶段任务建议

当前项目处于工程骨架阶段。

下一步建议顺序：

```text
1. 确认 README.md、AGENTS.md、docs/report.md 已存在
2. 编写 docs/data_contract.md
3. 编写 configs/paths.local.example.yaml
4. 编写 configs/paths.server.example.yaml
5. 编写 configs/preprocess.yaml
6. 实现 scripts/01_build_manifest.py
7. 准备 tests/fixtures/tiny_sample/
8. 实现 HDF5 schema validation
9. 实现 XSI / CAST manifest 读取
10. 进入 MVP-1：数据契约与 QC
```

---

## 27. 最终红线

以下行为绝对禁止：

```text
直接训练黑盒模型
跳过 QC
跳过坐标归一化
跳过强弱侧建模
跳过跨模态相关性实验
跳过物理 baseline
把 CAST Zc 输入 XSI-only 模型
用 XSI+CAST 融合模型证明 XSI-only 能力
把大数据提交到 Git
把模型权重提交到 Git
把日志提交到 Git
在 main 上直接开发
在服务器上手工改代码
没有 manifest 就声称实验可复现
没有图就相信结果
```

---

## 28. 一句话原则

> 先让数据可信，再让标签可信；先让物理 baseline 可信，再让深度模型可信；先让解释可信，再让结果可用。