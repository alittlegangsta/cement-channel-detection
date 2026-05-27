# Cement Channel Detection  
## 固井窜槽智能检测与声波波形可解释分析项目

本项目旨在利用 LWD 单极声波全波列数据 XSI 与 CAST 超声井壁成像数据，构建一个科学、可解释、可审计、可复现的固井窜槽智能检测系统。

本项目不是简单训练一个黑盒分类模型，而是围绕以下完整链条展开：

```text
原始测井数据
→ 数据契约
→ 数据质量控制
→ 深度对齐
→ 高边坐标归一化
→ 强弱侧建模
→ CAST 有效窗口
→ 弱标签生成
→ 跨模态相关性验证
→ 物理 baseline
→ XSI-only 模型
→ XSI+CAST 融合解释模型
→ XAI 可解释报告
→ 专家复核
```

完整技术方案见：

```text
docs/report.md
```

README 作为项目入口、开发手册和运行指南；所有物理原则、公式、标签规则、模型边界和验收标准，以 `docs/report.md` 为最高依据。

---

## 1. 当前项目状态

当前项目已完成 MVP-1、MVP-2 / MVP-2C、MVP-3 / MVP-3H、MVP-4A 和
MVP-4B 系列 weak-label sanity review。MVP-4B Stage 2、MVP-4B-R、
MVP-4B-R2 和 MVP-4B-R3 的 gate 均为 `no_go`。后续 MVP-4B-R4/R4b/R4c
显示 depth-level high-confidence weak-label candidate target 在 controlled
refinement 下可稳定超过 permutation，refinement gate 为 `go`，但只允许生成
人工决策包，不自动进入新实验分支。

当前结论：

```text
side-depth weak-label classification with current shallow XSI features is no-go
mvp4c_consideration_allowed = false
controlled_time_frequency_sanity_allowed = false
no_final_labels = true
depth_level_refinement_gate = go
next_branch_requires_human_approval = true
```

已完成的 MVP-4B 复盘见：

```text
docs/mvp4b_no_go_review.md
docs/decisions/ADR-0004-mvp4b-no-go-and-next-directions.md
docs/depth_level_refinement_decision_pack.md
docs/depth_level_manual_review_checklist.md
```

下一步必须先人工确认。可讨论方向：

```text
controlled depth-level feature refinement v2
depth-level manual review pack inspection
interval-level target review
controlled time-frequency feasibility review
```

禁止在新的 gate 前进入：

```text
MVP-4C
STC
APES
deep learning
production model
final labels
```

当前基础命令应全部通过：

```bash
make test-smoke
make test
make lint
python scripts/00_check_env.py
```

当前本地项目路径：

```text
/home/xiaoj/cement-channel-detection
```

当前本地 Python 环境：

```text
micromamba env: cement_env
Python: 3.10
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

---

## 2. 项目目标

本项目目标包括：

1. 利用 XSI 单极声波全波列数据识别固井水泥环窜槽及其严重程度；
2. 使用 CAST 超声声阻抗 `Zc` 生成弱监督标签；
3. 解决 XSI 与 CAST 在深度、方位、姿态、井斜和偏心条件下的对齐问题；
4. 显式建模强侧、弱侧、有效窗口和不确定区域；
5. 构建物理特征体系，包括 STFT、STC、APES、套管波能量、流体波能量、衰减斜率等；
6. 训练物理 baseline，用于验证数据与标签是否具有可学习关系；
7. 训练 XSI-only 模型，验证声波数据本身是否具备窜槽预测能力；
8. 构建 XSI+CAST 融合解释模型，辅助专家进行多模态综合解释；
9. 输出能够回溯到深度、方位、毫秒时间窗、慢度峰、CAST 低阻抗条带和强弱侧状态的解释报告。

本项目的核心原则：

> 数据先于模型，验证先于训练，物理解释先于黑盒分数。

---

## 3. 核心技术原则

### 3.1 CAST 标签不是绝对真值

CAST 生成的标签不是绝对真值，而是 weak label / teacher label。

CAST 标签可能受到以下因素影响：

```text
仪器响应
成像质量
井斜
偏心
套管接箍
轻质水泥
污染水泥
微环隙
自由套管
套管椭圆度
解释规则偏差
```

因此，模型训练目标不是无条件复制 CAST，而是在 XSI 物理响应支持下学习可解释的窜槽证据。

---

### 3.2 所有匹配前必须做高边坐标归一化

XSI 的 8 个 Side 与 CAST 的 180 个方位扇区必须统一到相对于高边的坐标系。

默认候选公式为：

$$
theta_aligned = (theta_raw + RelBearing) mod 360
$$

但实际方向必须验证：

$$
theta_aligned_plus  = (theta_raw + RelBearing) mod 360
theta_aligned_minus = (theta_raw - RelBearing) mod 360
$$

最终采用哪个方向，必须依靠：

```text
XSI–CAST 循环互相关
RelBearing 正负号消融
强弱侧极坐标图
CAST 方位图
专家抽查
```

---

### 3.3 显式建模强弱侧

井斜或仪器偏心会导致声波和超声数据出现半边强、半边弱的非对称响应。

强侧通常代表更好的观测质量，但不等于真实窜槽方向；弱侧信噪比可能较低，但不能直接丢弃。

必须同时保留：

```text
强侧特征
弱侧特征
强弱侧差异特征
全方位鲁棒统计特征
有效窗口特征
姿态置信度
方位质量权重
深度对齐置信度
```

---

### 3.4 训练前必须完成跨模态科学验证

正式训练深度模型前，必须完成：

```text
RelBearing 旋转前后对比
RelBearing 正负号对比
强弱侧加权消融
CAST 有效窗口消融
随机方位打乱检验
深度错位检验
负对照实验
正对照实验
物理 baseline
标签置信度审计
```

如果这些验证未通过，不允许直接训练深度模型。

---

### 3.5 严格区分 XSI-only 与 XSI+CAST 模型

| 模型类型 | 输入 | CAST 的角色 | 用途 |
|---|---|---|---|
| XSI-only 预测模型 | XSI + 姿态 + 质量权重 + 有效窗口标记 | 仅用于生成弱标签、置信度、teacher signal，不作为直接输入 | 验证声波数据本身是否具备预测能力 |
| XSI+CAST 融合解释模型 | XSI + CAST + 姿态 + 质量权重 | 作为独立模态参与融合 | 辅助专家综合解释 |
| 标签质量审计模型 | XSI + CAST + QC + 弱标签 | 可使用 CAST | 发现标签错误、错位和低置信区域 |
| CAST teacher 蒸馏 | 训练期可使用 CAST teacher signal，推理期只用 XSI | teacher / weak supervision | 将 CAST 弱标签结构迁移给 XSI 模型 |

XSI+CAST 融合模型的高分不能用来证明 XSI 单独具备窜槽预测能力。

---

## 4. 开发架构

本项目采用：

```text
Windows
└─ VS Code Insiders
   └─ WSL2 Ubuntu 24.04
      ├─ 项目源码
      ├─ Codex IDE 插件
      ├─ Codex CLI
      ├─ micromamba cement_env
      ├─ Python / Git / 测试工具
      └─ 本地轻量开发与单元测试

GitHub
├─ main
├─ dev
└─ feature/*

Ubuntu 18.04.6 服务器
├─ /home/你的用户名/project
├─ /home/你的用户名/data
├─ /home/你的用户名/models
├─ /home/你的用户名/logs
└─ 只负责 git pull、conda/micromamba 环境运行、GPU 训练/推理、输出日志
```

### 本地 WSL2 负责

```text
写代码
写文档
写配置
写测试
跑小样例
跑 smoke / unit test
做轻量调试
用 Codex 辅助开发
提交 Git 分支
```

### GitHub 负责

```text
版本管理
分支协作
Pull Request
CI 检查
技术决策记录
代码审查
可复现实验配置归档
```

### Ubuntu 18.04.6 服务器负责

```text
全量数据预处理
STC / APES 离线特征提取
GPU 训练
推理
日志输出
模型权重保存
长时间实验运行
```

服务器不用于日常手工改代码。代码修改必须在本地完成，通过 Git 同步到服务器。

---

## 5. 仓库结构

目标仓库结构如下。部分模块将在 MVP 阶段逐步补齐。

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
│     └─ test_smoke.py
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

以下内容不得提交到 GitHub：

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
```

真实数据、HDF5 缓存、模型权重和训练日志必须放在服务器或本地非 Git 管理目录。

推荐服务器目录：

```text
/home/你的用户名/project
/home/你的用户名/data
/home/你的用户名/models
/home/你的用户名/logs
```

---

## 7. 本地开发环境

当前本地环境：

```text
OS: Windows + WSL2 Ubuntu
Editor: VS Code Insiders
Project path: /home/xiaoj/cement-channel-detection
Env manager: micromamba
Env name: cement_env
Python: 3.10
Python path: /home/dministrator/micromamba/envs/cement_env/bin/python
```

激活环境：

```bash
micromamba activate cement_env
```

确认解释器：

```bash
which python
python --version
```

预期解释器：

```text
/home/dministrator/micromamba/envs/cement_env/bin/python
```

运行环境检查：

```bash
cd /home/xiaoj/cement-channel-detection
python scripts/00_check_env.py
```

---

## 8. 快速开始

进入项目：

```bash
cd /home/xiaoj/cement-channel-detection
```

激活环境：

```bash
micromamba activate cement_env
```

运行基础检查：

```bash
make test-smoke
make test
make lint
python scripts/00_check_env.py
```

预期结果：

```text
tests passed
All checks passed!
Environment check passed.
```

---

## 9. 常用命令

```bash
make test-smoke      # 运行 smoke tests
make test            # 运行全部测试
make lint            # 运行 ruff 静态检查
make format          # 自动格式化代码
make fix             # 自动修复 ruff 可修复问题
make check-env       # 检查项目环境和目录结构
```

推荐 `Makefile` 至少包含：

```makefile
.PHONY: test test-unit test-integration test-smoke lint check-env format fix

test:
	python -m pytest tests

test-smoke:
	python -m pytest tests/smoke

test-unit:
	python -m pytest tests/unit

test-integration:
	python -m pytest tests/integration

lint:
	python -m ruff check src scripts tests

format:
	python -m ruff format src scripts tests

fix:
	python -m ruff check src scripts tests --fix
	python -m ruff format src scripts tests

check-env:
	python scripts/00_check_env.py
```

---

## 10. Git 分支策略

本项目采用：

```text
main
dev
feature/*
```

### 分支职责

| 分支 | 作用 |
|---|---|
| `main` | 稳定版，只放经过验证的阶段成果 |
| `dev` | 日常集成分支 |
| `feature/*` | 每个具体小任务一个分支 |

### 日常开发流程

从 `dev` 拉最新代码：

```bash
git checkout dev
git pull origin dev
```

创建 feature 分支：

```bash
git checkout -b feature/data-contract
```

开发、测试：

```bash
make test
make lint
python scripts/00_check_env.py
```

提交：

```bash
git status
git add .
git commit -m "docs: add data contract draft"
```

推送：

```bash
git push -u origin feature/data-contract
```

在 GitHub 上开 Pull Request：

```text
feature/data-contract → dev
```

阶段稳定后再开 Pull Request：

```text
dev → main
```

不要直接在 `main` 上开发。

---

## 11. 服务器运行流程

服务器只做干净、可复现的运行。

### 11.1 首次部署

```bash
cd /home/你的用户名/project
git clone <your-repo-url> cement-channel-detection
cd cement-channel-detection
```

创建或激活服务器环境：

```bash
conda env create -f environment.server.yml
conda activate cement-channel
```

或使用服务器实际约定的 micromamba/conda 环境。

---

### 11.2 服务器拉取最新代码

正式实验建议使用 `main`：

```bash
cd /home/你的用户名/project/cement-channel-detection
git checkout main
git pull origin main
```

开发阶段实验可以使用 `dev`：

```bash
git checkout dev
git pull origin dev
```

临时测试某个功能才使用 `feature/*`。

---

### 11.3 服务器运行训练

示例：

```bash
export CUDA_VISIBLE_DEVICES=1,2
export TOKENIZERS_PARALLELISM=false

python scripts/06_train_baseline.py --config configs/train_baseline.yaml
```

或：

```bash
bash scripts/server/train_baseline.sh
```

服务器不应手工修改代码。需要修改时，在本地修改、提交、推送，然后服务器 `git pull`。

---

## 12. 数据管线

数据管线必须按阶段执行，不允许跳步：

```text
raw .mat
→ manifest
→ QC
→ depth alignment
→ high-side azimuth normalization
→ strong/weak side modeling
→ CAST effective window
→ weak label generation
→ feature extraction
→ baseline validation
→ XSI-only training
→ fusion explanation
→ evaluation and XAI report
```

---

## 13. HDF5 数据契约摘要

完整数据契约应写入：

```text
docs/data_contract.md
```

核心维度顺序：

```text
depth → receiver → side → time
```

推荐核心数据集：

```text
/aligned/xsi_waveform
/aligned/cast_zc
/axis/depth
/axis/xsi_side_azimuth_deg
/axis/cast_azimuth_deg
/pose/inc_deg
/pose/rel_bearing_deg
/alignment/local_depth_lag
/alignment/depth_alignment_confidence
/alignment/azimuth_alignment_confidence
/quality/xsi_snr
/quality/side_quality_weight
/quality/cast_sector_quality_weight
/quality/orientation_confidence
/quality/uncertain_mask
/quality/quality_flags
/label/presence
/label/severity
/label/confidence
/label/channel_object_id
/features/stc
/features/stft_mag
/features/stft_phase
/features/physics_features
/metadata/data_version
/metadata/label_version
/metadata/feature_version
/metadata/git_commit
/metadata/preprocess_config_hash
```

`presence_label` 建议使用：

```text
0  = 无窜槽
1  = 疑似窜槽
-1 = uncertain
```

同时保留独立的：

```text
/quality/uncertain_mask
```

---

## 14. 弱标签生成原则摘要

弱标签来自 CAST，但不得只依赖固定阈值。

核心思想：

```text
局部自适应背景基线
+ 物理保底阈值
+ 方位动态梯度
+ 纵向连续性
+ 质量掩码
+ 标签置信度
```

方位梯度项不能单独触发高置信窜槽标签，必须同时伴随一定程度的相对低阻抗下降。

标签质量需要综合：

```text
CAST 成像质量
XSI 波形质量
姿态置信度
深度对齐置信度
纵向连续性
混淆工况标记
```

详细公式见：

```text
docs/report.md
```

---

## 15. 特征工程原则摘要

### 15.1 XSI QC

必须计算或预留：

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

必须计算或预留：

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

### 15.3 可逆时频特征

优先使用 STFT / ISTFT。

必须记录：

```text
窗口长度
重叠率
窗函数
边界处理方式
ISTFT 重构误差
feature_version
```

### 15.4 STC / APES 离线预计算

STC、APES 等高成本物理特征不得在 PyTorch `DataLoader` 中动态计算。

必须离线预计算并写入：

```text
HDF5
Zarr
NumPy memmap
```

训练阶段只负责：

```text
轻量读取
裁窗
归一化
batch 组装
```

---

## 16. 训练前必须完成的验证

在训练深度模型前，必须完成：

```text
RelBearing 旋转前后对比
RelBearing 正负号对比
强弱侧加权消融
CAST 有效窗口消融
随机方位打乱检验
深度错位检验
负对照实验
正对照实验
物理 baseline
标签置信度审计
```

如果这些验证未通过，不允许直接训练深度模型。

---

## 17. 模型路线摘要

### 阶段 1：物理 baseline

推荐模型：

```text
Logistic Regression
Random Forest
Gradient Boosting
XGBoost / LightGBM
```

输入：

```text
套管波时间窗能量
流体波时间窗能量
13 接收器衰减斜率
STC 套管波慢度峰
STC 流体波慢度峰
强弱侧能量比
有效窗口能量
姿态置信度
质量权重
```

---

### 阶段 2：XSI-only 模型

允许输入：

```text
XSI waveform
STFT features
STC / physics features
Inc
RelBearing
quality weights
effective window markers
```

禁止输入：

```text
CAST Zc
CAST label map
CAST-derived direct feature maps
```

---

### 阶段 3：XSI+CAST 融合解释模型

允许输入：

```text
XSI
CAST
pose
quality weights
effective windows
```

用途：

```text
专家辅助解释
多模态一致性验证
错误标签审计
```

不得用于证明 XSI-only 能力。

---

## 18. 评估指标

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

上线前必须通过：

```text
RelBearing 旋转优于未旋转
强弱侧 / 有效窗口优于简单平均
物理 baseline 显著优于随机
深度错位和方位打乱会显著降低性能
XAI 解释能回到合理时间窗和 STC 模态
高风险漏判案例经过专家复核
```

---

## 19. 可解释性要求

模型解释必须输出：

```text
模型预测结果
预测置信度
不确定性评分
XSI 关键时间窗
STC 关键慢度峰
CAST 对应低 Zc 方位
强弱侧与偏心解释
是否通过解释可信度检查
```

解释可信度检查包括：

```text
random label test
model parameter randomization
input perturbation
side perturbation
STC consistency check
counterfactual masking
```

若归因区域主要落在噪声段、缺失段、泥浆直达波无关区域或低质量 CAST 扇区，则该解释不应被接受。

---

## 20. 实验记录规范

每次实验必须保存 manifest：

```text
experiment_id
git_commit
branch
config files
data_version
label_version
feature_version
start_time
end_time
host
GPU
random_seed
metrics
figures
logs
notes
failure_cases
```

没有 manifest 的实验，视为不可复现。

---

## 21. MVP 开发路线

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

## 22. 开发习惯

### 22.1 每次只做一件小事

推荐任务粒度：

```text
只写 HDF5 schema 检查
只写 RelBearing 旋转函数
只写 XSI SNR 计算
只写 CAST QC 预览图
只写 label_v001 生成逻辑
```

### 22.2 先写配置，再写代码

不要把路径、阈值和采样率写死在代码里。

优先写到：

```text
configs/*.yaml
```

### 22.3 先跑 tiny sample，再跑全量数据

所有脚本都应支持：

```bash
--config configs/xxx.yaml
--dry-run
--limit-depth 100
```

### 22.4 每步都画图

必须保存：

```text
深度对齐图
RelBearing 旋转前后图
CAST Zc 方位展开图
XSI Side 能量图
强弱侧极坐标图
标签 mask 图
STC 图
预测 vs 标签图
错误案例图
```

没有图，不相信结果。

### 22.5 不要用复杂模型掩盖数据问题

如果以下内容没通过，不允许进入深度模型阶段：

```text
QC
alignment
weak label audit
cross-modal correlation
negative control
positive control
physics baseline
```

---

## 23. 项目禁忌

禁止：

1. 训练循环直接读取大 `.mat` 文件；
2. 把原始数据提交到 Git；
3. 把 HDF5、Zarr、NumPy 大缓存提交到 Git；
4. 把模型权重提交到 Git；
5. 把日志文件提交到 Git；
6. 把 `.env`、密钥、服务器路径密码提交到 Git；
7. 在 `main` 分支直接开发；
8. 在服务器上手工乱改代码；
9. XSI-only 模型输入 CAST Zc；
10. XSI-only 模型输入 CAST label map；
11. 用 XSI+CAST 融合模型证明 XSI-only 能力；
12. 只看 Accuracy；
13. 没有 QC、对齐、负对照、正对照和物理 baseline 就训练深度模型；
14. 没有 manifest 就认为实验可复现；
15. 没有图就相信结果；
16. 用复杂深度模型掩盖标签和对齐问题。

---

## 24. 下一步计划

当前建议按以下顺序推进：

```text
1. 将最终技术方案保存为 docs/report.md
2. 补齐 AGENTS.md
3. 编写 docs/data_contract.md
4. 编写 docs/development_workflow.md
5. 编写 docs/experiment_protocol.md
6. 编写 docs/server_runbook.md
7. 编写 configs/paths.local.example.yaml
8. 编写 configs/paths.server.example.yaml
9. 编写 configs/preprocess.yaml
10. 实现 scripts/01_build_manifest.py
11. 准备 tests/fixtures/tiny_sample/
12. 开始 MVP-1：数据契约与 QC
```

后续阶段前必须补齐：

```text
13. 在实现弱标签生成前，编写 `docs/labeling_protocol.md`
14. 在实现相关性验证、baseline 和正式评估前，编写 `docs/evaluation_protocol.md`
15. 在实现物理 baseline、XSI-only 模型和融合模型前，编写 `docs/model_design.md`
16. 在进入 MVP-2 前，补齐 `configs/alignment.yaml`
17. 在进入 MVP-3 前，补齐 `configs/label_v001.yaml`
18. 在进入 MVP-5 前，补齐 `configs/feature_stft.yaml`、`configs/feature_stc.yaml`、`configs/train_baseline.yaml`
19. 在进入 MVP-6 前，补齐 `configs/train_xsi_only.yaml`
20. 在进入 MVP-7 前，补齐 `configs/train_fusion.yaml`
21. 在正式评估前，补齐 `configs/eval.yaml`
```
### 24.1 当前阶段：工程骨架与 MVP-1 准备

当前建议按以下顺序推进：

1. 将最终技术方案保存为 `docs/report.md`
2. 补齐 `AGENTS.md`
3. 编写 `docs/data_contract.md`
4. 编写 `docs/development_workflow.md`
5. 编写 `docs/experiment_protocol.md`
6. 编写 `docs/server_runbook.md`
7. 编写 `configs/paths.local.example.yaml`
8. 编写 `configs/paths.server.example.yaml`
9. 编写 `configs/preprocess.yaml`
10. 实现 `scripts/01_build_manifest.py`
11. 准备 `tests/fixtures/tiny_sample/`
12. 开始 MVP-1：数据契约与 QC

### 24.2 后续阶段前必须补齐

在进入对应 MVP 阶段前，必须补齐以下文档和配置：

13. 在实现弱标签生成前，编写 `docs/labeling_protocol.md`
14. 在实现相关性验证、baseline 和正式评估前，编写 `docs/evaluation_protocol.md`
15. 在实现物理 baseline、XSI-only 模型和融合模型前，编写 `docs/model_design.md`
16. 在进入 MVP-2 前，补齐 `configs/alignment.yaml`
17. 在进入 MVP-3 前，补齐 `configs/label_v001.yaml`
18. 在进入 MVP-5 前，补齐 `configs/feature_stft.yaml`
19. 在进入 MVP-5 前，补齐 `configs/feature_stc.yaml`
20. 在进入 MVP-5 前，补齐 `configs/train_baseline.yaml`
21. 在进入 MVP-6 前，补齐 `configs/train_xsi_only.yaml`
22. 在进入 MVP-7 前，补齐 `configs/train_fusion.yaml`
23. 在正式评估前，补齐 `configs/eval.yaml`

### 24.3 文档维护原则

`docs/report.md` 是最高技术方案，其他专题文档应从中拆分细化，不得与其冲突。

- `docs/data_contract.md`：数据结构、字段、HDF5 Schema、版本规则；
- `docs/labeling_protocol.md`：弱标签、置信度、uncertain、对象级标签；
- `docs/evaluation_protocol.md`：指标、消融、负/正对照、go/no-go；
- `docs/model_design.md`：baseline、XSI-only、fusion、损失函数、XAI；
- `docs/development_workflow.md`：开发流程；
- `docs/experiment_protocol.md`：实验记录和 manifest；
- `docs/server_runbook.md`：服务器运行。

专题文档可以先写骨架版，随后在对应 MVP 阶段逐步补全。

---

## 25. 一句话原则

> 先让数据可信，再让标签可信；先让物理 baseline 可信，再让深度模型可信；先让解释可信，再让结果可用。
