# Server Runbook  
## 固井窜槽智能检测项目服务器运行手册

本文件定义本项目在 Ubuntu 服务器上的目录规范、环境管理、代码同步、数据管理、预处理、训练、推理、日志、实验归档、故障排查和安全红线。

本文件必须与以下文件保持一致：

```text
README.md
AGENTS.md
docs/report.md
docs/data_contract.md
docs/development_workflow.md
docs/experiment_protocol.md
configs/*.yaml
```

其中：

- `README.md`：项目入口和快速开始；
- `AGENTS.md`：AI agent 与开发行为规则；
- `docs/report.md`：完整技术方案、物理原则、标签规则、模型路线；
- `docs/data_contract.md`：数据结构、HDF5 Schema、字段、单位和版本约束；
- `docs/development_workflow.md`：本地、GitHub、服务器协作流程；
- `docs/experiment_protocol.md`：实验记录、manifest、go/no-go 规则；
- `docs/server_runbook.md`：本文件，负责服务器运行规范。

如果本文档与 `docs/report.md` 冲突，以 `docs/report.md` 为准，并同步修订本文档。

---

## 1. 服务器职责

服务器只负责干净、可复现的大规模运行。

服务器负责：

```text
git pull
激活 conda / micromamba 环境
运行全量 manifest
运行全量 QC
运行深度与方位对齐
运行弱标签生成
运行离线 STC / APES / 物理特征提取
运行物理 baseline
运行 GPU 训练
运行推理
输出日志
保存模型
保存实验 manifest
保存评估报告
```

服务器不负责：

```text
日常手写代码
手工乱改脚本
直接在 main 上开发
临时修改后不提交
保存无版本实验
把大文件提交到 Git
替代本地单元测试
跳过 QC / alignment / baseline
```

服务器使用原则：

> 本地开发，GitHub 管理版本，服务器只拉取代码并运行可复现实验。

---

## 2. 服务器固定目录

本项目服务器路径固定为：

```text
/home/xiaoj/cement-channel-detection
```

数据目录固定为：

```text
/home/xiaoj/cement-channel-data
```

推荐服务器目录结构：

```text
/home/xiaoj/
├─ cement-channel-detection/     # Git 仓库，只放代码、文档、配置、测试
└─ cement-channel-data/          # 数据、特征、日志、模型、实验输出，不进 Git
   ├─ raw/
   ├─ interim/
   ├─ processed/
   ├─ features/
   ├─ manifests/
   ├─ reports/
   ├─ logs/
   ├─ models/
   ├─ predictions/
   └─ tmp/
```

创建数据目录：

```bash
mkdir -p /home/xiaoj/cement-channel-data/{raw,interim,processed,features,manifests,reports,logs,models,predictions,tmp}
```

---

## 3. Git 仓库目录

项目代码目录：

```bash
cd /home/xiaoj/cement-channel-detection
```

该目录只允许保存：

```text
源码
配置模板
文档
测试
脚本
小样例
实验 manifest
少量报告索引
```

不得保存：

```text
原始 .mat
大型 .h5 / .hdf5
大型 .zarr
大型 .npy / .npz
模型权重
训练日志
.env
服务器密钥
```

---

## 4. 数据目录规范

服务器数据目录：

```text
/home/xiaoj/cement-channel-data
```

### 4.1 raw

原始数据，只读，不修改。

```text
/home/xiaoj/cement-channel-data/raw/
```

推荐按井组织：

```text
raw/
├─ well_001/
│  ├─ XSI.mat
│  ├─ CAST.mat
│  └─ pose.mat
├─ well_002/
│  ├─ XSI.mat
│  ├─ CAST.mat
│  └─ pose.mat
└─ well_003/
   ├─ XSI.mat
   ├─ CAST.mat
   └─ pose.mat
```

规则：

```text
不得覆盖 raw
不得修改 raw
不得在 raw 中写中间文件
不得把 raw 提交到 Git
```

---

### 4.2 interim

中间转换结果：

```text
/home/xiaoj/cement-channel-data/interim/
```

用途：

```text
解包后的临时数组
小段抽样数据
初步转换结果
debug 中间文件
```

---

### 4.3 processed

对齐后的正式 HDF5 / Zarr / memmap：

```text
/home/xiaoj/cement-channel-data/processed/
```

示例：

```text
processed/
├─ aligned_data_v001.h5
├─ labels_v001.h5
└─ train_xsi_only_v001.h5
```

---

### 4.4 features

离线特征：

```text
/home/xiaoj/cement-channel-data/features/
```

示例：

```text
features/
├─ features_physics_v001.h5
├─ features_stft_v001.h5
├─ features_stc_v001.h5
└─ features_apes_v001.h5
```

---

### 4.5 manifests

实验 manifest 与数据清单：

```text
/home/xiaoj/cement-channel-data/manifests/
```

示例：

```text
manifests/
├─ raw_file_inventory.csv
├─ data_manifest_v001.json
├─ exp01_manifest_20260518_r001.json
└─ exp07_physics_baseline_20260518_r001.json
```

---

### 4.6 reports

报告与图片：

```text
/home/xiaoj/cement-channel-data/reports/
```

推荐结构：

```text
reports/
└─ experiments/
   └─ <experiment_id>/
      ├─ report.md
      ├─ metrics.json
      ├─ go_no_go.json
      ├─ figures/
      └─ failure_cases.csv
```

---

### 4.7 logs

运行日志：

```text
/home/xiaoj/cement-channel-data/logs/
```

示例：

```text
logs/
├─ exp01_manifest_20260518_r001.log
├─ exp05_label_v001_20260518_r001.log
└─ exp08_xsi_only_20260518_r001.log
```

---

### 4.8 models

模型权重：

```text
/home/xiaoj/cement-channel-data/models/
```

示例：

```text
models/
└─ exp08_xsi_only_20260518_r001/
   ├─ model.pt
   ├─ config.yaml
   ├─ metrics.json
   └─ model_card.md
```

模型权重不得提交到 Git。

---

### 4.9 predictions

推理结果：

```text
/home/xiaoj/cement-channel-data/predictions/
```

---

### 4.10 tmp

临时文件：

```text
/home/xiaoj/cement-channel-data/tmp/
```

`tmp/` 可定期清理，但清理前必须确认没有正在运行的任务使用。

---

## 5. 服务器环境

服务器可能为：

```text
Ubuntu 18.04.6
A100 40GB × 3
```

默认 GPU 策略：

```text
GPU0 可能被占用
默认使用 GPU1 / GPU2
```

推荐训练前设置：

```bash
export CUDA_VISIBLE_DEVICES=1,2
export TOKENIZERS_PARALLELISM=false
```

如果只使用一张 GPU：

```bash
export CUDA_VISIBLE_DEVICES=1
```

---

## 6. 代码同步流程

服务器不直接写代码，只拉取 GitHub 上已经提交的代码。

### 6.1 正式实验使用 main

```bash
cd /home/xiaoj/cement-channel-detection
git checkout main
git pull origin main
```

适用：

```text
阶段性稳定实验
正式训练
正式评估
可归档结果
```

---

### 6.2 开发实验使用 dev

```bash
cd /home/xiaoj/cement-channel-detection
git checkout dev
git pull origin dev
```

适用：

```text
开发阶段服务器验证
全量预处理试跑
非最终实验
```

---

### 6.3 临时测试使用 feature

```bash
cd /home/xiaoj/cement-channel-detection
git checkout feature/<task-name>
git pull origin feature/<task-name>
```

适用：

```text
临时验证某个 feature
短期测试
不作为最终实验结果
```

feature 分支服务器实验必须在 manifest 中记录 branch 名称。

---

## 7. 服务器运行前检查

每次服务器实验前执行：

```bash
cd /home/xiaoj/cement-channel-detection

git status
git branch --show-current
git log -1 --oneline

python scripts/00_check_env.py
```

如果服务器环境支持测试，也运行：

```bash
make test-smoke
make lint
```

正式实验前建议至少运行：

```bash
make test-smoke
python scripts/00_check_env.py
```

如果 `git status` 显示未提交修改：

```text
不要直接运行正式实验
先确认这些修改是否应该提交
```

---

## 8. 配置文件

服务器真实路径配置建议放在：

```text
configs/paths.server.yaml
```

该文件通常不提交 Git。

提交到 Git 的模板为：

```text
configs/paths.server.example.yaml
```

服务器路径配置应指向：

```yaml
project:
  root: /home/xiaoj/cement-channel-detection

data:
  root: /home/xiaoj/cement-channel-data
  raw: /home/xiaoj/cement-channel-data/raw
  interim: /home/xiaoj/cement-channel-data/interim
  processed: /home/xiaoj/cement-channel-data/processed
  features: /home/xiaoj/cement-channel-data/features
  manifests: /home/xiaoj/cement-channel-data/manifests
  reports: /home/xiaoj/cement-channel-data/reports
  logs: /home/xiaoj/cement-channel-data/logs
  models: /home/xiaoj/cement-channel-data/models
  predictions: /home/xiaoj/cement-channel-data/predictions
  tmp: /home/xiaoj/cement-channel-data/tmp
```

如果不存在 `configs/paths.server.yaml`：

```bash
cp configs/paths.server.example.yaml configs/paths.server.yaml
```

然后根据服务器实际路径修改。

---

## 9. 环境激活

服务器可能使用 conda 或 micromamba。具体以服务器实际配置为准。

### 9.1 conda 示例

```bash
conda activate cement-channel
```

### 9.2 micromamba 示例

```bash
micromamba activate cement_env
```

确认 Python：

```bash
which python
python --version
```

确认 PyTorch / CUDA：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

---

## 10. GPU 检查

查看 GPU：

```bash
nvidia-smi
```

持续监控：

```bash
watch -n 2 nvidia-smi
```

查看当前用户进程：

```bash
ps -u xiaoj -f
```

查看 Python 进程：

```bash
ps -ef | grep python
```

---

## 11. 运行阶段顺序

服务器运行必须遵守阶段顺序：

```text
EXP-1 raw manifest
EXP-2 QC
EXP-3 alignment
EXP-4 strong / weak side and effective window
EXP-5 label generation
EXP-6 correlation validation
EXP-7 physics baseline
EXP-8 XSI-only model
EXP-9 XSI+CAST fusion explanation
EXP-10 expert review / release evaluation
```

不允许跳过：

```text
QC
alignment
RelBearing 双符号验证
强弱侧建模
有效窗口
弱标签审计
相关性验证
负/正对照
物理 baseline
```

---

## 12. 运行 Manifest

### 12.1 构建原始数据清单

示例：

```bash
cd /home/xiaoj/cement-channel-detection

python scripts/01_build_manifest.py \
  --config configs/paths.server.yaml \
  --output /home/xiaoj/cement-channel-data/manifests/data_manifest_v001.json
```

预期输出：

```text
/home/xiaoj/cement-channel-data/manifests/raw_file_inventory.csv
/home/xiaoj/cement-channel-data/manifests/data_manifest_v001.json
```

---

## 13. 运行 QC

示例：

```bash
python scripts/02_run_qc.py \
  --config configs/preprocess.yaml \
  --paths configs/paths.server.yaml \
  --manifest /home/xiaoj/cement-channel-data/manifests/data_manifest_v001.json \
  --output-dir /home/xiaoj/cement-channel-data/reports/experiments/exp02_qc_v001
```

预期输出：

```text
xsi_qc_report.md
cast_qc_report.md
quality_masks.h5
qc_summary.png
low_confidence_intervals.csv
```

---

## 14. 运行对齐

示例：

```bash
python scripts/03_align_data.py \
  --paths configs/paths.server.yaml \
  --config configs/alignment.yaml \
  --manifest /home/xiaoj/cement-channel-data/manifests/data_manifest_v001.json \
  --output /home/xiaoj/cement-channel-data/processed/aligned_data_v001.h5
```

预期输出：

```text
aligned_data_v001.h5
alignment_report.md
rotation_ablation.png
local_depth_lag.png
orientation_confidence.png
```

必须检查：

```text
RelBearing + 方案
RelBearing - 方案
未旋转方案
随机旋转方案
低井斜 uncertain mask
local depth lag
```

---

## 15. 运行标签生成

示例：

```bash
python scripts/04_generate_labels.py \
  --paths configs/paths.server.yaml \
  --config configs/label_v001.yaml \
  --input /home/xiaoj/cement-channel-data/processed/aligned_data_v001.h5 \
  --output /home/xiaoj/cement-channel-data/processed/labels_v001.h5
```

预期输出：

```text
labels_v001.h5
label_report.md
label_preview.png
label_confidence.png
channel_objects.csv
```

必须检查：

```text
presence 编码
severity 编码
uncertain_mask
label_confidence
HardQualityMask
对象级标签
```

---

## 16. 运行离线特征提取

示例：

```bash
python scripts/05_extract_features.py \
  --paths configs/paths.server.yaml \
  --config configs/feature_stft.yaml \
  --input /home/xiaoj/cement-channel-data/processed/aligned_data_v001.h5 \
  --labels /home/xiaoj/cement-channel-data/processed/labels_v001.h5 \
  --output /home/xiaoj/cement-channel-data/features/features_stft_v001.h5
```

STC 示例：

```bash
python scripts/05_extract_features.py \
  --paths configs/paths.server.yaml \
  --config configs/feature_stc.yaml \
  --input /home/xiaoj/cement-channel-data/processed/aligned_data_v001.h5 \
  --output /home/xiaoj/cement-channel-data/features/features_stc_v001.h5
```

注意：

```text
STC / APES 不得在 PyTorch DataLoader 中动态计算
必须离线预计算
必须保存 feature_version
必须保存配置 hash
```

---

## 17. 运行相关性验证

示例：

```bash
python scripts/08_evaluate.py \
  --paths configs/paths.server.yaml \
  --config configs/eval.yaml \
  --mode correlation_validation \
  --aligned /home/xiaoj/cement-channel-data/processed/aligned_data_v001.h5 \
  --labels /home/xiaoj/cement-channel-data/processed/labels_v001.h5 \
  --features /home/xiaoj/cement-channel-data/features/features_stc_v001.h5 \
  --output-dir /home/xiaoj/cement-channel-data/reports/experiments/exp06_correlation_v001
```

必须包含：

```text
RelBearing 旋转消融
强弱侧消融
有效窗口消融
随机方位打乱
深度错位检验
负对照
正对照
go/no-go
```

如果该阶段 no-go：

```text
不得进入 XSI-only 深度模型训练
```

---

## 18. 运行物理 Baseline

示例：

```bash
python scripts/06_train_baseline.py \
  --paths configs/paths.server.yaml \
  --config configs/train_baseline.yaml \
  --aligned /home/xiaoj/cement-channel-data/processed/aligned_data_v001.h5 \
  --labels /home/xiaoj/cement-channel-data/processed/labels_v001.h5 \
  --features /home/xiaoj/cement-channel-data/features/features_stc_v001.h5 \
  --output-dir /home/xiaoj/cement-channel-data/reports/experiments/exp07_baseline_v001
```

必须报告：

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

如果 baseline 不优于随机或负对照：

```text
不得进入深度模型阶段
```

---

## 19. 运行 XSI-only 深度模型

运行前必须确认：

```text
物理 baseline 已通过
负对照已通过
正对照已通过
随机方位打乱会降分
深度错位会降分
XSI-only input guard 已启用
```

设置 GPU：

```bash
export CUDA_VISIBLE_DEVICES=1,2
export TOKENIZERS_PARALLELISM=false
```

示例：

```bash
python scripts/07_train_xsi_only.py \
  --paths configs/paths.server.yaml \
  --config configs/train_xsi_only.yaml \
  --aligned /home/xiaoj/cement-channel-data/processed/aligned_data_v001.h5 \
  --labels /home/xiaoj/cement-channel-data/processed/labels_v001.h5 \
  --features /home/xiaoj/cement-channel-data/features/features_stc_v001.h5 \
  --output-dir /home/xiaoj/cement-channel-data/models/exp08_xsi_only_v001
```

XSI-only 禁止输入：

```text
/aligned/cast_zc
/aligned/cast_zc_to_xsi8
CAST label map 作为输入
CAST-derived direct feature maps
```

允许使用 CAST 生成的：

```text
weak label
label confidence
effective window
teacher signal
training supervision
```

但推理阶段不得依赖 CAST。

---

## 20. 运行 XSI+CAST 融合解释模型

该模型只用于：

```text
专家辅助解释
多模态一致性验证
标签质量审计
错误案例分析
```

不得用于证明 XSI-only 能力。

示例：

```bash
python scripts/07_train_xsi_only.py \
  --paths configs/paths.server.yaml \
  --config configs/train_fusion.yaml \
  --mode xsi_cast_fusion \
  --aligned /home/xiaoj/cement-channel-data/processed/aligned_data_v001.h5 \
  --labels /home/xiaoj/cement-channel-data/processed/labels_v001.h5 \
  --features /home/xiaoj/cement-channel-data/features/features_stc_v001.h5 \
  --output-dir /home/xiaoj/cement-channel-data/models/exp09_fusion_v001
```

如果后续有专门脚本，可改为：

```bash
python scripts/09_train_fusion.py ...
```

---

## 21. 后台运行

长时间任务建议使用 `tmux` 或 `screen`。

### 21.1 tmux

新建会话：

```bash
tmux new -s cement_train
```

运行任务。

退出但不中断：

```text
Ctrl-b d
```

恢复：

```bash
tmux attach -t cement_train
```

查看会话：

```bash
tmux ls
```

---

### 21.2 nohup

示例：

```bash
nohup python scripts/06_train_baseline.py \
  --paths configs/paths.server.yaml \
  --config configs/train_baseline.yaml \
  > /home/xiaoj/cement-channel-data/logs/exp07_baseline_v001.log 2>&1 &
```

查看日志：

```bash
tail -f /home/xiaoj/cement-channel-data/logs/exp07_baseline_v001.log
```

---

## 22. 日志规范

每个服务器实验必须保存日志：

```text
/home/xiaoj/cement-channel-data/logs/<experiment_id>.log
```

日志必须包含：

```text
experiment_id
git_commit
branch
config paths
config hash
data_version
label_version
feature_version
schema_version
host
GPU
start_time
end_time
warnings
errors
output paths
go/no-go
```

---

## 23. 实验 Manifest

每个实验必须保存 manifest：

```text
/home/xiaoj/cement-channel-data/manifests/<experiment_id>.json
```

manifest 必须包含：

```text
experiment_id
stage
task
status
git branch
git commit
is_dirty
config files
config hash
data_version
label_version
feature_version
schema_version
random seed
host
GPU
start_time
end_time
duration
input files
output files
metrics
figures
logs
failure cases
go/no-go
```

没有 manifest 的实验，视为不可复现。

---

## 24. 结果归档

实验结果推荐结构：

```text
/home/xiaoj/cement-channel-data/reports/experiments/<experiment_id>/
├─ report.md
├─ metrics.json
├─ go_no_go.json
├─ figures/
└─ failure_cases.csv
```

模型结果：

```text
/home/xiaoj/cement-channel-data/models/<experiment_id>/
├─ model.pt
├─ config.yaml
├─ metrics.json
├─ calibration.json
└─ model_card.md
```

不要把模型权重提交 Git。

---

## 25. 运行后检查

每次服务器实验完成后检查：

```bash
tail -n 100 /home/xiaoj/cement-channel-data/logs/<experiment_id>.log
```

检查输出：

```bash
ls -lh /home/xiaoj/cement-channel-data/reports/experiments/<experiment_id>/
ls -lh /home/xiaoj/cement-channel-data/manifests/
```

如果是训练任务，检查模型目录：

```bash
ls -lh /home/xiaoj/cement-channel-data/models/<experiment_id>/
```

检查 Git 是否意外改动：

```bash
cd /home/xiaoj/cement-channel-detection
git status
```

如果出现大文件被 Git 跟踪：

```bash
git rm --cached <file>
```

不要删除真实文件，除非确认它只是临时无用文件。

---

## 26. 常见故障排查

### 26.1 CUDA 不可用

检查：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

可能原因：

```text
环境没有安装 CUDA 版 PyTorch
CUDA_VISIBLE_DEVICES 设置错误
GPU 被占满
驱动异常
```

---

### 26.2 GPU 显存不足

解决：

```text
减小 batch size
缩短 depth window
减少 num_workers
减少 STFT / STC 输入尺寸
使用单 GPU 调试
使用 mixed precision
先跑 limit-depth
```

---

### 26.3 DataLoader 很慢

检查：

```text
是否在 DataLoader 中读取 .mat
是否在 DataLoader 中计算 STC / APES
HDF5 chunk 是否不合理
num_workers 是否过大或过小
磁盘 I/O 是否瓶颈
```

原则：

```text
STC / APES 必须离线预计算
训练期只轻量读取
```

---

### 26.4 HDF5 读取报错

检查：

```text
文件是否损坏
schema 是否匹配
路径是否正确
权限是否正确
是否多个进程同时写
HDF5 是否写入未关闭
```

避免多个进程同时写同一个 HDF5。

---

### 26.5 路径错误

检查：

```bash
pwd
ls -lh /home/xiaoj/cement-channel-detection
ls -lh /home/xiaoj/cement-channel-data
```

检查配置：

```bash
cat configs/paths.server.yaml
```

---

### 26.6 Git 分支错误

检查：

```bash
git branch --show-current
git status
git log -1 --oneline
```

正式实验应优先在：

```text
main
```

开发实验可在：

```text
dev
```

---

### 26.7 实验结果不可复现

检查 manifest 是否包含：

```text
git_commit
config hash
data_version
label_version
feature_version
random seed
host
GPU
input files
output files
```

如果缺失，实验不可作为正式结论。

---

## 27. 服务器安全红线

绝对禁止：

```text
在服务器上手工乱改代码
在 main 上直接开发
运行无配置脚本
覆盖旧实验结果
删除原始 raw 数据
修改 raw 数据
把大文件提交到 Git
把 .env 提交到 Git
把模型权重提交到 Git
把日志提交到 Git
训练循环读取大 .mat
DataLoader 动态计算 STC / APES
XSI-only 输入 CAST Zc
跳过 QC / alignment / baseline
用融合模型证明 XSI-only 能力
```

---

## 28. 清理策略

可以清理：

```text
tmp/
中断任务产生的临时文件
确认无用的 debug 小文件
重复生成且可复现的中间缓存
```

谨慎清理：

```text
interim/
processed/
features/
reports/
logs/
models/
manifests/
```

禁止清理：

```text
raw/
正式实验 manifest
正式模型结果
正式报告
```

清理前建议先查看大小：

```bash
du -h --max-depth=1 /home/xiaoj/cement-channel-data
```

---

## 29. 推荐服务器日常流程

### 开始实验

```bash
cd /home/xiaoj/cement-channel-detection
git checkout dev
git pull origin dev
micromamba activate cement_env  # 或 conda activate cement-channel
python scripts/00_check_env.py
```

### 设置 GPU

```bash
export CUDA_VISIBLE_DEVICES=1,2
export TOKENIZERS_PARALLELISM=false
```

### 运行任务

```bash
python scripts/<stage_script>.py \
  --paths configs/paths.server.yaml \
  --config configs/<stage_config>.yaml
```

### 查看日志

```bash
tail -f /home/xiaoj/cement-channel-data/logs/<experiment_id>.log
```

### 结束检查

```bash
git status
ls -lh /home/xiaoj/cement-channel-data/manifests/
ls -lh /home/xiaoj/cement-channel-data/reports/
```

---

## 30. 当前阶段服务器任务建议

当前项目仍处于工程骨架与 MVP-1 准备阶段。

服务器暂时不应进行深度模型训练。

当前服务器可准备：

```text
1. 创建 /home/xiaoj/cement-channel-data 目录结构
2. 配置 configs/paths.server.yaml
3. 确认 Git pull 流程
4. 确认 Python / CUDA / PyTorch 环境
5. 跑 scripts/00_check_env.py
6. 等本地完成 scripts/01_build_manifest.py 后，服务器跑 raw manifest
7. 等 tiny sample 流程通过后，再跑真实小井段
```

暂时不要：

```text
全量训练
大规模 STC
大规模 APES
复杂 attention 模型
正式 XSI-only 深度模型
```

---

## 31. 最终原则

服务器不是试错沙盒，而是可复现实验执行环境。

每一次服务器运行都必须能回答：

```text
用的哪份代码？
哪个 Git commit？
哪个分支？
哪份数据？
哪个 data_version？
哪个 label_version？
哪个 feature_version？
哪些配置？
哪些 GPU？
输出在哪里？
日志在哪里？
manifest 在哪里？
是否允许进入下一阶段？
```

一句话原则：

> 本地开发，GitHub 归档，服务器运行；服务器只执行可复现实验，不制造不可追溯结果。