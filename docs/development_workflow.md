# Development Workflow  
## 固井窜槽智能检测项目开发流程规范

本文件定义本项目的日常开发流程、分支策略、本地开发方式、Codex / AI agent 使用方式、服务器运行方式、测试要求、提交规范和阶段推进顺序。

本文件应与以下文件保持一致：

```text
README.md
AGENTS.md
docs/report.md
docs/data_contract.md
configs/*.yaml
```

其中：

- `README.md`：项目入口和快速开始；
- `AGENTS.md`：AI agent 与开发行为规则；
- `docs/report.md`：完整技术方案、物理原则、公式、标签规则、模型路线；
- `docs/data_contract.md`：数据结构、HDF5 Schema、字段、单位和版本约束；
- `docs/development_workflow.md`：本文件，负责日常开发流程。

如果本文档与 `docs/report.md` 发生冲突，以 `docs/report.md` 为准，并同步修订本文档。

---

## 1. 总体开发原则

本项目不是普通深度学习分类项目，而是一个强物理约束的数据工程、弱监督标签、声学特征、模型训练、解释性分析和专家复核闭环项目。

核心原则：

```text
数据先于模型
验证先于训练
物理解释先于黑盒分数
配置优先于硬编码
小样例先于全量数据
本地开发先于服务器训练
feature 分支先于 dev
dev 稳定后再进 main
没有 manifest 的实验不可复现
没有图的结果不可信
```

任何开发工作都不得绕过：

```text
QC
alignment
RelBearing 双符号验证
强弱侧建模
有效窗口
弱标签审计
跨模态相关性验证
负对照
正对照
物理 baseline
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

Python 环境：

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

- 不要擅自把 `/home/dministrator/...` 改成 `/home/Administrator/...`；
- 以当前机器真实可执行路径为准；
- 所有本地开发命令默认在 WSL2 Ubuntu 内运行；
- 不要在 Windows 文件系统路径下直接运行大型数据处理。

---

## 3. 推荐开发架构

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
└─ 只负责 git pull、环境运行、GPU 训练/推理、输出日志
```

---

## 4. 本地、GitHub、服务器职责边界

### 4.1 本地 WSL2 负责

```text
写代码
写文档
写配置
写测试
跑 smoke / unit test
跑 tiny sample
轻量级 debug
生成小图
使用 Codex / AI agent 辅助开发
提交 feature 分支
```

本地不负责：

```text
全量 GPU 训练
长时间 STC / APES 全井段计算
保存大型模型
保存真实大数据
```

---

### 4.2 GitHub 负责

```text
版本管理
分支协作
Pull Request
代码审查
CI 检查
Issue / Milestone
技术决策记录
阶段性稳定版本归档
```

GitHub 不应保存：

```text
原始 .mat 数据
HDF5 / Zarr / NumPy 大缓存
模型权重
训练日志
.env
密钥
服务器凭据
```

---

### 4.3 服务器负责

服务器只负责干净、可复现的大规模运行：

```text
git pull
激活服务器环境
运行全量预处理
运行离线 STC / APES 特征提取
运行 GPU 训练
运行推理
保存日志
保存模型
保存实验 manifest
```

服务器不负责：

```text
手工开发代码
直接编辑 main 分支
临时乱改脚本
保存无版本实验
提交大文件到 Git
```

---

## 5. 仓库结构开发约定

目标仓库结构：

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
│
├─ scripts/
│
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ smoke/
│  └─ fixtures/
│     └─ tiny_sample/
│
├─ notebooks/
├─ experiments/
├─ reports/
└─ .github/
```

新增模块时必须考虑：

```text
是否需要测试
是否需要配置文件
是否需要文档
是否影响 data_version / label_version / feature_version
是否影响 docs/data_contract.md
是否影响 AGENTS.md
```

---

## 6. Git 分支策略

本项目采用：

```text
main
dev
feature/*
```

### 6.1 分支职责

| 分支 | 作用 |
|---|---|
| `main` | 稳定版，只放经过验证的阶段成果 |
| `dev` | 日常集成分支 |
| `feature/*` | 每个具体小任务一个分支 |

### 6.2 分支规则

1. 不得直接在 `main` 上开发。
2. 日常工作从 `dev` 创建 `feature/*`。
3. 每个 `feature/*` 只做一个明确任务。
4. `feature/*` 完成后通过 Pull Request 合并到 `dev`。
5. `dev` 阶段稳定后，再通过 Pull Request 合并到 `main`。
6. 服务器正式实验优先使用 `main` 或经过确认的 `dev`。
7. 临时测试功能时才在服务器 checkout `feature/*`。

---

## 7. 日常开发流程

### 7.1 开始一个新任务

进入项目：

```bash
cd /home/xiaoj/cement-channel-detection
```

激活环境：

```bash
micromamba activate cement_env
```

切到 `dev` 并拉取最新代码：

```bash
git checkout dev
git pull origin dev
```

创建 feature 分支：

```bash
git checkout -b feature/<task-name>
```

示例：

```bash
git checkout -b feature/data-contract
git checkout -b feature/manifest-builder
git checkout -b feature/relbearing-alignment
git checkout -b feature/xsi-qc
git checkout -b feature/label-v001
```

---

### 7.2 开发中循环

每完成一个小功能，执行：

```bash
make test-smoke
make test
make lint
python scripts/00_check_env.py
```

如果 lint 失败：

```bash
make fix
make lint
```

或：

```bash
python -m ruff check src scripts tests --fix
python -m ruff format src scripts tests
```

---

### 7.3 提交代码

查看改动：

```bash
git status
git diff
```

添加文件：

```bash
git add .
```

提交：

```bash
git commit -m "type: short description"
```

示例：

```bash
git commit -m "docs: add data contract"
git commit -m "config: add local path template"
git commit -m "feat: implement angle wrapping utilities"
git commit -m "test: add relbearing rotation tests"
git commit -m "fix: correct circular distance calculation"
```

推送：

```bash
git push -u origin feature/<task-name>
```

然后在 GitHub 上开 Pull Request：

```text
feature/<task-name> → dev
```

---

## 8. Commit Message 规范

推荐格式：

```text
<type>: <message>
```

常用 type：

| type | 用途 |
|---|---|
| `chore` | 工程杂项、初始化、依赖配置 |
| `docs` | 文档 |
| `config` | 配置文件 |
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `test` | 测试 |
| `refactor` | 重构 |
| `style` | 格式化 |
| `ci` | GitHub Actions / CI |
| `perf` | 性能优化 |
| `exp` | 实验配置或实验记录 |

示例：

```text
chore: initialize project scaffold
docs: add development workflow
config: add preprocessing defaults
feat: implement manifest builder
test: add schema validation tests
fix: handle azimuth wraparound
refactor: split label confidence utilities
```

---

## 9. Pull Request 规范

每个 PR 应包含：

```text
改动目的
涉及文件
是否影响数据契约
是否影响标签规则
是否影响模型输入
是否影响 XSI-only 防泄漏
测试结果
风险说明
下一步
```

### 9.1 PR Checklist

提交 PR 前必须确认：

```text
[ ] 当前分支不是 main
[ ] 只解决一个明确任务
[ ] make test-smoke 通过
[ ] make test 通过
[ ] make lint 通过
[ ] python scripts/00_check_env.py 通过
[ ] 没有提交数据、模型、日志、.env
[ ] 没有硬编码本地绝对路径
[ ] 如果改了数据结构，已更新 docs/data_contract.md
[ ] 如果改了模型输入，已检查 XSI-only 防泄漏
[ ] 如果改了标签逻辑，已更新 label_version 或说明原因
[ ] 如果改了特征逻辑，已更新 feature_version 或说明原因
[ ] 如果做了重要技术决策，已新增 ADR
```

---

## 10. AI / Codex 使用流程

本项目允许使用 AI agent / Codex CLI / Codex IDE 插件辅助开发，但必须遵守 `AGENTS.md`。

### 10.1 适合交给 AI 的任务

```text
生成文档初稿
检查文档一致性
编写配置模板
实现小工具函数
补单元测试
检查 ruff / pytest 错误
重构小模块
生成 schema validator 草稿
生成脚本参数解析
生成 README / docs 交叉引用
```

### 10.2 不应直接交给 AI 全自动完成的任务

```text
直接训练模型
直接操作真实大数据
直接删除文件
直接修改服务器代码
直接改 main 分支
直接覆盖实验结果
直接生成全量标签并作为结论
直接设计最终模型结论
```

### 10.3 推荐 Codex 提示词

```text
请阅读 README.md、AGENTS.md、docs/report.md 和相关文档。

任务：
1. 只完成当前 feature 分支的单一任务；
2. 不要修改无关文件；
3. 不要提交数据、模型、日志或 .env；
4. 不要硬编码绝对路径；
5. 修改后运行 make test-smoke、make test、make lint、python scripts/00_check_env.py；
6. 如果失败，做最小修复并重新运行；
7. 最后总结修改内容、测试结果和潜在风险。
```

### 10.4 AI 生成内容后的人工检查

AI 生成内容后，必须人工检查：

```text
是否符合 docs/report.md
是否符合 AGENTS.md
是否引入标签泄漏
是否跳过 QC 或 alignment
是否硬编码路径
是否把 CAST 输入 XSI-only
是否覆盖已有结果
是否缺少测试
```

---

## 11. 本地测试规范

### 11.1 测试目录

```text
tests/
├─ unit/
├─ integration/
├─ smoke/
└─ fixtures/
   └─ tiny_sample/
```

### 11.2 测试类型

| 类型 | 命令 | 作用 |
|---|---|---|
| smoke | `make test-smoke` | 快速确认项目结构、import、基础命令 |
| unit | `make test-unit` | 测试单个函数 |
| integration | `make test-integration` | 测试小样例流程 |
| all | `make test` | 运行全部测试 |

### 11.3 优先测试对象

以下功能必须优先写测试：

```text
angle wrap to [0, 360)
circular distance
RelBearing + / - rotation
low inclination confidence
CAST 180 → XSI 8 聚合
depth lag estimation
weighted aggregation
label candidate generation
label confidence calculation
quality_flags bitmask
HDF5 schema validation
XSI-only input guard
```

---

## 12. 本地 tiny sample 流程

任何新管线都必须先支持 tiny sample。

tiny sample 位置：

```text
tests/fixtures/tiny_sample/
```

tiny sample 原则：

```text
体积极小
可公开或合成
不含真实敏感井数据
能覆盖 XSI / CAST / Inc / RelBearing / depth
能快速运行
能用于单元测试和集成测试
```

建议 tiny sample 维度：

```text
depth = 4 或 8
receiver = 13
side = 8
time = 32 或 64
cast_azimuth = 180
```

开发顺序：

```text
先 synthetic tiny sample
再小井段真实样例
再服务器全量数据
```

---

## 13. 配置文件开发流程

配置文件位于：

```text
configs/
```

配置优先原则：

```text
路径不写死
阈值不写死
采样率不写死
滤波参数不写死
STFT / STC 参数不写死
标签规则不写死
GPU 编号不写死
输出路径不写死
```

新增功能时，应优先考虑是否需要配置：

```text
paths.local.example.yaml
paths.server.example.yaml
preprocess.yaml
alignment.yaml
label_v001.yaml
feature_stft.yaml
feature_stc.yaml
train_baseline.yaml
train_xsi_only.yaml
train_fusion.yaml
eval.yaml
```

配置修改必须在实验 manifest 中记录 hash。

---

## 14. 数据处理开发顺序

### 14.1 MVP-1：数据契约与 QC

优先实现：

```text
docs/data_contract.md
configs/paths.local.example.yaml
configs/paths.server.example.yaml
configs/preprocess.yaml
scripts/01_build_manifest.py
schema validator
tiny sample
XSI QC placeholder
CAST QC placeholder
```

不做：

```text
深度模型
复杂 attention
全量训练
最终标签结论
```

---

### 14.2 MVP-2：对齐与方位归一化

实现：

```text
RelBearing + / - 双符号旋转
低井斜 mask
orientation_confidence
local_depth_lag
depth_alignment_confidence
azimuth_alignment_confidence
alignment report
rotation ablation figure
```

---

### 14.3 MVP-3：弱标签审计

实现：

```text
自适应背景基线
物理保底阈值
方位梯度约束
纵向连续性
HardQualityMask
label_confidence
uncertain_mask
对象级标签
label preview figure
```

---

### 14.4 MVP-4：跨模态相关性验证

实现：

```text
循环互相关
强弱侧消融
有效窗口消融
随机方位打乱
深度错位检验
负对照
正对照
go / no-go 报告
```

---

### 14.5 MVP-5：物理 baseline

实现：

```text
physics feature table
Logistic Regression
Random Forest
Gradient Boosting
feature importance
baseline report
```

---

### 14.6 MVP-6：XSI-only 深度模型

实现：

```text
XSI-only Dataset
XSI-only input guard
STFT branch
STC / physics branch
multi-task head
uncertainty output
calibration metrics
```

严禁：

```text
CAST Zc 作为输入
CAST label map 作为输入
融合模型结果证明 XSI-only 能力
```

---

## 15. 服务器开发与运行流程

### 15.1 服务器首次部署

```bash
cd /home/你的用户名/project
git clone <repo-url> cement-channel-detection
cd cement-channel-detection
```

创建环境：

```bash
conda env create -f environment.server.yml
conda activate cement-channel
```

或使用服务器实际 micromamba / conda 环境。

---

### 15.2 正式实验拉取代码

正式实验优先使用 `main`：

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

临时测试 feature：

```bash
git checkout feature/<task-name>
git pull origin feature/<task-name>
```

---

### 15.3 服务器运行命令

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

---

### 15.4 服务器禁止事项

服务器上禁止：

```text
手工修改代码后不提交
直接在 main 上改代码
直接运行无配置脚本
把结果写到根目录 /
覆盖已有实验输出
删除历史实验
提交大数据或模型权重到 Git
```

---

## 16. 实验开发流程

任何实验都必须有：

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
figures
logs
notes
failure_cases
```

实验输出建议结构：

```text
experiments/
└─ manifests/
   └─ <experiment_id>.json

reports/
└─ figures/
   └─ <experiment_id>/

logs/
└─ <experiment_id>.log

models/
└─ <experiment_id>/
```

注意：

- `experiments/manifests/*.json` 可提交；
- 大日志、大模型、大图可根据 `.gitignore` 控制是否提交；
- 模型权重不得提交到 Git；
- 大型 HDF5 / Zarr / NumPy 文件不得提交到 Git。

---

## 17. 文档维护流程

新增或修改功能时，应同步检查文档。

| 改动类型 | 应检查文档 |
|---|---|
| 数据字段变化 | `docs/data_contract.md` |
| 标签规则变化 | `docs/labeling_protocol.md`, `docs/report.md` |
| 模型输入变化 | `docs/model_design.md`, `AGENTS.md` |
| 评估指标变化 | `docs/evaluation_protocol.md` |
| 服务器运行变化 | `docs/server_runbook.md` |
| 开发流程变化 | `docs/development_workflow.md` |
| 重大技术决策 | `docs/decisions/ADR-*.md` |

---

## 18. ADR 规则

重大技术决策必须写 ADR。

ADR 位置：

```text
docs/decisions/
```

命名：

```text
ADR-0001-repo-structure.md
ADR-0002-data-format.md
ADR-0003-label-versioning.md
```

适合写 ADR 的情况：

```text
选择 HDF5 / Zarr / memmap
确定 RelBearing 符号验证策略
确定 XSI-only 防泄漏边界
确定 label_version 策略
确定 STFT / STC 特征方案
确定训练 split 策略
确定服务器路径规范
```

ADR 应包含：

```text
背景
决策
原因
替代方案
影响
日期
```

---

## 19. 常见开发错误

禁止出现：

```text
在 main 上直接开发
忘记从 dev 拉最新代码
一个 feature 分支做多个无关任务
把数据提交到 Git
把模型权重提交到 Git
把 .env 提交到 Git
硬编码本地路径
脚本不能 dry-run
脚本默认覆盖输出
没有 tiny sample 就跑全量
没有 QC 就生成标签
没有 alignment 就训练模型
把 CAST Zc 输入 XSI-only
只看 Accuracy
没有 manifest 就声称实验完成
```

---

## 20. 推荐每日工作节奏

### 开始工作

```bash
cd /home/xiaoj/cement-channel-detection
micromamba activate cement_env
git checkout dev
git pull origin dev
git checkout -b feature/<today-task>
```

### 开发中

```bash
make test-smoke
make test
make lint
python scripts/00_check_env.py
```

### 结束前

```bash
git status
git diff
make test
make lint
git add .
git commit -m "type: message"
git push -u origin feature/<today-task>
```

### 开 PR 前

```text
确认没有大文件
确认没有 .env
确认没有硬编码路径
确认测试通过
确认文档同步
确认只做了一个任务
```

---

## 21. 当前阶段建议任务顺序

当前项目处于工程骨架与 MVP-1 准备阶段。

建议顺序：

```text
1. 确认 README.md、AGENTS.md、docs/report.md 已存在
2. 完成 docs/data_contract.md
3. 完成 docs/development_workflow.md
4. 完成 docs/experiment_protocol.md
5. 完成 docs/server_runbook.md
6. 完成 configs/paths.local.example.yaml
7. 完成 configs/paths.server.example.yaml
8. 完成 configs/preprocess.yaml
9. 实现 scripts/01_build_manifest.py
10. 准备 tests/fixtures/tiny_sample/
11. 实现 schema validator
12. 进入 MVP-1：数据契约与 QC
```

---

## 22. 最终原则

本项目的开发流程必须服务于科学性、可解释性和可复现性。

任何代码如果：

```text
不可测试
不可配置
不可追溯
不可解释
不可复现
可能造成标签泄漏
可能绕过 QC / alignment / baseline
```

都不应进入 `dev`，更不应进入 `main`。

一句话原则：

> 小步开发，配置驱动，先测再训，先证伪再建模，先可解释再可用。