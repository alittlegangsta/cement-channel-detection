# Data Contract  
## 固井窜槽智能检测项目数据契约

本文件定义本项目从原始测井数据到 HDF5 / Zarr / NumPy memmap 中间产物、弱标签、质量控制、物理特征、训练输入和评估输出之间的统一数据契约。

本数据契约是项目工程实现的基础，必须与以下文件保持一致：

```text
README.md
AGENTS.md
docs/report.md
configs/*.yaml
```

其中：

- `README.md`：项目入口与开发流程；
- `AGENTS.md`：AI agent 与开发行为规则；
- `docs/report.md`：完整技术方案、物理原则、公式、标签规则、模型边界；
- `docs/data_contract.md`：本文件，负责数据结构、字段、单位、维度、版本和校验规则。

如果本文件与 `docs/report.md` 冲突，应优先遵守 `docs/report.md`，并同步修订本文件。

---

## 1. 设计目标

本数据契约的目标是：

1. 固定 XSI、CAST、姿态、质量控制、弱标签、物理特征和元数据的字段命名；
2. 固定核心数组的维度顺序、dtype、单位和取值范围；
3. 避免不同脚本各自定义字段名、方位约定、标签编码或深度轴；
4. 支持本地 tiny sample、服务器全量数据、离线特征和训练数据的一致读取；
5. 支持数据版本、标签版本、特征版本和配置 hash 的可追溯；
6. 防止 XSI-only 模型误读 CAST Zc 或 CAST label map 造成标签泄漏；
7. 为后续 `schema validator`、`manifest builder`、`QC agent`、`alignment agent` 和训练脚本提供统一依据。

核心原则：

```text
统一维度
统一单位
统一方位约定
统一标签编码
统一质量掩码
统一版本字段
不覆盖旧结果
不把 CAST 输入 XSI-only 模型
```

---

## 2. 数据处理阶段

本项目数据管线必须按阶段执行，不允许跳步。

```text
raw .mat
→ raw manifest
→ XSI / CAST QC
→ depth alignment
→ high-side azimuth normalization
→ strong / weak side modeling
→ CAST effective window
→ weak label generation
→ offline feature extraction
→ baseline validation
→ XSI-only training
→ fusion explanation
→ evaluation and XAI report
```

每个阶段都必须输出：

```text
中间数据文件
配置文件副本或 hash
日志
可视化图
schema / QC 报告
版本信息
```

---

## 3. 文件类型约定

### 3.1 原始数据

原始数据通常来自：

```text
*.mat
```

可能包含：

```text
XSI 波形数据
CAST Zc 数据
Inc
RelBearing
深度轴
仪器姿态
其他测井或质量控制变量
```

原始 `.mat` 文件：

- 不得直接进入训练循环；
- 不得提交到 Git；
- 不得在 PyTorch `Dataset.__getitem__` 中动态读取；
- 只允许在 manifest、QC、对齐和转换阶段读取。

#### 3.1.1 受控 MAT 元数据读取

MVP-1 允许在 raw manifest 之后执行受控 MAT 元数据读取，用于确认原始文件是否可打开、顶层变量名、shape、dtype / class 和启发式 role hint。

约束：

- 优先使用 `scipy.io.whosmat` 读取 MATLAB v5 / v7 文件目录；
- 对 MATLAB v7.3 / HDF5 文件，只允许用 `h5py` 读取 group / dataset 元数据；
- 禁止使用 `scipy.io.loadmat` 读取大型变量；
- 禁止读取完整 XSI 波形矩阵或完整 CAST 图像；
- 该步骤只输出 `mat_metadata_v001.json`，不生成 HDF5，不做 QC 算法、不做 alignment、不做 label。

#### 3.1.2 Raw 变量映射模板

MVP-1 允许基于 `mat_metadata_v001.json` 生成 raw metadata audit report 和 `configs/raw_variable_mapping.example.yaml`。

该模板只记录变量名、shape、dtype / class 和 role hint 的启发式候选，不代表最终确认的变量语义。

约束：

- 不读取真实 `.mat` 内容；
- 不使用 `scipy.io.loadmat`；
- 不做 QC、alignment、RelBearing 旋转、label 或 HDF5 生成；
- `configs/raw_variable_mapping.example.yaml` 可提交 Git；
- `configs/raw_variable_mapping.yaml` 应由人工复制、确认后作为本地私有配置使用，不应提交 Git。

#### 3.1.3 受控 MATLAB struct 字段探查

当 `mat_metadata_v001.json` 显示关键数据藏在 MATLAB 顶层 struct 中时，MVP-1 允许执行受控 struct 字段探查。

约束：

- MATLAB v5 / v7 文件只允许使用 `scipy.io.loadmat(variable_names=[...])` 加载指定 top-level variable；
- 必须限制文件数量、每文件变量数量、字段递归深度和数组 preview 元素数量；
- 只保存字段路径、shape、dtype / class、element_count、role hint 和少量 preview stats；
- 不保存完整数组，不做 QC、alignment、RelBearing 旋转、label 或 HDF5 生成；
- 单个文件探查失败必须写入结构化 errors，不得中断整个探查流程。

#### 3.1.4 Raw 变量映射建议草稿

MVP-1 允许基于 `mat_struct_probe_v001.json` 生成 raw variable mapping suggestions 和
`configs/raw_variable_mapping.draft.yaml`，用于辅助人工确认后续 small-slice reader 的变量路径。

约束：

- 该步骤只读取 struct probe JSON，不重新打开 `.mat` 文件；
- 每个建议必须包含 confidence 和 reason；
- 无法可靠推荐的字段必须保留 `TODO_CONFIRM`；
- `configs/raw_variable_mapping.draft.yaml` 是临时草稿，不应提交 Git；
- `configs/raw_variable_mapping.yaml` 只能由人工确认后创建，并且不应提交 Git；
- 在人工确认 `configs/raw_variable_mapping.yaml` 之前，不得进入 controlled small-slice MAT reader。

#### 3.1.5 受控 small-slice MAT 读取

人工确认 `configs/raw_variable_mapping.yaml` 后，MVP-1 允许读取极小 MAT 片段，用于验证变量路径、
shape、dtype、单位和数值范围。

约束：

- 默认读取不超过 10 个 depth sample、32 个 time sample、13 个 receiver、8 个 side 和 180 个 CAST 方位；
- MATLAB v5 / v7 compressed struct 必须使用受控解析或等价机制，只 materialize 小片段数组；
- 输出仅允许写入外部数据目录的 `interim/`，不得提交 Git；
- 允许生成 `small_slice_summary_v001.json` 和极小 `small_slice_v001.npz`；
- CAST `Zc` 若源顺序为 `[cast_azimuth, depth]`，small-slice canonical 建议为 `[depth, cast_azimuth]`；
- XSI waveform 若源顺序为 `[time, depth]`，small-slice canonical 建议为 `[depth, time]`；
- 不做正式插值、depth alignment、RelBearing 旋转、label 或训练数据生成。

#### 3.1.6 Tiny HDF5 prototype

MVP-1 允许基于 `small_slice_v001.npz` 生成 tiny HDF5 prototype，用于验证最小 HDF5 schema。

约束：

- 输入只能是 Stage 7 small slice，不得重新读取 `.mat`；
- `/aligned/xsi_waveform` 使用 canonical `[depth, receiver, side, time]`；
- `/aligned/cast_zc` 使用 canonical `[depth, cast_azimuth]`；
- 如果 depth 轴不一致，只记录 warning，不做 depth alignment；
- 如果 time 单位未知，可以写入 `/axis/time_sample_index` 代替 `/axis/time_ms`；
- 输出写入外部数据目录的 `processed/`，不得提交 Git；
- 不做 RelBearing 旋转、label、feature extraction 或 model training。

#### 3.1.7 MVP-1 initial QC skeleton

MVP-1 允许在 tiny slice / tiny HDF5 上运行初始 QC skeleton。

允许范围：

- shape check、finite / NaN / Inf / zero ratio；
- simple clipping-like ratio；
- min / max / mean / std；
- CAST Zc、XSI waveform、Inc 和 RelBearing 的基础 range summary；
- 结构化 warnings / errors。

禁止范围：

- STC、APES、复杂首波拾取；
- depth alignment、RelBearing 旋转；
- label generation、feature extraction 或 model training。

---

### 3.2 中间数据

推荐使用：

```text
HDF5
Zarr
NumPy memmap
```

其中：

| 格式 | 主要用途 |
|---|---|
| HDF5 | 主数据容器，适合保存对齐数据、标签、质量权重、物理特征 |
| Zarr | 可选，用于并行读写或云/分块场景 |
| NumPy memmap | 高频访问的大型连续特征缓存 |
| JSON | manifest、schema report、实验 manifest |
| YAML | 配置文件 |
| CSV | 低置信区段、错误案例索引、简表输出 |
| PNG / PDF | 可视化报告 |

---

### 3.3 禁止提交到 Git 的数据文件

以下文件不得提交到 Git：

```text
*.mat
*.h5
*.hdf5
*.zarr/
*.npy
*.npz
*.pt
*.pth
*.ckpt
*.onnx
*.log
data/
outputs/
artifacts/
checkpoints/
models/
logs/
.env
```

---

## 4. 统一维度约定

### 4.1 核心维度顺序

XSI 原始波形维度必须统一为：

```text
depth → receiver → side → time
```

即：

```text
[depth, receiver, side, time]
```

禁止在不同模块中混用：

```text
[receiver, depth, side, time]
[depth, side, receiver, time]
[side, depth, receiver, time]
```

如果原始 `.mat` 数据维度不同，必须在数据转换阶段显式转置，并在 metadata 中记录原始维度顺序。

---

### 4.2 维度名称

| 维度名 | 含义 | 典型长度 |
|---|---|---|
| `depth` | 统一深度采样点 | 数据决定 |
| `receiver` | XSI 接收器编号 | 13 |
| `side` | XSI 方位 Side | 8 |
| `time` | XSI 时间采样点 | 1024 |
| `cast_azimuth` | CAST 方位扇区 | 180 |
| `slowness` | STC 慢度网格 | 配置决定 |
| `time_window` | STC / 特征时间窗 | 配置决定 |
| `feature` | 物理手工特征维度 | 配置决定 |
| `object` | 连通窜槽对象编号 | 数据决定 |

---

### 4.3 单位约定

| 字段 | 单位 |
|---|---|
| 深度 | 优先使用 m；若原始为 ft，必须转换或明确记录 |
| 时间 | ms |
| 采样间隔 | μs 或 s，metadata 中必须明确 |
| 方位角 | degree |
| RelBearing | degree |
| Inc | degree |
| Zc | MRayl |
| 慢度 | μs/ft 或 μs/m，必须在 metadata 中记录 |
| 频率 | Hz 或 kHz，必须在 metadata 中记录 |

建议在 metadata 中保存：

```text
depth_unit
time_unit
sampling_interval_us
azimuth_unit
zc_unit
slowness_unit
frequency_unit
```

---

## 5. 坐标与方位约定

### 5.1 方位角范围

所有方位角统一为：

```text
[0, 360)
```

任何角度写入文件前必须执行 wrap：

```text
theta = theta mod 360
```

---

### 5.2 CAST 方位定义

CAST 有 180 个方位扇区时，默认原始方位为：

```text
theta_cast_raw[j] = 2 * j
j = 0, 1, ..., 179
```

对应 dataset：

```text
/axis/cast_azimuth_deg
```

shape：

```text
[cast_azimuth]
```

---

### 5.3 XSI Side 方位定义

XSI 有 8 个 Side 时，默认形式为：

```text
theta_xsi_raw[i] = theta_0 + 45 * i
i = 0, 1, ..., 7
```

但 `theta_0` 不得武断写死。

必须从以下来源确认：

```text
仪器说明
数据字典
Side A-H 定义
原始 agents.md
专家确认
```

对应 dataset：

```text
/axis/xsi_side_azimuth_deg
```

shape：

```text
[side]
```

---

### 5.4 高边坐标归一化

所有 XSI 与 CAST 的匹配前，必须转换到相对于高边的坐标系。

默认候选公式：

```text
theta_aligned = (theta_raw + RelBearing) mod 360
```

但最终符号必须通过双符号实验确认：

```text
theta_aligned_plus  = (theta_raw + RelBearing) mod 360
theta_aligned_minus = (theta_raw - RelBearing) mod 360
```

必须在 metadata 中保存：

```text
relbearing_sign_convention
relbearing_sign_selected
relbearing_validation_method
relbearing_validation_score
```

推荐取值：

```text
relbearing_sign_selected = "plus" | "minus" | "unknown"
```

---

### 5.5 圆周拓扑

方位维度是圆周，不是线性序列。

所有方位计算必须使用循环距离：

```text
d_theta(theta1, theta2) = min(abs(theta1 - theta2), 360 - abs(theta1 - theta2))
```

Side A 与 Side H 相邻，0° 与 360° 相邻。

模型、插值、IoU、增强和注意力不得将 0° 与 360° 视为远距离。

---

## 6. 推荐 HDF5 顶层结构

推荐主 HDF5 文件结构：

```text
/aligned
/axis
/pose
/alignment
/quality
/label
/objects
/features
/splits
/metadata
```

其中：

| Group | 作用 |
|---|---|
| `/aligned` | 对齐后的 XSI / CAST 主数据 |
| `/axis` | 深度、方位、时间、慢度等坐标轴 |
| `/pose` | 井斜、RelBearing、姿态相关数据 |
| `/alignment` | 深度对齐、方位对齐、lag、置信度 |
| `/quality` | XSI / CAST QC、质量权重、质量掩码 |
| `/label` | 弱标签、严重程度、置信度、uncertain mask |
| `/objects` | 对象级窜槽标签 |
| `/features` | STFT、STC、APES、物理手工特征 |
| `/splits` | train / val / test 划分 |
| `/metadata` | 版本、单位、配置 hash、代码版本 |

---

## 7. 核心 HDF5 Schema

### 7.1 `/aligned`

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/aligned/xsi_waveform` | float32 | `[depth, receiver, side, time]` | 是 | 对齐后的 XSI 原始波形 |
| `/aligned/cast_zc` | float32 | `[depth, cast_azimuth]` | 是 | 对齐后的 CAST Zc |
| `/aligned/cast_zc_to_xsi8` | float32 | `[depth, side]` | 可选 | CAST 180 方位聚合到 XSI 8 Side |
| `/aligned/xsi_waveform_filtered` | float32 | `[depth, receiver, side, time]` | 可选 | 滤波后的 XSI 波形 |

注意：

- `/aligned/xsi_waveform` 应尽量保存原始但已统一维度的数据；
- 滤波结果如需保存，应另存为 `/aligned/xsi_waveform_filtered`；
- 不得覆盖原始对齐波形。

---

### 7.2 `/axis`

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/axis/depth` | float32 | `[depth]` | 是 | 统一深度轴 |
| `/axis/time_ms` | float32 | `[time]` | 是 | XSI 时间轴，单位 ms |
| `/axis/receiver_index` | int16 | `[receiver]` | 是 | 接收器索引 |
| `/axis/side_index` | int16 | `[side]` | 是 | XSI Side 索引 |
| `/axis/xsi_side_azimuth_deg` | float32 | `[side]` | 是 | XSI Side 方位角 |
| `/axis/cast_azimuth_deg` | float32 | `[cast_azimuth]` | 是 | CAST 方位角 |
| `/axis/slowness` | float32 | `[slowness]` | 特征阶段 | STC 慢度轴 |
| `/axis/stc_time_window_ms` | float32 | `[time_window]` | 特征阶段 | STC 时间窗中心 |

---

### 7.3 `/pose`

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/pose/inc_deg` | float32 | `[depth]` | 是 | 井斜角 |
| `/pose/rel_bearing_deg` | float32 | `[depth]` | 是 | RelBearing |
| `/pose/orientation_confidence` | float32 | `[depth]` | 是 | 姿态置信度 |
| `/pose/orientation_uncertain` | bool | `[depth]` | 是 | 低井斜或姿态不稳定标记 |

姿态置信度建议范围：

```text
0.0 = 完全不可信
1.0 = 高可信
```

低井斜段应标记：

```text
orientation_uncertain = true
```

---

### 7.4 `/alignment`

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/alignment/local_depth_lag` | float32 | `[depth]` | 是 | 局部深度错位估计 |
| `/alignment/depth_alignment_confidence` | float32 | `[depth]` | 是 | 深度对齐置信度 |
| `/alignment/alignment_residual` | float32 | `[depth]` | 推荐 | 对齐残差 |
| `/alignment/depth_uncertain_mask` | bool | `[depth]` | 是 | 深度对齐不可信区域 |
| `/alignment/azimuth_alignment_confidence` | float32 | `[depth]` | 是 | 方位对齐置信度 |
| `/alignment/relbearing_sign_candidate_score_plus` | float32 | scalar 或 `[depth]` | 推荐 | +RelBearing 方案得分 |
| `/alignment/relbearing_sign_candidate_score_minus` | float32 | scalar 或 `[depth]` | 推荐 | -RelBearing 方案得分 |

深度错位估计的单位必须与 `/axis/depth` 一致。

#### 7.4.1 MVP-2 depth axis audit artifacts

MVP-2 进入正式插值或标签前，必须先生成 depth-only 审计报告：

```text
/home/xiaoj/cement-channel-data/reports/depth_axis_audit_report.md
/home/xiaoj/cement-channel-data/reports/depth_axis_audit_report.json
```

该报告只允许读取 `CAST.Depth`、`XSILMR{receiver}.Depth` 和 `Depth_inc`，
不得读取完整 XSI waveform 或完整 `CAST.Zc`。报告至少记录：

```text
cast_depth length/min/max/monotonic/median_step/nan_count/duplicate_count
xsi_depth per receiver length/min/max/monotonic/median_step/nan_count/duplicate_count
pose_depth length/min/max/monotonic/median_step/nan_count/duplicate_count
receiver-to-receiver depth consistency
common overlap interval
candidate canonical depth grid
warnings/errors/no-go blockers
```

如果 depth unit 仍为 `unknown_to_verify`，MVP-2 可保留 `conditional_go`，
但必须在报告 warning 中记录并等待人工复核。

#### 7.4.2 MVP-2 canonical depth grid proposal artifacts

Depth audit 通过或 `conditional_go` 后，必须生成受控验证用 canonical depth grid
proposal：

```text
/home/xiaoj/cement-channel-data/reports/depth_grid_proposal.md
/home/xiaoj/cement-channel-data/reports/depth_grid_proposal.json
configs/alignment.depth_grid.example.yaml
```

该阶段只允许读取 `depth_axis_audit_report.json`，不得读取 MAT、waveform 或
`CAST.Zc`。proposal 必须记录：

```text
common_overlap_min/max
depth_start/depth_stop
depth_step
sample_count
grid_order = increasing
allow_extrapolation = false
step selection rationale
warnings/errors/no-go blockers
```

若不同来源的 median depth step 差异较大，可以继续输出保守 grid，但必须记录
warning；若 `depth_start`、`depth_stop` 或 `depth_step` 无法明确计算，则为
`no_go`，不得进入插值预览。

#### 7.4.3 MVP-2 controlled depth-only reader artifacts

Depth grid proposal 明确后，MVP-2 可生成 depth-only / pose-only 受控读取结果：

```text
/home/xiaoj/cement-channel-data/interim/depth_only_v001.npz
/home/xiaoj/cement-channel-data/interim/depth_only_summary_v001.json
```

该文件只允许包含：

```text
cast_depth
xsi_depth_by_receiver
pose_depth
inc_deg
relbearing_deg
```

该阶段不得读取或保存 XSI waveform、完整 `CAST.Zc`、弱标签、特征或模型输入。
summary JSON 必须记录 shape、dtype、finite ratio、范围、warnings/errors 和
`not_performed`。

#### 7.4.4 MVP-2 small-slice depth resampling preview artifacts

Controlled depth-only reader 与 depth grid proposal 均可用后，可生成小片段插值预览：

```text
/home/xiaoj/cement-channel-data/interim/depth_resample_preview_v001.npz
/home/xiaoj/cement-channel-data/reports/depth_resample_preview_report.md
/home/xiaoj/cement-channel-data/reports/depth_resample_preview_report.json
```

该阶段只验证数据能否映射到 proposed canonical depth grid，不得保存正式
aligned HDF5。允许内容包括：

```text
canonical_depth
source index on grid for CAST / pose / XSI receiver depth
pose Inc / RelBearing interpolation preview
small-slice CAST.Zc interpolation preview, only when small-slice depth overlaps the grid
small-slice XSI waveform interpolation preview, only when small-slice depth overlaps the grid
valid masks and interpolation NaN / extrapolation statistics
```

默认禁止外推；若 small-slice 的 CAST / XSI 首段不覆盖 proposed grid，只能记录
`skipped_no_common_overlap` warning，不得读取全量 waveform 或 full `CAST.Zc` 来补齐。

#### 7.4.4b MVP-2 overlap-targeted small-slice artifacts

若默认 `small_slice_v001.npz` 与 proposed canonical grid 没有共同覆盖，可在共同
overlap 中部重新读取受控小片段：

```text
/home/xiaoj/cement-channel-data/interim/small_slice_overlap_v001.npz
/home/xiaoj/cement-channel-data/interim/small_slice_overlap_summary_v001.json
/home/xiaoj/cement-channel-data/interim/depth_resample_overlap_preview_v001.npz
/home/xiaoj/cement-channel-data/reports/depth_resample_overlap_preview_report.md
/home/xiaoj/cement-channel-data/reports/depth_resample_overlap_preview_report.json
```

默认窗口来自 `depth_grid_proposal.json` 的 common overlap 中点，窗口长度不得超过
2.0 m，默认 `max_depth_samples` 和 `max_time_samples` 必须保持小片段规模。
该流程可按 depth offset 读取 MAT 中的局部 `CAST.Zc` 与 XSI waveform，但不得读取
full waveform 或 full `CAST.Zc`。

若 overlap-targeted slice 仍不能形成 CAST / XSI / pose 共同覆盖，报告必须写入
error 并停止 RelBearing 证据增强。

#### 7.4.5 MVP-2 RelBearing angle utilities

RelBearing 方位归一化必须先实现独立、可测试的角度工具，再进入符号验证：

```text
wrap_deg(theta) -> [0, 360)
circular_distance_deg(theta_a, theta_b)
circular_mean_deg(theta, weights)
theta_aligned_plus  = (theta_raw + RelBearing) mod 360
theta_aligned_minus = (theta_raw - RelBearing) mod 360
theta_no_rotation   = theta_raw mod 360
orientation_confidence from Inc
orientation_uncertain low-inc mask
```

此阶段不得选择最终 plus/minus 符号，不得生成标签。若 Side A 的物理零度未确认，
`xsi_side_azimuth_deg` 仍必须来自配置或显式候选，不得在正式 alignment 中硬编码。

Overlap-targeted RelBearing validation 可额外输出：

```text
/home/xiaoj/cement-channel-data/reports/relbearing_sign_validation_overlap_report.md
/home/xiaoj/cement-channel-data/reports/relbearing_sign_validation_overlap_report.json
```

即使 overlap-targeted preview 有 CAST/XSI 小片段证据，若 plus/minus 仍无法区分，
decision 必须保持 `insufficient_evidence`，不得自动写入正式 alignment 配置。

Halliburton RB / Relative Bearing 文档定义下，若 raw side azimuth 以 tool key 为
`0°`，且 looking downhole 顺时针增加，则文档优先公式为：

```text
theta_aligned = (theta_raw + RelBearing) mod 360
```

MVP-2 当前状态必须记录为：

```text
relbearing_sign_status: documentation_preferred_plus_data_unresolved
documentation_preferred_sign: plus
data_driven_validation: insufficient_evidence
single_sign_alignment_approved: false
approved_downstream_mode: plus_primary_minus_ablation
```

该状态不等同于 `confirmed_plus`。Side A-H 相对 tool key 的顺序尚未独立确认，
导出矩阵 / 图像方向仍可能存在 looking-uphole / looking-downhole 翻转，因此不得生成
single-sign production alignment。后续只能以 documentation-preferred plus 为主候选、
minus 为对照消融的 dual-sign / ablation 模式进入下一阶段。

#### 7.4.6 MVP-2 orientation confidence artifacts

RelBearing plus/minus 符号未确认时，仍可独立基于 `Inc` 生成高边方向稳定性
mask。该阶段输入仅为受控 depth-only 输出：

```text
/home/xiaoj/cement-channel-data/interim/depth_only_v001.npz
```

输出：

```text
/home/xiaoj/cement-channel-data/interim/orientation_confidence_v001.npz
/home/xiaoj/cement-channel-data/reports/orientation_confidence_report.md
/home/xiaoj/cement-channel-data/reports/orientation_confidence_report.json
```

`orientation_confidence_v001.npz` 必须至少包含：

```text
pose_depth
inc_deg
orientation_confidence
low_inc_mask
stable_inc_mask
orientation_uncertain
```

默认阈值：

```text
I_min_deg = 1.0
I_stable_deg = 5.0
Inc <= I_min_deg      -> orientation_confidence = 0
Inc >= I_stable_deg   -> orientation_confidence = 1
I_min < Inc < I_stable -> linear transition
```

报告必须记录 Inc min/max/mean、low-inclination 样本比例、stable-inclination
样本比例、orientation confidence 分布、warnings/errors，并明确该 mask 与
RelBearing plus/minus sign 无关。

#### 7.4.7 MVP-2 gate report artifacts

MVP-2 完成 depth audit、depth grid proposal、depth-only reader、overlap-targeted
resampling、RelBearing validation 和 orientation confidence 后，必须生成：

```text
/home/xiaoj/cement-channel-data/reports/mvp2_gate_report.md
/home/xiaoj/cement-channel-data/reports/mvp2_gate_report.json
```

若 depth axes valid、depth grid exists、depth-only reader works、overlap-targeted
resampling works、orientation confidence exists，且 RelBearing 状态为
`documentation_preferred_plus_data_unresolved`，gate decision 必须为
`conditional_go`。该 conditional go 只允许进入 MVP-3 的
`plus_primary_minus_ablation` workflow，不允许 single-sign production alignment、
直接生成最终弱标签、feature extraction 或 model training。若存在 blocking errors，
decision 必须为 `no_go`。

#### 7.4.8 MVP-2C RelBearing / side order / CAST direction manual calibration

MVP-2 gate 为 `conditional_go` 且 RelBearing 仍为
`documentation_preferred_plus_data_unresolved` 时，可执行 MVP-2C 人工审查与多假设
校准。该阶段应先从 `depth_only_v001.npz`、`orientation_confidence_v001.npz` 和
depth grid overlap 中主动扫描多个高质量候选窗口，再对入选窗口按需读取局部
small-slice。不得把单个 fallback window 当作有效证据；每个窗口仍必须保持
small-slice 限制，不得读取完整 XSI waveform 或 full `CAST.Zc`。

必须比较的假设空间：

```text
relbearing_sign = plus / minus
xsi_side_order = clockwise / counterclockwise
cast_azimuth_direction = normal / reversed
side_a_offset_deg = 0,45,90,135,180,225,270,315
```

输出：

```text
/home/xiaoj/cement-channel-data/reports/relbearing_calibration_report.md
/home/xiaoj/cement-channel-data/reports/relbearing_calibration_report.json
/home/xiaoj/cement-channel-data/reports/relbearing_candidate_windows.md
/home/xiaoj/cement-channel-data/reports/relbearing_manual_review/
```

`relbearing_manual_review/` 可包含 CAST Zc raw / plus-minus / normal-reversed 对比图、
XSI side energy raw / plus-minus / side-order 对比图、hypothesis score summary 和
`review_summary_template.md`。图像和报告均为人工审查 artifact，不得提交到 Git。

判定规则：

```text
至少 5 个有效窗口才允许 data-supported recommendation
至少 70% 有效窗口支持同一候选
best-vs-second score gap 必须超过阈值
fallback_window_counted_as_evidence = false
不满足时 final_recommendation = unresolved_keep_plus_primary_minus_ablation
满足时也只能输出 recommendation，不得写成 confirmed
single_sign_alignment_approved = false
production_alignment_config_written = false
```

人工排除区间可通过 `no_eccentric_or_rb_unreliable_intervals` 提供，但必须与项目内部
depth 单位一致。若排除区间使用 `ft` 而内部 depth 轴为 `m` 或 `unknown_to_verify`，
不得直接混用；必须先转换单位或记录 TODO / warning。

---

### 7.5 `/quality`

#### 7.5.1 XSI QC

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/quality/xsi_snr` | float32 | `[depth, side]` | 是 | XSI 各 Side 信噪比 |
| `/quality/xsi_dead_receiver_mask` | bool | `[depth, receiver]` 或 `[receiver]` | 推荐 | 死接收器标记 |
| `/quality/xsi_dead_side_mask` | bool | `[depth, side]` 或 `[side]` | 推荐 | 异常 Side 标记 |
| `/quality/xsi_saturation_ratio` | float32 | `[depth, side]` | 推荐 | 饱和比例 |
| `/quality/xsi_clipping_ratio` | float32 | `[depth, side]` | 推荐 | 截断比例 |
| `/quality/xsi_noise_floor` | float32 | `[depth, side]` | 推荐 | 噪声底 |
| `/quality/xsi_first_arrival_time_ms` | float32 | `[depth, receiver, side]` | 推荐 | 首波到时 |
| `/quality/xsi_time_zero_shift_ms` | float32 | `[depth, side]` | 可选 | 时间零点漂移 |
| `/quality/xsi_receiver_consistency` | float32 | `[depth, side]` | 推荐 | 接收器一致性 |
| `/quality/xsi_polarity_check` | int8 / bool | `[depth, side]` | 可选 | 极性检查 |
| `/quality/xsi_gain_change_flag` | bool | `[depth, side]` | 可选 | 增益变化标记 |

---

#### 7.5.2 CAST QC

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/quality/cast_sector_quality_weight` | float32 | `[depth, cast_azimuth]` | 是 | CAST 方位质量权重 |
| `/quality/cast_missing_sector_ratio` | float32 | `[depth]` | 推荐 | 缺失方位比例 |
| `/quality/cast_echo_amplitude_quality` | float32 | `[depth, cast_azimuth]` | 可选 | 回波幅度质量 |
| `/quality/cast_travel_time_quality` | float32 | `[depth, cast_azimuth]` | 可选 | travel time 质量 |
| `/quality/cast_radius_variation` | float32 | `[depth, cast_azimuth]` | 可选 | 半径变化 |
| `/quality/cast_ovality` | float32 | `[depth]` 或 `[depth, cast_azimuth]` | 可选 | 套管椭圆度 |
| `/quality/cast_thickness_anomaly` | bool | `[depth, cast_azimuth]` | 可选 | 套管厚度异常 |
| `/quality/cast_collar_mask` | bool | `[depth]` 或 `[depth, cast_azimuth]` | 推荐 | 接箍或几何突变掩码 |
| `/quality/cast_low_confidence_mask` | bool | `[depth, cast_azimuth]` | 推荐 | 低置信 CAST 区域 |

---

#### 7.5.3 强弱侧与有效窗口

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/quality/theta_strong_deg` | float32 | `[depth]` | 是 | 强侧 / 贴壁侧方向 |
| `/quality/theta_weak_deg` | float32 | `[depth]` | 是 | 弱侧 / 悬空侧方向 |
| `/quality/eccentricity_score` | float32 | `[depth]` | 是 | 偏心或方位非对称程度 |
| `/quality/strong_side_confidence` | float32 | `[depth]` | 是 | 强侧识别置信度 |
| `/quality/side_quality_weight` | float32 | `[depth, side]` | 是 | XSI Side 质量权重 |
| `/quality/effective_window_mask` | bool | `[depth, side]` | 是 | XSI 有效窗口 Side |
| `/quality/effective_window_quality_mask` | bool | `[depth, side]` | 推荐 | 高质量窗口 |
| `/quality/effective_window_anomaly_mask` | bool | `[depth, side]` | 推荐 | 异常候选窗口 |

---

#### 7.5.4 通用质量标记

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/quality/uncertain_mask` | bool | `[depth, side]` | 是 | 综合不确定区域 |
| `/quality/hard_quality_mask` | bool | `[depth, side]` | 是 | 硬过滤有效区域 |
| `/quality/quality_flags` | uint16 / int32 | `[depth, side]` | 推荐 | 质量与混淆工况 bitmask |

推荐 `quality_flags` bit 定义：

| Bit | 名称 | 含义 |
|---:|---|---|
| 0 | `LOW_INC` | 低井斜，高边方向不稳定 |
| 1 | `DEPTH_UNCERTAIN` | 深度错位或对齐置信度低 |
| 2 | `AZIMUTH_UNCERTAIN` | 方位对齐置信度低 |
| 3 | `XSI_LOW_SNR` | XSI 低信噪比 |
| 4 | `XSI_DEAD_RECEIVER` | XSI 死接收器 |
| 5 | `XSI_DEAD_SIDE` | XSI Side 异常 |
| 6 | `CAST_LOW_QUALITY` | CAST 成像质量差 |
| 7 | `CAST_MISSING_SECTOR` | CAST 方位缺失 |
| 8 | `CAST_COLLAR` | 接箍或几何突变 |
| 9 | `MICROANNULUS_SUSPECT` | 疑似微环隙 |
| 10 | `FREE_PIPE_SUSPECT` | 疑似自由套管 |
| 11 | `LIGHT_CEMENT_SUSPECT` | 疑似轻质或污染水泥 |
| 12 | `ECCENTERING_STRONG` | 强偏心 |
| 13 | `LABEL_LOW_CONFIDENCE` | 低置信标签 |
| 14 | `EXPERT_REVIEW_REQUIRED` | 需要专家复核 |
| 15 | `RESERVED` | 预留 |

如 bit 不够，可升级到 `uint32`。

---

### 7.6 `/label`

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/label/presence` | int8 | `[depth, side]` | 是 | 是否疑似窜槽 |
| `/label/severity` | int8 | `[depth, side]` | 是 | 严重程度 |
| `/label/confidence` | float32 | `[depth, side]` | 是 | 标签置信度 |
| `/label/uncertain_mask` | bool | `[depth, side]` | 是 | 标签不确定区域 |
| `/label/evidence_type` | int16 | `[depth, side]` | 推荐 | 标签证据类型 |
| `/label/azimuth_span_deg` | float32 | `[depth, side]` 或 `[object]` | 可选 | 异常方位宽度 |
| `/label/depth_span` | float32 | `[depth, side]` 或 `[object]` | 可选 | 异常纵向长度 |
| `/label/channel_object_id` | int32 | `[depth, side]` | 是 | 连通对象 ID |

---

### 7.7 标签编码

#### `presence`

```text
 0 = 无窜槽
 1 = 疑似窜槽
-1 = uncertain
```

#### `severity`

```text
 0 = 胶结良好 / 无明显窜槽
 1 = 轻微异常
 2 = 部分窜槽
 3 = 严重窜槽
-1 = uncertain
```

#### `confidence`

```text
0.0 = 完全不可信
1.0 = 高可信
```

#### `evidence_type`

推荐编码：

| 编码 | 名称 | 含义 |
|---:|---|---|
| 0 | `NONE` | 无异常证据 |
| 1 | `ABS_THRESHOLD` | 绝对 Zc 阈值 |
| 2 | `RELATIVE_DROP` | 相对背景下降 |
| 3 | `AZIMUTH_GRADIENT` | 方位梯度异常 |
| 4 | `DEPTH_CONTINUITY` | 纵向连续性证据 |
| 5 | `FUSED_RULE` | 多规则融合 |
| 6 | `EXPERT_REVIEW` | 专家复核标签 |
| 7 | `UNCERTAIN` | 不确定 |

如果一个位置同时满足多个证据类型，建议另存 bitmask：

```text
/label/evidence_flags
```

---

### 7.8 `/objects`

对象级标签用于描述空间连通窜槽对象。

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/objects/channel_object_id` | int32 | `[object]` | 是 | 对象 ID |
| `/objects/channel_depth_top` | float32 | `[object]` | 是 | 顶部深度 |
| `/objects/channel_depth_bottom` | float32 | `[object]` | 是 | 底部深度 |
| `/objects/channel_center_depth` | float32 | `[object]` | 是 | 中心深度 |
| `/objects/channel_center_azimuth_deg` | float32 | `[object]` | 是 | 中心方位 |
| `/objects/channel_azimuth_width_deg` | float32 | `[object]` | 是 | 方位宽度 |
| `/objects/channel_depth_length` | float32 | `[object]` | 是 | 纵向长度 |
| `/objects/channel_continuity_score` | float32 | `[object]` | 是 | 连续性评分 |
| `/objects/channel_confidence` | float32 | `[object]` | 是 | 对象级置信度 |
| `/objects/channel_max_severity` | int8 | `[object]` | 是 | 对象最高严重程度 |
| `/objects/channel_evidence_type` | int16 | `[object]` | 推荐 | 主要证据类型 |

对象 ID 规则：

```text
0 = 背景 / 非对象
1, 2, 3, ... = 连通窜槽对象
-1 = uncertain / 无法归属
```

对象生成必须考虑圆周边界。例如 350°–10° 的异常应视为同一个对象。

---

### 7.9 `/features`

#### STFT

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/features/stft_mag` | float32 | 自定义 | 可选 | STFT 幅度 |
| `/features/stft_phase` | float32 | 自定义 | 可选 | STFT 相位 |
| `/features/stft_reconstruction_error` | float32 | `[depth, side]` 或 scalar | 推荐 | ISTFT 重构误差 |

STFT 参数必须写入 metadata：

```text
stft_window_length
stft_overlap
stft_window_function
stft_boundary_mode
stft_reconstruction_threshold
```

---

#### STC

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/features/stc` | float32 | `[depth, side, slowness, time_window]` | 推荐 | STC 慢度—时间特征 |
| `/features/stc_peak_slowness` | float32 | `[depth, side]` 或 `[depth, side, mode]` | 推荐 | 峰值慢度 |
| `/features/stc_peak_energy` | float32 | `[depth, side]` 或 `[depth, side, mode]` | 推荐 | 峰值能量 |
| `/features/stc_peak_width` | float32 | `[depth, side]` 或 `[depth, side, mode]` | 可选 | 峰宽 |
| `/features/stc_coherence` | float32 | `[depth, side]` | 推荐 | 相干度 |

---

#### APES

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/features/apes_amplitude` | float32 | 自定义 | 可选 | APES 幅度 |
| `/features/apes_phase` | float32 | 自定义 | 可选 | APES 相位 |

---

#### 物理手工特征

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/features/physics_features` | float32 | `[depth, side, feature]` | 推荐 | 物理手工特征 |
| `/features/physics_feature_names` | string | `[feature]` | 推荐 | 特征名 |
| `/features/strong_side_features` | float32 | `[depth, feature]` | 可选 | 强侧特征 |
| `/features/weak_side_features` | float32 | `[depth, feature]` | 可选 | 弱侧特征 |
| `/features/effective_window_features` | float32 | `[depth, feature]` | 可选 | 有效窗口特征 |
| `/features/all_side_robust_features` | float32 | `[depth, feature]` | 可选 | 全方位鲁棒统计特征 |

推荐物理特征名：

```text
casing_wave_energy
fluid_wave_energy
formation_wave_energy
receiver_attenuation_slope
stc_casing_peak
stc_fluid_peak
bandpower_low
bandpower_mid
bandpower_high
envelope_peak
strong_weak_energy_delta
strong_weak_energy_ratio
effective_window_energy
xsi_snr
orientation_confidence
depth_alignment_confidence
```

---

### 7.10 `/splits`

训练、验证、测试划分必须按井或井段分组，避免相邻深度泄漏。

| Dataset | dtype | shape | 必需 | 说明 |
|---|---:|---:|---:|---|
| `/splits/split_name` | string | scalar | 推荐 | 当前 split 名称 |
| `/splits/train_mask` | bool | `[depth]` | 推荐 | 训练深度 |
| `/splits/val_mask` | bool | `[depth]` | 推荐 | 验证深度 |
| `/splits/test_mask` | bool | `[depth]` | 推荐 | 测试深度 |
| `/splits/well_id` | string / int32 | `[depth]` | 推荐 | 井 ID |
| `/splits/group_id` | string / int32 | `[depth]` | 推荐 | 分组 ID |

不得随机逐点划分相邻深度样本。

---

### 7.11 `/metadata`

| Dataset / Attribute | dtype | 必需 | 说明 |
|---|---:|---:|---|
| `/metadata/data_version` | string | 是 | 数据版本 |
| `/metadata/label_version` | string | 是 | 标签版本 |
| `/metadata/feature_version` | string | 是 | 特征版本 |
| `/metadata/schema_version` | string | 是 | 数据契约版本 |
| `/metadata/git_commit` | string | 是 | 代码 commit |
| `/metadata/branch` | string | 推荐 | Git 分支 |
| `/metadata/preprocess_config_hash` | string | 是 | 预处理配置 hash |
| `/metadata/label_config_hash` | string | 是 | 标签配置 hash |
| `/metadata/feature_config_hash` | string | 是 | 特征配置 hash |
| `/metadata/source_files` | string/list | 是 | 原始数据文件 |
| `/metadata/created_at` | string | 是 | 创建时间 |
| `/metadata/created_by` | string | 推荐 | 创建者 |
| `/metadata/host` | string | 推荐 | 主机名 |
| `/metadata/python_version` | string | 推荐 | Python 版本 |
| `/metadata/depth_unit` | string | 是 | 深度单位 |
| `/metadata/time_unit` | string | 是 | 时间单位 |
| `/metadata/azimuth_unit` | string | 是 | 方位单位 |
| `/metadata/zc_unit` | string | 是 | Zc 单位 |
| `/metadata/slowness_unit` | string | 特征阶段 | 慢度单位 |
| `/metadata/sampling_interval_us` | float | 是 | XSI 采样间隔 |
| `/metadata/xsi_receiver_count` | int | 是 | 接收器数量 |
| `/metadata/xsi_side_count` | int | 是 | XSI Side 数量 |
| `/metadata/cast_azimuth_count` | int | 是 | CAST 方位数量 |
| `/metadata/time_sample_count` | int | 是 | 时间采样点数量 |
| `/metadata/relbearing_sign_selected` | string | 是 | RelBearing 符号 |
| `/metadata/model_mode` | string | 训练数据阶段 | 模型模式 |

`model_mode` 推荐取值：

```text
xsi_only
xsi_cast_fusion
label_audit
teacher_distillation
```

---

## 8. 文件命名规范

推荐命名：

```text
<stage>_<data_version>_<label_version>_<feature_version>.h5
```

示例：

```text
aligned_data_v001.h5
labels_v001.h5
features_stft_v001.h5
features_stc_v001.h5
train_xsi_only_v001.h5
```

中间文件不得覆盖旧版本。

如果配置变化，应新建文件或新建版本号。

---

## 9. 版本规则

### 9.1 `data_version`

以下变化必须更新 `data_version`：

```text
原始数据文件变化
深度轴变化
单位转换变化
维度转置规则变化
RelBearing 原始读取方式变化
XSI / CAST 原始变量选择变化
```

示例：

```text
data_v001
data_v002
```

---

### 9.2 `label_version`

以下变化必须更新 `label_version`：

```text
Zc 物理保底阈值变化
自适应基线窗口变化
alpha / beta 变化
方位梯度阈值变化
纵向连续性规则变化
HardQualityMask 变化
label_confidence 公式变化
对象级连通域规则变化
```

示例：

```text
label_v001
label_v002
```

---

### 9.3 `feature_version`

以下变化必须更新 `feature_version`：

```text
STFT 参数变化
ISTFT 重构阈值变化
STC 慢度网格变化
APES 参数变化
滤波参数变化
物理时间窗变化
强弱侧特征定义变化
有效窗口特征定义变化
```

示例：

```text
feature_stft_v001
feature_stc_v001
feature_physics_v001
```

---

### 9.4 `schema_version`

本文件变更字段、shape、dtype 或编码时，必须更新：

```text
schema_version
```

示例：

```text
schema_v001
```

---

## 10. XSI-only 防泄漏规则

对于 `model_mode = xsi_only` 的训练文件或数据加载器：

允许作为输入：

```text
/aligned/xsi_waveform
/pose/inc_deg
/pose/rel_bearing_deg
/pose/orientation_confidence
/quality/side_quality_weight
/quality/effective_window_mask
/features/stft_mag
/features/stft_phase
/features/stc
/features/physics_features
```

允许作为监督目标：

```text
/label/presence
/label/severity
/label/confidence
/label/channel_object_id
/quality/uncertain_mask
```

禁止作为模型输入：

```text
/aligned/cast_zc
/aligned/cast_zc_to_xsi8
/label/presence 作为输入
/label/severity 作为输入
CAST-derived direct feature maps
```

CAST 可以用于：

```text
生成弱标签
生成 label_confidence
生成 effective_window
teacher signal
训练监督
```

但推理阶段不得依赖 CAST。

---

## 11. 数据校验规则

任何 HDF5 / Zarr / memmap 文件在进入训练前必须通过 schema validation。

### 11.1 必检项目

必须检查：

```text
必需 group 是否存在
必需 dataset 是否存在
shape 是否匹配
dtype 是否匹配
depth 维度是否一致
side 维度是否为 8
receiver 维度是否为 13
cast_azimuth 维度是否为 180
time 维度是否为 1024
方位角是否在 [0, 360)
confidence 是否在 [0, 1]
presence 编码是否合法
severity 编码是否合法
uncertain_mask shape 是否匹配
metadata 是否完整
config hash 是否存在
git_commit 是否存在
```

---

### 11.2 建议检查项目

建议检查：

```text
depth 是否单调递增
time_ms 是否单调递增
RelBearing 是否在 [0, 360)
Inc 是否在合理范围
Zc 是否存在异常极值
NaN / Inf 比例
XSI 波形是否全零
CAST 方位缺失比例
label_confidence 分布
HardQualityMask 覆盖比例
quality_flags bit 是否合法
```

---

### 11.3 校验输出

schema validation 应输出：

```text
schema_report.json
schema_report.md
blocking_errors.json
warnings.json
```

阻断性错误包括：

```text
必需字段缺失
核心 shape 不匹配
presence / severity 编码非法
depth 维度不一致
metadata 缺失
XSI-only 训练文件暴露 CAST 输入
```

警告包括：

```text
NaN 比例偏高
低置信标签比例偏高
某些可选 QC 字段缺失
强弱侧置信度偏低
低井斜比例偏高
```

---

## 12. 配置文件对应关系

数据契约依赖以下配置：

```text
configs/paths.local.example.yaml
configs/paths.server.example.yaml
configs/preprocess.yaml
configs/alignment.yaml
configs/label_v001.yaml
configs/feature_stft.yaml
configs/feature_stc.yaml
configs/eval.yaml
```

所有配置文件修改都必须影响对应 hash：

```text
preprocess_config_hash
alignment_config_hash
label_config_hash
feature_config_hash
eval_config_hash
```

---

## 13. 与脚本的关系

推荐脚本与数据契约对应如下：

| 脚本 | 输入 | 输出 |
|---|---|---|
| `scripts/01_build_manifest.py` | raw `.mat` | raw manifest |
| `scripts/02_run_qc.py` | raw / aligned data | `/quality/*` |
| `scripts/03_align_data.py` | raw data + pose | `/aligned/*`, `/axis/*`, `/pose/*`, `/alignment/*` |
| `scripts/04_generate_labels.py` | `/aligned/cast_zc`, QC | `/label/*`, `/objects/*` |
| `scripts/05_extract_features.py` | XSI + QC + config | `/features/*` |
| `scripts/06_train_baseline.py` | features + labels | baseline metrics |
| `scripts/07_train_xsi_only.py` | XSI features + labels | model + metrics |
| `scripts/08_evaluate.py` | predictions + labels | evaluation report |

---

## 14. tiny sample 数据契约

`tests/fixtures/tiny_sample/` 用于 smoke / unit / integration 测试。

tiny sample 可使用极小合成数据，例如：

```text
depth = 4
receiver = 13
side = 8
time = 32
cast_azimuth = 180
```

tiny sample 必须满足：

```text
可公开
体积极小
能快速运行
包含 XSI / CAST / Inc / RelBearing / depth
能覆盖 RelBearing 旋转、CAST 聚合、标签生成、schema 校验
```

tiny sample 不得包含真实敏感井数据。

---

## 15. 最小可接受数据文件

一个进入 MVP-1 的最小 HDF5 文件至少应包含：

```text
/aligned/xsi_waveform
/aligned/cast_zc
/axis/depth
/axis/time_ms
/axis/xsi_side_azimuth_deg
/axis/cast_azimuth_deg
/pose/inc_deg
/pose/rel_bearing_deg
/metadata/data_version
/metadata/schema_version
/metadata/git_commit
/metadata/depth_unit
/metadata/time_unit
/metadata/azimuth_unit
/metadata/zc_unit
```

一个进入训练前的 HDF5 文件至少应额外包含：

```text
/quality/xsi_snr
/quality/side_quality_weight
/quality/cast_sector_quality_weight
/quality/orientation_confidence
/quality/uncertain_mask
/quality/hard_quality_mask
/label/presence
/label/severity
/label/confidence
/label/channel_object_id
/features/physics_features
/metadata/label_version
/metadata/feature_version
/metadata/preprocess_config_hash
/metadata/label_config_hash
/metadata/feature_config_hash
```

---

## 16. 常见错误

禁止出现以下问题：

```text
XSI 维度顺序不一致
CAST 方位不是 [0, 360)
RelBearing 符号没有记录
低井斜段没有 uncertain 标记
presence_label 没有 uncertain 编码
label_confidence 缺失
quality_flags 缺失或无定义
HDF5 缺少 metadata
训练文件没有 data_version / label_version / feature_version
XSI-only 输入中包含 CAST Zc
DataLoader 动态读取 .mat
STC / APES 在 DataLoader 中动态计算
训练集和测试集相邻深度泄漏
```

---

## 17. 后续实现建议

优先实现以下模块：

```text
src/cement_channel/data/schema.py
src/cement_channel/data/manifest.py
src/cement_channel/data/io_hdf5.py
src/cement_channel/utils/angles.py
src/cement_channel/qc/xsi_qc.py
src/cement_channel/qc/cast_qc.py
src/cement_channel/alignment/relbearing.py
src/cement_channel/labels/label_rules.py
```

优先测试：

```text
tests/unit/test_angles.py
tests/unit/test_schema.py
tests/unit/test_label_encoding.py
tests/unit/test_quality_flags.py
tests/unit/test_relbearing.py
tests/integration/test_tiny_sample_schema.py
```

---

## 18. 最终原则

数据契约不是文档摆设，而是所有代码、配置、训练和评估的硬约束。

任何数据文件如果不符合本契约：

```text
不得进入标签生成
不得进入特征提取
不得进入模型训练
不得进入评估报告
不得作为实验结论依据
```

一句话原则：

> 先让数据结构可信，再让标签可信；先让数据版本可追溯，再让模型结果可复现。
