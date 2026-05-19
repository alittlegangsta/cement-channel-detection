# Report

## 固井窜槽智能检测与声波波形可解释分析项目升级报告

## 1. 执行摘要

本项目目标是利用 LWD 单极声波全波列数据 XSI，智能识别固井水泥环窜槽及其严重程度，并用 CAST 超声井壁成像计算得到的声阻抗 Zc 作为弱监督标签来源。现有 `agents.md` 已明确：标签生成采用 “Zc 绝对阈值 + 方位动态梯度” 双重机制，基础阈值参考 `Zc < 2.5 MRayl`，并需按 Zc 降低程度和方位宽度划分胶结良好、轻微/部分窜槽、严重窜槽等等级。

但当前最大风险并不是模型不够复杂，而是 XSI 与 CAST 的空间、方位、姿态和物理响应如果没有先被科学对齐，模型将学习到混乱特征。尤其在存在井斜角和仪器偏心时，声波测井信号会出现一侧强、一侧弱的非对称响应；超声探测由于探头相对套管壁距离、贴壁状态、流体间隙和偏心几何变化，也可能出现类似强弱侧差异。因此，必须把“强弱侧建模”纳入数据预处理、标签置信度、特征提取、采样策略和模型解释的主流程，而不是作为后处理备注。

本报告建议采用五条主线：

1. **统一到高边坐标系**：所有 XSI Side 与 CAST 180 扇区必须先用 RelBearing 旋转到“相对于高边”的坐标系。

2. **显式建模强弱侧**：不再简单对 8 个 Side 或 180 个 CAST 方位取平均，而是区分贴壁侧、悬空侧和不确定侧。

3. **CAST 反推有效窗口**：利用 CAST 180 方位密度识别高质量观测方向与低阻抗异常方向，优先选取相邻的 XSI Side 做精细对比，同时保留全方位 Side 作为模型输入和偏心诊断依据。

4. **训练前做跨模态相关性实验**：验证旋转、加权、有效窗口确实提升 XSI 与 CAST 的一致性。

5. **模型以物理可解释为约束**：先做物理 baseline，再做 STFT/STC 双分支融合模型，最终解释必须回到毫秒时间轴、慢度—时间图和工程语言。

---

## 2. 项目基线与关键约束

本项目数据包括：

| 数据   | 当前说明                                                                               | 对方案影响                                                           |
| ---- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| XSI  | 13 个接收器，每个深度点 8 个方位 Side A-H，1024 个时间采样点，采样间隔 10 μs，总时长 10.24 ms，有效频带主要在 30 kHz 以下 | 需要按“深度 × 接收器 × 方位 × 时间”建模，不能压平成普通表格                             |
| CAST | 180 个方位扇区                                                                          | 可用于高分辨率方位标签、偏心方向估计、有效窗口选择                                       |
| 姿态   | `Inc` 与 `RelBearing`                                                               | 用于坐标旋转、奇点平滑、强弱侧解释                                               |
| 深度匹配 | 缺少 GR 曲线，只能依赖数学插值，以 CAST 高分辨率深度轴为基准                                                | 必须保存对齐残差和对齐审计图                                                  |
| 工程环境 | Ubuntu 18.04.6，3 张 A100 40GB，数据与模型必须保存到 `/home`                                    | 训练脚本默认使用 `CUDA_VISIBLE_DEVICES=1,2`，大 `.mat` 必须先转 HDF5 或 memmap |
| 任务流程 | Step 1–4 每步必须输出可视化和中间文件                                                            | 任何一步不可跳过，不允许直接训练黑盒模型                                            |

这些约束来自项目原始 `agents.md`，必须作为硬约束执行。

### 2.1 统一数据契约与 HDF5 Schema

为保证数据预处理、标签生成、特征工程、模型训练和可解释性分析之间的一致性，本项目必须在正式训练前冻结统一数据契约。所有中间文件和最终训练文件均应遵循明确的 HDF5 / Zarr / NumPy memmap schema，禁止不同脚本各自定义字段名、维度顺序或方位约定。

推荐 HDF5 结构如下：

| Group / Dataset                           | dtype          | shape                                     | 说明                                     |
| ----------------------------------------- | --------------:| -----------------------------------------:| -------------------------------------- |
| `/aligned/xsi_waveform`                   | float32        | `[depth, receiver, side, time]`           | 对齐后的 XSI 原始波形                          |
| `/aligned/cast_zc`                        | float32        | `[depth, cast_azimuth]`                   | 对齐后的 CAST 声阻抗                          |
| `/axis/depth`                             | float32        | `[depth]`                                 | 统一深度轴                                  |
| `/axis/xsi_side_azimuth_deg`              | float32        | `[side]`                                  | XSI 8 个 Side 的原始或对齐方位                  |
| `/axis/cast_azimuth_deg`                  | float32        | `[cast_azimuth]`                          | CAST 180 个方位角                          |
| `/pose/inc_deg`                           | float32        | `[depth]`                                 | 井斜角                                    |
| `/pose/rel_bearing_deg`                   | float32        | `[depth]`                                 | 相对方位角                                  |
| `/alignment/local_depth_lag`              | float32        | `[depth]`                                 | 局部深度错位估计                               |
| `/alignment/depth_alignment_confidence`   | float32        | `[depth]`                                 | 深度对齐置信度                                |
| `/alignment/azimuth_alignment_confidence` | float32        | `[depth]`                                 | 方位对齐置信度                                |
| `/quality/xsi_snr`                        | float32        | `[depth, side]`                           | XSI 各方位信噪比                             |
| `/quality/side_quality_weight`            | float32        | `[depth, side]`                           | XSI Side 质量权重                          |
| `/quality/cast_sector_quality_weight`     | float32        | `[depth, cast_azimuth]`                   | CAST 扇区质量权重                            |
| `/quality/orientation_confidence`         | float32        | `[depth]`                                 | 姿态置信度                                  |
| `/quality/uncertain_mask`                 | bool           | `[depth, side]`                           | 不确定区域掩码                                |
| `/quality/quality_flags`                  | uint16 / int32 | `[depth, side]` 或 `[depth, cast_azimuth]` | 质量与混淆工况标记，如低井斜、接箍、CAST 缺失、XSI 死道、深度错位等 |
| `/label/presence`                         | int8           | `[depth, side]`                           | 是否疑似窜槽                                 |
| `/label/severity`                         | int8           | `[depth, side]`                           | 窜槽严重程度                                 |
| `/label/confidence`                       | float32        | `[depth, side]`                           | 标签置信度                                  |
| `/label/channel_object_id`                | int32          | `[depth, side]`                           | 连通窜槽对象 ID                              |
| `/features/stc`                           | float32        | `[depth, side, slowness, time_window]`    | STC 慢度—时间特征                            |
| `/features/stft_mag`                      | float32        | 自定义                                       | STFT 幅度                                |
| `/features/stft_phase`                    | float32        | 自定义                                       | STFT 相位                                |
| `/features/physics_features`              | float32        | `[depth, side, feature]`                  | 物理手工特征                                 |
| `/metadata/data_version`                  | string         | scalar                                    | 数据版本                                   |
| `/metadata/label_version`                 | string         | scalar                                    | 标签版本                                   |
| `/metadata/feature_version`               | string         | scalar                                    | 特征版本                                   |
| `/metadata/git_commit`                    | string         | scalar                                    | 代码版本                                   |
| `/metadata/preprocess_config_hash`        | string         | scalar                                    | 预处理配置 hash                             |

维度顺序必须固定为：

```text
depth → receiver → side → time
```

---

## 3. 两类测井响应的物理关系

声波 XSI 与超声 CAST 并不是简单的一一对应关系。

XSI 声波全波列主要反映套管波、地层波、流体波等多种模态的传播和衰减。若套管—水泥胶结良好，套管波能量向水泥和地层泄散，套管波幅度应明显衰减；若存在自由套管、微环隙或窜槽，套管波能量可能增强，地层波响应也可能发生变化。

CAST 超声则主要通过高频脉冲回波和套管谐振衰减估计套管外材料声阻抗 Zc，因而在周向上更敏感，适合识别低声阻抗条带、局部缺失水泥和方位连续窜槽。

因此，正确融合方式不是简单让 XSI 拟合 CAST 图像，而是建立如下关系：

| CAST 表现                 | XSI 期望响应                        | 解释              |
| ----------------------- | ------------------------------- | --------------- |
| 某方位 Zc 明显低于背景           | 相应高边归一化方位的套管波能量、衰减斜率、STC 峰值可能异常 | 可能存在局部流体通道或胶结缺失 |
| CAST 低 Zc 条带纵向连续        | XSI 异常应在深度上具有连续性，而非单点随机噪声       | 窜槽是空间连通缺陷       |
| CAST 整体低 Zc 但无明显方位梯度    | 可能是轻质水泥、污染水泥或整体材料差异             | 不应直接判为方位窜槽      |
| XSI 一侧强、一侧弱，CAST 也呈半边异常 | 需优先排查井斜、偏心、贴壁/悬空侧差异             | 可能是仪器几何导致的观测偏差  |
| CAST 与 XSI 对齐后无稳定相关     | 不应进入主模型训练                       | 标签或对齐策略可能失效     |

---

### 3.1 关键混淆工况与排除策略

在弱监督标签生成和模型训练中，必须显式区分“真实窜槽”与其他可能造成类似声学响应的复杂工况。若不加区分，模型可能把自由套管、微环隙、轻质水泥、快速地层或仪器伪影错误学习为窜槽特征。

| 混淆工况        | 可能表现                      | 误判风险           | 处理策略                                    |
| ----------- | ------------------------- | -------------- | --------------------------------------- |
| 微环隙         | XSI 套管波增强，CAST 可能局部低阻抗或异常 | 容易与窜槽混淆        | 标记为 `microannulus_suspect`，不直接作为高置信窜槽标签 |
| 自由套管        | 长井段套管波强、衰减低               | 可能被误判为严重窜槽     | 结合纵向长度、方位宽度和地层波响应区分                     |
| 轻质水泥 / 污染水泥 | CAST 背景整体偏低，方位梯度不明显       | 可能导致大量假阳性      | 使用自适应基线、材料元数据和低置信标签                     |
| 快速地层        | 到时提前，波形模态重叠               | 可能污染套管波时间窗     | 引入首波到时 QC 和地层波异常标记                      |
| 套管接箍        | 局部强反射或阻抗突变                | 可能被误判为窜槽边界     | 若有接箍曲线或几何信息，应生成 `casing_collar_mask`    |
| 工具卡顿 / 测速异常 | 深度采样不均，局部波形异常             | 可能形成伪连续异常      | 引入 logging speed / depth QC             |
| CAST 成像质量差  | 方位缺失、回波弱、阻抗异常跳变           | 低 Zc 不一定代表水泥缺失 | 降低 `cast_sector_quality_weight`         |
| 仪器偏心        | 半边强、半边弱                   | 可能把偏心方向误判为窜槽方向 | 使用强弱侧权重、有效窗口和跨模态一致性验证                   |

所有疑似混淆工况都应进入 `quality_flags`，并参与 `label_confidence` 计算。对于无法明确区分的区域，应优先标记为 `uncertain`，而不是强行生成硬标签。

---

## 4. 坐标系归一化：所有匹配前必须先旋转

### 4.1 高边坐标定义

在任何数据匹配、标签生成、相关性分析或模型训练前，必须将 XSI 与 CAST 统一映射到“相对于高边”的坐标系。

在未确认厂家定义前，采用以下公式作为默认候选约定；最终符号方向必须通过 4.2 的双符号实验和专家抽查确认：

$$
\theta_{\text{aligned}} = (\theta_{\text{raw}} + RelBearing) \bmod 360  
$$

其中：

- $\theta_{\text{raw}}$：原始仪器坐标系下的方位角；

- `RelBearing`：工具相对高边的方位角；

- $\theta_{\text{aligned}}$：旋转后相对于高边的方位角。

对于 CAST：

$$
\theta^{CAST}_{j,\text{raw}} = 2j,\quad j=0,1,\dots,179
$$

因为 CAST 有 180 个方位扇区，每个扇区约 2°。

对于 XSI：

$$
\theta^{XSI}_{i,\text{raw}} = \theta_0 + 45i,\quad i=0,1,\dots,7
$$

但这里需要注意：$\theta_0$ 不能武断假设，必须从仪器说明、数据字典或 Side A-H 的定义中确认。如果 Side A 的物理零度未知，应把 `xsi_side_azimuth_deg` 作为配置文件保存，而不是写死在代码里。

### 4.2 RelBearing 符号方向必须验证

虽然默认使用：

$$
\theta_{\text{aligned}} = (\theta_{\text{raw}} + RelBearing) \bmod 360
$$

但实际工具厂家或数据导出约定可能存在正负方向差异。因此建议在预处理阶段同时测试：

$$
\theta_{\text{aligned}}^{(+)} = (\theta_{\text{raw}} + RelBearing) \bmod 360
$$

$$
\theta_{\text{aligned}}^{(-)} = (\theta_{\text{raw}} - RelBearing) \bmod 360
$$

然后通过以下指标选择正确符号：

1. CAST 强弱侧图像是否与井斜高边/低边物理预期一致；

2. XSI 方位能量峰值与 CAST 异常条带的循环互相关是否在 0° 附近达到最大；

3. 随机旋转或错误符号下的相关性是否显著下降；

4. 专家抽查典型深度段是否符合 VDL/CAST 联合解释。

### 4.3 低井斜段处理

`agents.md` 已指出，当 `Inc` 趋近 0° 时，需要平滑处理 `RelBearing` 奇点噪声。

建议定义姿态置信度：

$$
c_{inc}(d)= 
\begin{cases} 
0, & Inc(d) < I_{min} \\ 
\frac{Inc(d)-I_{min}}{I_{stable}-I_{min}}, & I_{min}\le Inc(d)<I_{stable} \\ 
1, & Inc(d)\ge I_{stable} 
\end{cases}
$$

当井斜过低时，高边方向本身不稳定，此时不应强行使用 RelBearing 生成高置信方位标签，而应把该深度段标记为 `orientation_uncertain`。

### 4.4 局部深度错位估计

仅依赖数学插值可以统一 XSI 与 CAST 的采样网格，但不能保证两类测井响应在物理深度上完全同步。由于工具响应深度、采样密度、测速误差、井眼环境和不同仪器处理流程的差异，局部井段仍可能存在深度错位。若不显式评估该问题，CAST 生成的弱标签可能被错误赋给相邻深度的 XSI 波形。

因此，在完成基础深度插值后，应进一步估计局部深度 lag：

$$
\Delta d^*(z)=\arg\max_{\Delta d} corr(F_{XSI}(z), F_{CAST}(z+\Delta d))
$$

其中：

- $F_{XSI}(z)$：XSI 在深度 \(z\) 附近的物理特征，如套管波能量、STC 峰值、强弱侧能量比；
- $F_{CAST}(z)$：CAST 在深度 \(z\) 附近的标签特征，如低 Zc 占比、方位梯度、纵向连续性；
- $\Delta d^*(z)$：局部最优深度偏移。

建议保存以下字段：

| 字段                           | 含义        |
| ---------------------------- | --------- |
| `local_depth_lag`            | 局部最佳深度偏移  |
| `depth_alignment_confidence` | 深度对齐置信度   |
| `alignment_residual`         | 对齐残差      |
| `depth_uncertain_mask`       | 深度错位不可信区域 |

若某深度窗的最佳 lag 超过预设阈值，或互相关峰值不明显，则该区域不应进入高置信硬监督训练，而应降低 `label_confidence` 或标记为 `uncertain`。深度错位实验也应作为训练前 go/no-go 检查的一部分。

---

## 5. 强弱侧建模：不能再简单平均

### 5.1 为什么不能简单平均

在井斜或偏心条件下，仪器并不位于井筒中心。贴近套管壁的一侧通常信号更强、信噪比更高；远离套管壁的一侧可能由于传播路径、耦合状态、流体间隙和几何衰减导致信号变弱。此时如果直接对 8 个 XSI Side 求平均，会把强侧的有效地层/套管响应与弱侧的低信噪比数据混在一起。

CAST 也是同理。虽然 CAST 有 180 个密集方位点，但偏心、贴壁、椭圆度或局部探测条件变化可能导致某些方位的 Zc 更可靠，某些方位更像观测伪影。`gemini_report.md` 已指出仪器偏心会造成声波到达时间和能量分布不对称，也会导致超声阻抗计算失真和图像伪影。

因此，强弱侧不是一个可以忽略的噪声项，而是必须进入数据结构的核心变量。

### 5.2 强弱侧估计

每个深度 (d) 上建议估计一个偏心/强侧方向：

$$
\theta_{strong}(d)
$$

优先使用 CAST 的 180 方位数据估计，因为 CAST 方位分辨率更高。若 CAST 文件中包含半径、回波幅度、旅行时或质量控制变量，应优先用这些变量估计贴壁侧；如果目前只有 Zc，则可以用 Zc 的周向非对称性、局部稳定性、缺失/异常模式作为间接指标，但必须降低置信度，避免把真实窜槽误认为偏心。

建议保存以下字段：

| 字段                           | 含义                     |
| ---------------------------- | ---------------------- |
| `theta_strong_deg`           | 估计强侧/贴壁侧方向             |
| `theta_weak_deg`             | 与强侧相反的弱侧/悬空侧方向         |
| `eccentricity_score`         | 方位非对称强度                |
| `strong_side_confidence`     | 强侧识别置信度                |
| `orientation_uncertain`      | RelBearing 或 Inc 不稳定标记 |
| `side_quality_weight`        | 每个 XSI Side 的质量权重      |
| `cast_sector_quality_weight` | 每个 CAST 扇区的质量权重        |

### 5.3 能量加权补偿

对 XSI 方位特征聚合时，不能使用简单平均：

$$
\bar{x}(d)=\frac{1}{8}\sum_{i=1}^{8}x(d,i)
$$

应改为质量加权：

$$
\bar{x}_{weighted}(d)=
\frac{\sum_{i=1}^{8}w_i(d)x(d,i)}
{\sum_{i=1}^{8}w_i(d)+\epsilon}
$$

其中权重$w_i(d)$可由以下因素组成：

$$
w_i(d)=c_{snr,i}(d)\cdot c_{orient}(d)\cdot c_{cast,i}(d)\cdot c_{window,i}(d)
$$

含义如下：

| 权重项            | 作用                   |
| -------------- | -------------------- |
| $c_{snr,i}$    | XSI 该 Side 的信噪比      |
| $c_{orient}$   | RelBearing/Inc 方向稳定性 |
| $c_{cast,i}$   | CAST 对应方位质量          |
| $c_{window,i}$ | 是否位于有效窗口内            |

但需要强调：**强侧权重高，只代表观测质量更高，不代表该方位一定更接近真实地层缺陷。** 因此弱侧不能完全丢弃，而应同时保留：

1. 强侧特征；

2. 弱侧特征；

3. 强弱侧差异特征；

4. 全方位鲁棒统计特征；

5. 有效窗口特征。

建议特征包括：

$$
E_{strong},\quad E_{weak},\quad \Delta E = E_{strong}-E_{weak},\quad R_E=\frac{E_{strong}}{E_{weak}+\epsilon}
$$

### 5.4 周向拓扑与圆周建模

XSI 的 8 个 Side 与 CAST 的 180 个方位扇区本质上位于井筒圆周上，方位维度不是普通线性序列。Side A 与 Side H 在物理上相邻，CAST 的 0° 与 358° 也相邻。因此，所有方位建模、卷积、注意力、插值、距离计算和评估指标都必须遵循圆周拓扑。

建议采用以下约束：

1. **循环 padding**：在方位维度进行卷积或局部窗口操作时，应使用 circular padding，而不是 zero padding。

2. **循环距离**：方位误差必须使用循环角距离：
   
   $$
   d_\theta(\theta_1,\theta_2)=\min(|\theta_1-\theta_2|, 360-|\theta_1-\theta_2|)
   $$

3. **角度编码**：模型输入中若包含方位角，应使用 \(\sin\theta\) 与 \(\cos\theta\) 编码，而不是直接输入 0–360 数值。

4. **循环增强**：若进行方位数据增强，只能使用保持物理意义的循环平移，并同步更新 RelBearing、Side 方位和标签方位。

5. **方位 IoU**：评估方位定位时，应使用圆周连续区域的 IoU，而不是普通线性区间 IoU。

该设计可以避免模型错误地认为 0° 和 360° 是远距离位置，从而提升窜槽方位定位的物理一致性。

---

## 6. CAST 180 方位用于识别有效窗口

CAST 有 180 个方位点，远密于 XSI 的 8 个 Side。应充分利用这一优势，先在 CAST 中识别仪器偏心方向、低阻抗异常方向和高可信探测方向，再映射回 XSI Side。

### 6.1 有效窗口定义

对每个深度 (d)，定义有效窗口：

$$
W_{quality}(d)=
\{\theta: |\theta-\theta_{strong}(d)|_{circ}\le \Delta\theta_q\}
$$

$$
W_{anomaly}(d)=
\{\theta: |\theta-\theta_{lowZc}(d)|_{circ}\le \Delta\theta_a\}
$$

$$
W_{eff}(d)=W_{quality}(d)\cup W_{anomaly}(d)
$$

其中：

- $W_{quality}$：偏向贴壁侧或高信噪比方向；

- $W_{anomaly}$：偏向 CAST 低阻抗或高方位梯度异常方向；

- $W_{eff}$：综合质量与异常证据后的精细对比窗口。

其中 $|\cdot|_{circ}$表示循环角距离；$\Delta\theta_q$ 为高质量观测窗口半宽，$\Delta\theta_a$ 为异常候选窗口半宽，二者可初始设为 30°–60°，再通过消融实验调优。  

$\theta_{lowZc}(d)$表示 CAST 在深度 \(d\) 处低阻抗异常或最大方位梯度异常的中心方位；若该深度不存在可靠低阻抗异常，则 $W_{anomaly}(d)$ 可置为空集或低置信窗口。

XSI 的 8 个 Side 中，只有落入或接近该窗口的 Side 被用于精细声波—超声对比：

$$
S_{eff}(d)=
\{i:\theta^{XSI}_{i,aligned}\in W_{eff}(d)\}
$$

如果没有 Side 完全落入窗口，则选择循环角距离最近的 1–2 个 Side。

### 6.2 有效窗口不等于只看强侧

有效窗口有两个用途：

1. **提高相关性分析的信噪比**：用高质量方位验证 XSI 与 CAST 是否存在物理一致性；

2. **避免全方位平均稀释窜槽特征**：窄方位窜槽可能被 8 Side 平均或 180 扇区平均冲淡。

但最终模型仍应看到全方位结构。推荐输入包含：

| 输入组                     | 目的            |
| ----------------------- | ------------- |
| `xsi_all_sides`         | 保留完整周向信息      |
| `xsi_eff_window`        | 提供高 SNR 精细对比  |
| `xsi_strong_weak_delta` | 显式建模偏心/井斜影响   |
| `cast_label_180`        | 生成弱标签         |
| `cast_label_to_xsi_8`   | 与 XSI Side 对齐 |
| `quality_masks`         | 告诉模型哪些区域不可信   |

---

## 7. 标签体系升级：从硬标签改为弱标签 + 置信度 + 不确定区

原始 `agents.md` 的标签策略是 CAST Zc 绝对阈值与动态梯度结合，并划分严重程度。这个方向正确，但必须升级为弱监督标签体系，因为报告已明确当前 Ground Truth 策略仍不完善，需要结合数据分析和实验验证继续优化。需要强调的是，CAST 生成的标签不是绝对真值，而是带有仪器响应、成像质量、偏心状态、材料背景和解释规则偏差的 weak label / teacher label。模型训练目标不是无条件复制 CAST，而是在 XSI 物理响应支持下学习可解释的窜槽证据。

建议标签对象包含：

| 字段                 | 说明                                                                                                                          |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `presence_label`   | 是否疑似窜槽：0/1/uncertain`presence_label` 建议采用 `int8` 编码：0 表示无窜槽，1 表示疑似窜槽，-1 表示 uncertain；同时保留独立的 `uncertain_mask`，以便训练时屏蔽不确定样本。 |
| `severity_label`   | 0 良好，1 轻微，2 部分，3 严重                                                                                                         |
| `azimuth_span_deg` | 低 Zc 异常方位宽度                                                                                                                 |
| `depth_span`       | 异常纵向连续长度                                                                                                                    |
| `label_confidence` | 标签置信度                                                                                                                       |
| `uncertain_mask`   | 对齐差、姿态差、强弱侧冲突、低井斜等区域                                                                                                        |
| `evidence_type`    | threshold / gradient / continuity / expert_review                                                                           |
| `label_version`    | 标签规则版本                                                                                                                      |

标签生成逻辑建议为：

$$
label = f(Zc,\ \nabla_{\theta}Zc,\ continuity_{depth},\ span_{\theta},\ quality)
$$

其中不要只看 `Zc < 2.5`，还要看：

1. 方位局部突变；

2. 是否纵向连续；

3. 是否与强弱侧偏心模式混淆；

4. 是否在低井斜 RelBearing 不稳定区；

5. 是否只出现在 CAST 质量差方位；

6. 是否与 XSI 套管波/STC 证据相互支持。

### 7.1 自适应背景基线追踪策略

考虑到不同层段的泥浆密度变化、套管壁厚差异以及传感器温漂会导致声阻抗 $Zc$ 的背景值发生偏移，本项目不再依赖单一固定阈值，而是将固定阈值降级为物理保底阈值，与局部自适应基线、方位梯度和纵向连续性共同构成弱标签判据。引入自适应基线算法：

1. **局部基线估计**：利用长距离滑动窗口（自适应窗口，默认 50–100 m，可根据 Zc 背景稳定性、套管段边界和元数据自动缩放）计算当前井段的 $Zc$ 中位数或高分位数（如 90% 分位,若窗口内 Zc 分布整体低、方位梯度弱且缺少可靠高阻抗样本，则不强行生成高置信标签，而是降低 `label_confidence`），将其定义为该段的“胶结良好基线值” $Zc_{base}$。

2. **相对下降阈值**：将窜槽判定标准定义为相对降幅。
   
   - **判别公式**：
     
     $$
     Label_{candidate}(d,\theta)=  
\Big[  
Zc(d,\theta) < Zc_{base}(d)\cdot(1-\alpha)  
\Big]  
\cup  
\Big[  
Zc(d,\theta) < Zc_{min\_limit}  
\Big]  
\cup
\Big[  
|\nabla_\theta Zc(d,\theta)|>\tau_\nabla  
\land  
Zc(d,\theta)<Zc_{base}(d)\cdot(1-\beta)  
\Big]
     $$

其中 $0<\beta<\alpha$。方位梯度项不能单独触发高置信窜槽标签，必须同时伴随一定程度的相对低阻抗下降，以减少接箍、偏心、成像质量差或局部噪声导致的假阳性。  

随后施加纵向连续性和质量约束：

$$
Label_{final}(d,\theta)=
Label_{candidate}(d,\theta)
\cap ConnectedDepthSpan(L_{min})
\cap HardQualityMask(d,\theta)
$$

其中 `HardQualityMask` 仅用于排除严重无效数据，例如 CAST 大面积缺失、XSI 死接收器、深度严重错位或接箍/几何突变强干扰区域。对于质量一般但并非完全无效的样本，不直接删除，而是降低标签置信度：

$$
label\_confidence =
q_{cast}^{\gamma_1}
\cdot q_{xsi}^{\gamma_2}
\cdot c_{orient}^{\gamma_3}
\cdot c_{depth}^{\gamma_4}
\cdot c_{continuity}^{\gamma_5}
$$

$$
label\_confidence \leftarrow clip(label\_confidence,0,1)
$$

3. **环境因子补偿**：在计算基线时，自动关联井眼尺寸和套管规格元数据。若 $Zc$ 偏离物理预期范围过大，则自动降低该段标签的置信度权重（`label_confidence`），并在训练中通过代价敏感学习进行抑制。

### 7.2 对象级窜槽标签

窜槽不是孤立的单点异常，而是具有纵向连续性和方位宽度的空间连通对象。因此，除了逐深度、逐方位的像素级标签外，还应构建对象级标签，用于工程解释、专家复核和对象级评估。

建议新增以下对象级字段：

| 字段                         | 说明                                                        |
| -------------------------- | --------------------------------------------------------- |
| `channel_object_id`        | 连通窜槽对象编号                                                  |
| `channel_depth_top`        | 窜槽对象顶部深度                                                  |
| `channel_depth_bottom`     | 窜槽对象底部深度                                                  |
| `channel_center_depth`     | 窜槽中心深度                                                    |
| `channel_center_azimuth`   | 窜槽中心方位                                                    |
| `channel_azimuth_width`    | 窜槽方位宽度                                                    |
| `channel_depth_length`     | 窜槽纵向长度                                                    |
| `channel_continuity_score` | 纵向连续性评分                                                   |
| `channel_confidence`       | 对象级置信度                                                    |
| `channel_evidence_type`    | threshold / gradient / continuity / expert_review / fused |

对象级标签可通过对 `presence_label` 和 `severity_label` 进行深度—方位连通域分析获得。生成对象时应考虑圆周边界，例如跨越 350°–10° 的异常应被视为同一个连通对象，而不是两个断裂对象。

对象级标签的作用包括：

1. 支持“窜槽段”级别的专家复核；
2. 计算对象级 Recall、Precision 和 IoU；
3. 统计窜槽顶部、底部、中心方位和方位宽度误差；
4. 避免模型只在像素级取得高分，却无法识别连续工程缺陷。

---

## 8. 训练前必须完成的跨模态相关性实验

这是新版报告最关键的新增部分。

在完成数据预处理、深度对齐、RelBearing 旋转、强弱侧建模和有效窗口选择后，不能立刻训练深度模型。必须先做一组科学实验，证明 XSI 与 CAST 之间存在稳定、可解释、显著优于随机对齐的相关性。

### 8.1 核心假设

| 假设             | 内容                                                      | 通过标准                            |
| -------------- | ------------------------------------------------------- | ------------------------------- |
| H1 方位归一化有效     | RelBearing 旋转后，XSI 方位特征与 CAST 方位标签的循环相关性提高              | 旋转后相关性显著高于旋转前                   |
| H2 强弱侧权重有效     | 强侧/有效窗口内的 XSI 与 CAST 相关性高于全方位简单平均                       | 加权方法优于 simple mean              |
| H3 CAST 有效窗口有效 | CAST 180 方位识别出的质量窗口与异常窗口能提升 XSI–CAST 相关性、方位定位和窄方位异常检出能力 | 有效窗口内 AUCPR/F1/方位 IoU 高于全方位简单平均 |
| H4 STC 物理特征有效  | STC 套管波/流体波特征与 CAST 低 Zc 异常存在对应关系                       | STC baseline 优于随机特征             |
| H5 标签不是纯噪声     | 用物理特征 baseline 能预测弱标签，且优于方位随机打乱                         | permutation test 显著             |

### 8.2 推荐实验设计

#### 实验 A：方位循环互相关

对每个深度 (d)，将 CAST 180 方位聚合到 XSI 8 Side：

$$
Zc_{8}(d,i)=Agg_{\theta\in sector_i} Zc(d,\theta)
$$

聚合时不要只用均值，应同时保存：

- sector median；

- sector min；

- sector 10% quantile；

- fraction below threshold；

- max azimuthal gradient。

然后提取 XSI 方位特征：

$$
F_{xsi}(d,i)={E_{casing}, E_{fluid}, STC_{peak}, attenuation, envelope, bandpower}
$$

由于 XSI 异常特征通常是“数值越大越异常”，而 CAST 的 Zc 是“数值越低越异常”，因此先定义 CAST 方位异常强度：

$$
A_{cast}(d,i)=
\frac{Zc_{base}(d)-Zc_8(d,i)}
{Zc_{base}(d)+\epsilon}
$$

然后计算每个深度窗的循环互相关：

$$
\rho_d(k)=
corr_{i=1,\dots,8}
\left(
F_{xsi}(d,i),
A_{cast}(d,(i+k)\bmod 8)
\right)
$$

跨深度聚合后得到：

$$
\rho(k)=
median_{d\in D}\left[\rho_d(k)\right]

$$

理想情况下，正确 RelBearing 旋转后，\(\rho(k)\) 的峰值应接近 \(k=0\)，并显著高于未旋转、错误符号旋转或随机方位打乱结果。

#### 实验 B：强弱侧加权消融

比较四种策略：

| 策略         | 描述                                |
| ---------- | --------------------------------- |
| Baseline-0 | 不旋转，8 Side 简单平均                   |
| Baseline-1 | RelBearing 旋转后简单平均                |
| Method-1   | RelBearing 旋转 + 强弱侧               |
| Method-2   | RelBearing 旋转 + CAST 有效窗口 + 强弱侧特征 |

比较指标：

- Spearman / Pearson 相关；

- distance correlation；

- AUCPR；

- F1；

- Recall；

- IoU；

- permutation p-value。

只有 Method-1 或 Method-2 明显优于 Baseline-0/1，才说明强弱侧建模确实有价值。

#### 实验 C：随机方位打乱检验

对 CAST 方位随机循环平移：

$$
Zc_{shuffle}(d,\theta)=Zc(d,\theta+\delta_d)
$$

其中 $\delta_d$ 为随机角度。若模型或相关性分析在随机打乱后仍然表现很好，说明它可能学到的是深度趋势、全局阻抗背景或数据泄漏，而不是方位窜槽。

#### 实验 D：深度错位检验

人为将 CAST 标签上下平移若干深度采样点，比较性能变化。如果错位后模型仍然不降分，说明模型没有真正利用局部深度对应关系。

#### 实验 E：物理 baseline 检验

在训练深度模型前，必须先训练一个物理特征 baseline：

输入：

- XSI 套管波时间窗能量；

- 流体波时间窗能量；

- 13 接收器衰减斜率；

- STC 套管波慢度峰；

- STC 流体波慢度峰；

- 强弱侧能量比；

- 有效窗口能量；

- 姿态置信度；

- CAST-derived quality mask。

模型：

- Logistic Regression；

- Random Forest；

- Gradient Boosting；

- XGBoost/LightGBM，如环境允许。

如果这些简单模型完全无法优于随机，则说明标签、对齐或特征定义仍有问题，不应直接上深度网络。

### 8.3 负对照与正对照实验

除旋转消融、强弱侧加权消融、随机方位打乱和深度错位检验外，还必须增加负对照和正对照实验，用于识别数据泄漏、伪相关和 pipeline 实现错误。

#### 负对照实验

负对照实验使用理论上不应具备窜槽预测能力的输入变量。如果这些变量也能取得较高性能，则说明数据集可能存在深度泄漏、标签偏置或评估流程错误。

推荐负对照包括：

| 负对照输入                         | 检查目的            |
| ----------------------------- | --------------- |
| depth index                   | 检查模型是否仅学习深度趋势   |
| logging order                 | 检查是否存在采集顺序泄漏    |
| random noise                  | 检查评估流程是否异常      |
| smoothed global Zc background | 检查模型是否只学习全局背景   |
| Inc / RelBearing only         | 检查姿态变量是否与标签异常耦合 |
| well_id only                  | 检查是否存在按井泄漏      |

若负对照模型显著优于随机水平，必须暂停主模型训练，重新审查数据划分、标签生成和评估逻辑。

#### 正对照实验

正对照实验用于验证 pipeline 是否能找回已知异常。可通过合成或半合成方式构造已知位置的窜槽模式：

1. 在 CAST 中插入低 Zc 方位条带；
2. 在 XSI 中插入套管波增强或衰减斜率降低；
3. 在特定深度—方位窗口中模拟 STC 套管波峰值异常；
4. 人为设置已知 RelBearing 旋转角，验证方位归一化是否能恢复正确位置。

正对照实验的通过标准是：标签生成、相关性分析、物理 baseline 和可解释性模块均能在允许误差范围内定位到合成异常的位置和方位。

---

## 9. 特征工程方案

### 9.1 XSI 原始波形质量控制

在进行 STFT、STC、APES 或任何深度模型输入构建前，必须先完成 XSI 原始波形质量控制。坏波形、死接收器、饱和波形、极低信噪比 Side 或时间零点漂移都会直接污染物理特征与模型训练。

建议计算并保存以下 QC 字段：

| 字段                         | 说明               |
| -------------------------- | ---------------- |
| `xsi_dead_receiver_mask`   | 失效接收器标记          |
| `xsi_dead_side_mask`       | 长期低能量或异常 Side 标记 |
| `xsi_saturation_ratio`     | 波形饱和比例           |
| `xsi_clipping_ratio`       | 截断比例             |
| `xsi_noise_floor`          | 噪声底              |
| `xsi_snr`                  | 信噪比              |
| `xsi_first_arrival_time`   | 首波到时             |
| `xsi_time_zero_shift`      | 时间零点漂移           |
| `xsi_receiver_consistency` | 13 个接收器之间的一致性    |
| `xsi_polarity_check`       | 极性检查             |
| `xsi_gain_change_flag`     | 增益变化标记           |

重点检查包括：

1. 13 个接收器的首波到时是否随源距呈合理变化；
2. 某个 receiver 或 Side 是否长期异常；
3. 波形是否出现饱和、截断或强尖峰；
4. 高通滤波是否引入边界伪影；
5. 强弱侧能量差异是否来自真实偏心，而不是传感器故障。

若某深度窗或方位窗的 XSI QC 不合格，则该区域不应进入高置信训练样本，可进入 `uncertain_mask` 或仅用于无监督/半监督一致性约束。

### 9.2 XSI 声波特征

`agents.md` 要求 XSI 时间单位统一到 ms，并应用 4 阶 Butterworth 高通滤波，截止频率 1 kHz。

建议 XSI 特征分为三层：

#### A. 原始波形层

保留：

$$
X(d,r,s,t)
$$

其中：

- (d)：深度；

- (r)：13 个接收器；

- (s)：8 个 Side；

- (t)：1024 个时间采样点。

#### B. 可逆时频层

优先使用 STFT/ISTFT，而不是把 CWT 作为唯一主表示。`chagpt_report.md` 已明确建议 STFT/ISTFT 作为可逆时频主方案，STC 作为物理解释主方案。

每个样本保存：

- STFT 幅度；

- STFT 相位；

- ISTFT 重构误差；

- 频带能量；

- 时间窗能量。

STFT 参数必须满足可逆重构条件，并记录 ISTFT 重构误差：

$$
\epsilon_{rec} =
\frac{\|x-\hat{x}\|_2}{\|x\|_2+\epsilon}
$$

若：

$$
\epsilon_{rec}>\tau_{rec}
$$

则该 STFT 配置不得作为可逆主分支特征版本。所有 STFT 参数，包括窗口长度、重叠率、窗函数、边界处理方式和重构误差，必须写入 `feature_version` 元数据。

#### C. 物理解释层

提取：

| 特征              | 物理意义          |
| --------------- | ------------- |
| 套管波时间窗能量        | 胶结差或自由套管时可能增强 |
| 套管波沿 13 接收器衰减斜率 | 胶结好时衰减更明显     |
| 流体波时间窗能量        | 流体通道或耦合异常可能影响 |
| STC 套管波慢度峰      | 套管传播模态        |
| STC 流体波慢度峰      | 流体路径异常        |
| 频散斜率            | 波形模态变化        |
| 强弱侧能量比          | 偏心/井斜/贴壁影响    |
| 有效窗口能量          | 高置信方位响应       |

#### D. 算力下放与离线特征工程

由于 APES（幅度与相位估计）和高分辨率 STC（Slowness-Time Coherence）算法通常涉及高维矩阵运算、慢度网格搜索、阵列相干性计算与大量滑动窗口处理，如果在 PyTorch `DataLoader` 中动态计算，将显著增加 CPU 负担，造成 GPU 等待数据、训练吞吐量下降，甚至导致多卡训练效率严重受限。因此，本项目将 STC、APES 等高成本物理特征从训练循环中剥离，采用“离线预计算 + 固化存储 + 训练期轻量读取”的工程策略。

1. **算法库化与高性能实现**
   
   将核心物理特征提取算法，特别是 STC、APES、套管波能量衰减、慢度峰值搜索等模块，封装为独立的高性能计算库。可选实现路径包括 MATLAB Compiler SDK、C++/OpenMP、CUDA/C++、Numba 或 PyTorch 自定义算子。
   
   在实现选择上，应优先考虑可复现性、部署便利性、服务器环境兼容性和长期维护成本。对于计算密集型模块，可通过 C++ 多线程、OpenMP 或 CUDA 并行化降低 Python 层循环开销，避免 Python 全局解释器锁（GIL）对并行计算的限制。

2. **离线预计算流水线**
   
   在模型训练启动前，先运行独立的物理特征预处理管线。该管线从经过深度对齐、方位归一化和质量控制后的 XSI 波形中批量提取物理特征，生成包括但不限于以下张量：
   
   - 深度—方位—慢度 STC 特征张量；
   - 套管波、流体波、地层波相关时间窗能量；
   - 慢度峰值、峰值能量、峰宽和相干度；
   - APES 幅度与相位估计特征；
   - 强侧、弱侧、有效窗口和全方位鲁棒统计物理特征；
   - 特征质量标记与缺失标记。
   
   训练阶段不再重复执行高成本物理计算，只进行轻量级的数据读取、裁窗、归一化、增强和 batch 组装。

3. **HDF5 / Memmap 固化存储**
   
   离线计算结果应以分块存储方式写入 HDF5、Zarr 或 NumPy memmap 文件。HDF5 主要依赖 chunked storage、压缩、局部读取和缓存机制提升 I/O 效率；如需真正的内存映射能力，可将高频访问特征另存为 `.npy` memmap 或 Zarr 格式。
   
   每个特征文件必须保存完整的元信息，包括：
   
   - `feature_version`；
   - 原始数据版本与 hash；
   - 特征提取代码版本或 git commit；
   - STC / APES 参数；
   - 采样率、时间窗、慢度网格、频带范围；
   - 滤波参数；
   - chunk 配置；
   - 生成时间、运行环境和异常日志。
   
   训练代码只能读取与当前实验配置匹配的特征版本。如果特征参数、数据版本或代码版本不一致，应直接阻断训练，避免出现“旧特征配新模型”的不可追溯问题。

4. **训练期吞吐优化**
   
   训练阶段的数据加载器应只负责从固化特征文件中按需读取局部深度窗和方位窗，并结合标签、质量权重、有效窗口和强弱侧掩码构造输入 batch。对于频繁访问的特征，可配置局部缓存或预取队列，以减少磁盘 I/O 抖动。
   
   该设计的目标是让 GPU 主要消耗在模型前向、反向传播和参数更新上，而不是等待 CPU 动态计算 STC/APES 特征。

---

### 9.3 CAST 超声特征

CAST 不应只用于生成单一标签，也应用于标签置信度和有效窗口估计。

推荐 CAST 特征：

| 特征                       | 用途        |
| ------------------------ | --------- |
| `Zc_min`                 | 检测低阻抗异常   |
| `Zc_median`              | 背景阻抗      |
| `Zc_q10`                 | 局部低阻抗鲁棒指标 |
| `azimuth_gradient_max`   | 动态梯度异常    |
| `low_zc_fraction`        | 方位宽度      |
| `connected_depth_length` | 纵向连续性     |
| `theta_low_zc_center`    | 窜槽中心方位    |
| `theta_strong`           | 偏心/贴壁方向   |
| `eccentricity_score`     | 强弱侧程度     |
| `cast_quality_weight`    | 标签可信度     |

### 9.4 CAST 成像质量控制

CAST 的 Zc 低值并不一定全部代表水泥缺失，也可能由超声回波质量差、方位缺失、套管椭圆度、厚度异常、接箍、工具偏心或井筒流体条件变化引起。因此，在使用 CAST 生成弱标签前，必须先完成 CAST 成像质量控制。

若 CAST 数据中包含原始回波、半径、内径、套管厚度、椭圆度、travel time、回波幅度、增益或质量标记，应优先纳入 QC。建议保存以下字段：

| 字段                            | 说明          |
| ----------------------------- | ----------- |
| `cast_missing_sector_ratio`   | 缺失方位比例      |
| `cast_echo_amplitude_quality` | 回波幅度质量      |
| `cast_travel_time_quality`    | 超声旅行时质量     |
| `cast_radius_variation`       | 半径变化        |
| `cast_ovality`                | 套管椭圆度       |
| `cast_thickness_anomaly`      | 套管厚度异常      |
| `cast_collar_mask`            | 接箍或几何突变掩码   |
| `cast_low_confidence_mask`    | 低置信 CAST 区域 |
| `cast_sector_quality_weight`  | 方位扇区质量权重    |

CAST QC 应进入标签置信度计算：

$$
label_confidence = f(q_{cast}, q_{xsi}, c_{orient}, c_{depth}, c_{continuity})
$$

其中：

- $q_{cast}$：CAST 成像质量；
- $q_{xsi}$：XSI 波形质量；
- $c_{orient}$：姿态置信度；
- $c_{depth}$：深度对齐置信度；
- $c_{continuity}$：纵向连续性置信度。

如果 CAST 某方位 Zc 低，但该方位回波质量差、缺失率高或处于接箍/几何突变区域，应降低标签置信度，而不是直接生成高置信窜槽标签。

---

## 10. 模型路线：先物理 baseline，再双分支融合

### 10.1 严禁一开始端到端

纯端到端模型可能在弱标签、方位错位、类别不平衡和强弱侧伪影下得到虚高分数，但解释性很差。`chagpt_report.md` 已明确指出，纯端到端深网对标签噪声、对齐误差和类不平衡极敏感，不建议作为第一路线。

### 10.2 推荐三阶段模型路线

#### 阶段 1：物理 baseline

目的不是追求最高精度，而是验证数据是否科学。

输入：

- XSI 手工物理特征；

- STC 特征；

- 强弱侧特征；

- 姿态质量特征；

- 有效窗口特征。

输出：

- 窜槽概率；

- 严重程度；

- 方位位置；

- 标签置信度。

#### 阶段 2：XSI 主模型

如果最终目标是“只用 XSI 预测 CAST 标注的窜槽”，则训练输入不能直接包含 CAST Zc，否则会发生标签泄漏。

推荐结构：

$$
XSI\ waveform \rightarrow \text{STFT branch} + \text{STC/physics branch} \rightarrow \text{fusion head}
$$

输出：

- presence；

- severity；

- azimuth sector；

- confidence；

- uncertainty。

##### 10.2.1 补充架构设计：非对称交叉注意力融合 (Asymmetric Cross-Attention)

在处理 XSI 与 CAST 的融合时，不建议简单采用全局统一的特征拼接（Concat）或全连接融合。原因是井斜、仪器偏心、贴壁/悬空状态会导致 XSI 在周向上的信噪比明显不均：强侧通常具有更高的声波耦合质量，而弱侧可能存在能量衰减、到时畸变或低信噪比问题。如果模型忽略这种物理位置差异，容易把偏心伪影、弱侧噪声或 CAST 标签偏差错误学习为窜槽特征。

因此，融合架构应具备“物理位置感知”能力，引入强弱侧掩码、方位质量权重、RelBearing 归一化坐标、有效窗口标记和跨模态一致性约束，采用质量门控的非对称融合策略。

###### 1. 强侧：对称融合

对于贴壁侧或高信噪比方位，XSI 与 CAST 通常都具有较高的信息可靠性。此时可以采用相对对称的融合方式，使两类证据在模型中具有接近的表达权重。

强侧融合可采用：

$$
F_{strong}=Fusion(F^{strong}_{XSI},F^{strong}_{CAST})
$$

其中 `Fusion` 可为 cross-attention、gated fusion 或 feature-level transformer。该分支重点学习 XSI 套管波、流体波、STC 慢度峰与 CAST 低阻抗条带之间的对应关系。

###### 2. 弱侧：质量门控的非对称交叉注意力

对于悬空侧或低信噪比方位，XSI 证据可能较弱，但不能简单丢弃。弱侧融合应采用质量门控的非对称交叉注意力机制：CAST 可作为高分辨率方位证据，引导模型在 XSI 中检索残余的物理响应；但 CAST 不能无条件主导最终判定，必须受到质量权重和跨模态一致性约束。

可表示为：

$$
Q = W_Q F^{weak}_{CAST}
$$

$$
K = W_K F^{weak}_{XSI},\quad V = W_V F^{weak}_{XSI}
$$

$$
A_{weak}=softmax\left(\frac{QK^\top}{\sqrt{d_k}} + M_{quality}\right)V
$$

其中 $M_{quality}$是质量掩码或注意力 bias，用于降低低质量方位、低姿态置信度和低 CAST QC 区域的注意力权重。 $M_{quality}$应作为 attention bias 使用：高质量位置取 0 或较小惩罚，低质量或无效位置取负值，严重无效位置可取 \(-\infty\)，从而在 softmax 中降低其注意力权重。

同时引入门控项：

$$
g=\sigma\left( 
W_g[ 
q_{cast},q_{xsi},c_{orient},c_{side},c_{window},c_{depth}] 
+b_g 
\right)
$$

最终弱侧融合特征为：

$$
F_{weak}=g\cdot A_{weak}+(1-g)\cdot F^{weak}_{XSI}
$$

- g→1：更信任 CAST 引导下从 XSI 中检索出的残余特征；
- g→0：更信任原始 XSI 弱侧特征，或 CAST 质量不足时不采用 CAST 引导。

> g 不是 CAST 置信度本身，而是“是否采用 CAST 引导的跨模态检索结果”的门控权重。

其中：

- $q_{cast}$：CAST 方位质量权重；
- $q_{xsi}$：XSI 该 Side 的信噪比或质量权重；
- $c_{orient}$：RelBearing / Inc 姿态置信度；
- $c_{side}$​：强弱侧置信度；
- $c_{window}$​：是否位于有效窗口。
- $c_{depth}$：局部深度对齐置信度。

该机制允许 CAST 在弱侧提供方位引导，但不会让模型无条件依赖 CAST。若 CAST 与 XSI 证据一致，模型可提高该方位的窜槽置信度；若 CAST 异常但 XSI 物理响应不支持，模型应输出更高不确定性，而不是直接给出高置信窜槽结论。

###### 3. 区分 XSI-only 模型与 XSI+CAST 融合模型

必须明确区分两类模型目标：

| 模型类型            | 输入                        | CAST 的角色                            | 适用场景              |
| --------------- | ------------------------- | ----------------------------------- | ----------------- |
| XSI-only 预测模型   | XSI + 姿态 + 强弱侧权重 + 有效窗口标记 | 仅用于生成弱标签、置信度、teacher signal，不作为直接输入 | 未来希望仅凭声波数据预测窜槽    |
| XSI+CAST 融合解释模型 | XSI + CAST + 姿态 + 质量权重    | 作为独立模态参与融合                          | 辅助专家综合解释、验证多模态一致性 |

如果项目目标是训练一个仅凭 XSI 进行窜槽预测的模型，则 CAST 不应作为模型直接输入，否则会造成标签泄漏。此时 CAST 只能用于弱标签生成、有效窗口估计、标签置信度评估和教师模型蒸馏。

如果项目目标是构建 XSI+CAST 联合解释模型，则可以使用非对称交叉注意力，但最终输出必须同时给出：

- XSI 证据强度；
- CAST 证据强度；
- 跨模态一致性；
- 方位质量权重；
- 不确定性评分；
- 是否可能受偏心或弱侧低信噪比影响。

###### 4. 约束与验收要求

非对称交叉注意力模块必须通过以下消融实验验证：

1. 与简单 Concat 融合对比；
2. 与全局平均融合对比；
3. 与不区分强弱侧的 cross-attention 对比；
4. 随机打乱 CAST 方位后，性能应明显下降；
5. 错误 RelBearing 符号下，方位定位性能应下降；
6. 弱侧高置信预测必须能回溯到 XSI 残余物理响应或明确标记为 CAST 主导、低 XSI 支持。

因此，非对称交叉注意力的目标不是“让 CAST 直接替代 XSI”，而是在强弱侧质量不均的情况下，以物理位置、质量权重和跨模态一致性为约束，实现更稳健的多模态证据融合。

### 10.3 防止 CAST 标签泄漏的模型边界

由于 CAST 是本项目弱标签的主要来源，必须严格区分“XSI-only 预测能力验证”与“XSI+CAST 多模态融合解释”。如果训练和评估时将 CAST Zc 直接作为模型输入，同时又使用 CAST 生成的标签作为监督目标，则模型可能只是学习“用答案预测答案”，导致严重标签泄漏。

推荐将模型目标划分如下：

| 任务              | 允许输入                                   | 禁止输入                   | 目标                     |
| --------------- | -------------------------------------- | ---------------------- | ---------------------- |
| XSI-only 窜槽预测   | XSI、Inc、RelBearing、强弱侧权重、有效窗口标记、XSI QC | CAST Zc、CAST label map | 验证声波数据本身是否具备预测能力       |
| XSI+CAST 融合解释   | XSI、CAST、姿态、质量权重                       | 无                      | 辅助专家综合解释多模态证据          |
| 标签质量审计          | XSI、CAST、QC、弱标签                        | 无                      | 发现标签错误、错位和低置信区域        |
| CAST teacher 蒸馏 | 训练期可使用 CAST teacher signal；推理期只用 XSI   | 推理期 CAST 输入            | 将 CAST 弱标签结构迁移到 XSI 模型 |

因此，XSI+CAST 融合模型的高分不能用来证明 XSI 单独具备窜槽识别能力。真正验证 XSI 声波可预测性的模型必须是 XSI-only，并且在测试阶段不得接触 CAST Zc 或 CAST 派生标签图。

所有实验报告必须明确标注模型类型：

```textile
model_mode = xsi_only | xsi_cast_fusion | label_audit | teacher_distillation
```

---

## 11. 类别不平衡与采样策略

`agents.md` 已明确要求针对窜槽样本稀少采用 Focal Loss 或 Dice Loss，并在 Dataloader 阶段对窜槽深度段进行 Oversampling。

推荐策略：

1. **按井划分训练/验证/测试**，避免同一井相邻深度泄漏；

2. **按窜槽深度窗过采样**，而不是随机点采样；

3. **对严重窜槽、轻微窜槽、疑似窜槽分别采样**；

4. **uncertain 样本不进入硬监督损失**，可用于一致性正则或半监督；

5. **损失函数采用多任务组合**：

$$
L =
L_{presence}^{focal}
+
\lambda_1 L_{severity}^{ordinal}
+
\lambda_2 L_{azimuth}^{BCE+Dice}
+
\lambda_3 L_{confidence}^{Brier}
+
\lambda_4 L_{consistency}
$$

其中：

- $L_{presence}^{focal}$：用于处理窜槽/非窜槽极度不平衡；
- $L_{severity}^{ordinal}$：用于严重程度分级，保留等级顺序关系；
- $L_{azimuth}^{BCE+Dice}$：用于方位掩码分割；
- $L_{confidence}^{Brier}$：用于置信度校准；
- $L_{consistency}$：用于约束 XSI 时频分支、STC 物理分支和跨模态证据的一致性。

---

## 12. 评估体系

原始 `agents.md` 已明确禁止只看 Accuracy，必须关注 Precision、Recall、F1 和 IoU。

升级后的评估体系如下：

| 维度    | 指标                                |
| ----- | --------------------------------- |
| 检出能力  | Recall、AUCPR、F1                   |
| 误报控制  | Precision、False Positive per 100m |
| 方位定位  | Azimuth IoU、角度误差                  |
| 深度定位  | Depth IoU、连续段召回                   |
| 严重程度  | Macro-F1、Ordinal MAE              |
| 校准能力  | ECE、Brier Score、可靠性曲线             |
| 泛化能力  | 按井留一验证、跨井测试                       |
| 抗伪影能力 | 强弱侧消融、RelBearing 错符号检验、随机方位打乱     |
| 可解释性  | 归因是否落在套管波/STC 合理区域                |

上线前必须通过以下门槛：

1. RelBearing 旋转优于未旋转；

2. 强弱侧/有效窗口优于简单平均；

3. 物理 baseline 显著优于随机；

4. 深度错位和方位打乱会显著降低性能；

5. XAI 解释能回到合理时间窗和 STC 模态；

6. 高风险漏判案例经过专家复核。

---

### 12.1 工程成本敏感评估

固井窜槽检测的工程目标不是单纯追求最高 Accuracy 或 F1，而是在控制人工复核工作量的前提下尽量减少高风险漏判。由于漏判严重窜槽的代价通常高于误报，最终阈值选择应采用成本敏感策略。

建议增加以下工程指标：

| 指标                              | 说明                   |
| ------------------------------- | -------------------- |
| `review_segments_per_100m`      | 每 100 m 需要专家复核的疑似段数量 |
| `false_alarm_segments_per_100m` | 每 100 m 误报段数量        |
| `missed_high_severity_channels` | 漏判的高严重度窜槽对象数         |
| `recall_at_fixed_review_budget` | 固定复核预算下的召回率          |
| `precision_at_95_recall`        | 在召回率达到 95% 时的查准率     |
| `high_severity_recall`          | 严重窜槽召回率              |
| `object_level_recall`           | 对象级窜槽召回率             |
| `object_level_precision`        | 对象级窜槽查准率             |

阈值选择不应只采用 F1 最优点，而应根据业务风险定义：

1. 优先保证严重窜槽 Recall；
2. 在满足最低 Recall 门槛后最大化 Precision；
3. 控制每 100 m 推送给专家的疑似段数量；
4. 对低置信预测输出“需复核”，而不是强行给出确定结论。

推荐上线门槛示例：

```text
high_severity_recall >= 0.95
object_level_recall >= 0.90
review_segments_per_100m <= business_budget
ECE <= calibration_threshold
```

---

## 13. 可解释性方案

可解释性不能只给一张 Grad-CAM 热力图。`gemini_report.md` 已提醒，显著性图可能出现捷径学习，热力图如果落在无物理意义区域，反而说明模型有问题。

推荐解释链：

1. **输入层解释**：哪个深度、哪个 Side、哪个时间窗贡献最大；

2. **时频层解释**：哪个频带、哪个毫秒窗口异常；

3. **STC 层解释**：套管波慢度峰、流体波慢度峰是否异常；

4. **强弱侧解释**：异常是否只出现在强侧/弱侧，是否可能是偏心；

5. **CAST 对照解释**：对应方位是否存在低 Zc、动态梯度和纵向连续性；

6. **工程语言输出**：

示例：

> 在 4200 ft 附近，高边顺时针 60°–100° 方位出现疑似严重窜槽。CAST 显示该方位 Zc 明显低于背景，并具有纵向连续低阻抗条带；XSI 在 RelBearing 旋转后对应 Side 的套管波 0.5–1.5 ms 时间窗能量异常增强，13 接收器衰减斜率偏低，STC 套管波慢度峰持续存在。强弱侧分析显示该异常不完全等同于偏心强侧伪影，因此判为高置信窜槽。

---

### 13.1 解释可信度检查

可解释性分析不能只追求热力图“看起来合理”，还必须验证解释是否真正依赖模型学到的有效特征。为防止 Grad-CAM、Saliency、SHAP 或其他归因方法产生误导性解释，应引入解释可信度检查。

推荐检查包括：

| 检查                            | 目的                         |
| ----------------------------- | -------------------------- |
| random label test             | 使用随机标签训练后，解释不应仍然呈现稳定物理模式   |
| model parameter randomization | 随机化模型参数后，归因图应明显变化          |
| input perturbation            | 遮挡高归因时间窗后，预测置信度应下降         |
| side perturbation             | 打乱方位后，方位解释应失效              |
| STC consistency check         | 高归因时间窗应能对应 STC 模态变化        |
| counterfactual masking        | 移除套管波或流体波关键窗口后，模型结论应发生合理变化 |

若模型给出严重窜槽判断，但归因区域主要落在无物理意义的噪声段、缺失段、泥浆直达波无关区域或低质量 CAST 扇区，则该解释不应被接受。此类样本应进入错误案例库，用于后续数据清洗、标签审计或模型重训。

最终解释报告必须同时输出：

1. 模型预测结果；
2. 预测置信度；
3. 不确定性评分；
4. XSI 关键时间窗；
5. STC 关键慢度峰；
6. CAST 对应低 Zc 方位；
7. 强弱侧与偏心解释；
8. 是否通过解释可信度检查。

---

## 14. 更新后的 Agent/流水线建议

在原有 agents 基础上，建议新增一个核心智能体：

### `alignment-physics-validation-agent`

**职责：**

1. 执行 RelBearing 高边坐标归一化；

2. 平滑低井斜段 RelBearing 奇点；

3. 估计强侧/弱侧方向；

4. 生成有效窗口；

5. 计算强弱侧权重；

6. 做 XSI–CAST 相关性实验；

7. 输出是否允许进入训练阶段的 go/no-go 结论。

**输出物：**

| 输出                        | 内容               |
| ------------------------- | ---------------- |
| `aligned_coordinates.h5`  | 高边坐标化后的 XSI/CAST |
| `side_quality_weights.h5` | XSI 8 Side 权重    |
| `effective_windows.h5`    | 每个深度的有效窗口        |
| `correlation_report.md`   | 相关性实验报告          |
| `rotation_ablation.png`   | 旋转前后对比           |
| `strong_weak_polar.png`   | 强弱侧极坐标图          |
| `go_no_go.json`           | 是否进入模型训练         |

如果该 agent 输出 no-go，则不允许进入主模型训练。

### 14.1 新增 QC 与数据契约 Agent

除 `alignment-physics-validation-agent` 外，建议新增两个基础 Agent，确保数据质量和数据契约在所有训练前被强制执行。

#### `data-contract-agent`

**职责：**

1. 校验 HDF5 / Zarr / memmap 文件结构；
2. 检查维度顺序是否符合 `depth → receiver → side → time`；
3. 检查方位角范围是否为 `[0, 360)`；
4. 检查 `data_version`、`label_version`、`feature_version` 是否存在；
5. 校验特征参数、标签参数和训练配置是否匹配；
6. 若 schema 不一致，直接阻断训练。

**输出物：**

| 输出                     | 内容             |
| ---------------------- | -------------- |
| `schema_report.json`   | 数据结构校验结果       |
| `version_check.json`   | 数据/标签/特征版本匹配情况 |
| `blocking_errors.json` | 阻断训练的问题列表      |

#### `qc-agent`

**职责：**

1. 执行 XSI 原始波形 QC；
2. 执行 CAST 成像质量 QC；
3. 生成死接收器、低质量 Side、缺失方位、接箍/几何异常等 mask；
4. 计算 XSI 和 CAST 质量权重；
5. 将 QC 结果写入 `quality_flags` 和 `label_confidence`。

**输出物：**

| 输出                             | 内容          |
| ------------------------------ | ----------- |
| `xsi_qc_report.md`             | XSI 波形质量报告  |
| `cast_qc_report.md`            | CAST 成像质量报告 |
| `quality_masks.h5`             | 所有质量掩码      |
| `qc_summary.png`               | QC 可视化总览    |
| `low_confidence_intervals.csv` | 低置信井段列表     |

如果 `data-contract-agent` 或 `qc-agent` 输出 blocking error，则后续标签生成、特征提取和模型训练均不得启动。

---

## 15. 最终推荐执行顺序

| 阶段      | 名称          | 核心任务                                                 | 是否可跳过 |
| ------- | ----------- | ---------------------------------------------------- | ----- |
| Phase 0 | 数据契约冻结      | 确认 XSI Side 方位、CAST 方位、RelBearing 符号、深度轴、HDF5 schema | 不可跳过  |
| Phase 1 | 高边坐标归一化     | 所有 XSI/CAST 统一到相对高边坐标系                               | 不可跳过  |
| Phase 2 | 强弱侧与有效窗口    | 估计偏心方向、质量权重、有效窗口                                     | 不可跳过  |
| Phase 3 | 弱标签生成       | Zc 阈值 + 方位梯度 + 连续性 + 置信度                             | 不可跳过  |
| Phase 4 | 跨模态相关性验证    | 旋转、加权、窗口、随机打乱、错位消融                                   | 不可跳过  |
| Phase 5 | 物理 baseline | STC/能量/衰减/强弱侧特征 + 传统模型                               | 不可跳过  |
| Phase 6 | 深度模型        | STFT 可逆支路 + STC 物理支路 + 多任务头                          | 可迭代   |
| Phase 7 | XAI 与专家审查   | 回到时间轴、STC、CAST 方位图和工程解释                              | 不可跳过  |
| Phase 8 | 发布门槛        | AUCPR、Recall、IoU、ECE、错位/打乱检验                         | 不可跳过  |

### 15.1 最小可执行里程碑 MVP

为避免项目一开始陷入复杂模型实现，应采用分阶段 MVP 路线。每个阶段都有明确的交付物、通过标准和禁止事项。只有当前阶段通过验收后，才允许进入下一阶段。

| MVP 阶段              | 必须完成                                      | 明确不做                 |
| ------------------- | ----------------------------------------- | -------------------- |
| MVP-1 数据契约与 QC      | HDF5 schema、XSI QC、CAST QC、版本字段、质量掩码      | 不训练深度模型              |
| MVP-2 对齐与方位归一化      | RelBearing 双符号验证、低井斜 mask、高边坐标图、深度 lag 估计 | 不生成最终训练集             |
| MVP-3 弱标签审计         | 自适应基线、方位梯度、纵向连续性、uncertain mask、对象级标签     | 不做复杂神经网络             |
| MVP-4 跨模态相关性验证      | 旋转消融、强弱侧消融、错位检验、随机方位打乱、负/正对照              | 不追求最高精度              |
| MVP-5 物理 baseline   | STC、能量、衰减、强弱侧特征 + 传统模型                    | 不上大模型                |
| MVP-6 XSI-only 深度模型 | STFT 可逆支路 + STC 物理支路 + 多任务头               | 不把 CAST 作为输入         |
| MVP-7 XSI+CAST 融合解释 | 非对称交叉注意力、跨模态一致性、不确定性输出                    | 不用融合模型证明 XSI-only 能力 |
| MVP-8 专家复核与发布门槛     | 错误案例库、解释报告、对象级评估、工程成本指标                   | 不自动替代人工决策            |

每个 MVP 阶段必须输出：

1. 可复现实验配置；
2. 中间数据文件；
3. 可视化图；
4. 指标报告；
5. 错误案例；
6. go/no-go 结论。

若任一阶段未通过 go/no-go 门槛，则不得用后续复杂模型掩盖前序数据问题。

---

## 16. 最终结论

两份报告都提供了有价值的基础，但都缺少你补充的关键环节：**井斜/偏心导致的强弱侧非对称必须被显式建模**。在这个项目中，RelBearing 旋转不是一个普通预处理步骤，而是所有跨模态匹配的前置条件；CAST 180 方位也不只是标签来源，而应作为识别偏心方向、有效窗口和标签置信度的核心工具。

最终建议路线是：

> **高边坐标归一化 → 强弱侧质量建模 → CAST 有效窗口识别 → 弱标签置信度生成 → XSI–CAST 相关性实验 → 物理 baseline → STFT/STC 双分支模型 → 可解释性与专家审查。**

这样做的优势是：模型训练前先证明数据关系是物理上成立的；训练中避免把强弱侧伪影、方位错位和弱标签噪声当成窜槽特征；训练后能够把模型判断回溯到 CAST 方位低阻抗、XSI 毫秒级波形、STC 慢度峰和工程解释上。

这才是一个科学、可审计、可解释、可落地的固井窜槽智能检测方案。
