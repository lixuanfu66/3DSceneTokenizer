# Qwen 3D Token 任务体系设计

## 目标

后续路线是：先用 Octree Node VQVAE 把自驾场景编码成离散 3D token，再把场景 token 序列化后输入 Qwen3.5 4B，让语言模型在同一套 3D 表示上学习：

- 3D 空间理解
- 3D 结构生成
- 3D token 与文字对齐
- 驾驶场景常识与经验
- 风险判断、行为预测与驾驶决策规划

核心原则是：不要把 Qwen 只训练成“读 3D token 后回答文字”的模型，而要让它同时具备读、写、补全、编辑、预测 3D token 的能力。这样 3D 编码本身会被更强的下游任务反向筛选和强化。

## 模块解耦原则

本文件只设计 `3D VLM` 侧任务，不把 Qwen 训练逻辑写进 `3D VQVAE / Tokenizer` 内部。

系统边界应保持如下关系：

```text
3D data -> 3D Tokenizer -> versioned scene tokens -> 3D VLM -> text / JSON / 3D tokens
              ^                                           |
              |                                           v
        optional decoder <-------- generated/edited tokens
```

### 3D VQVAE / Tokenizer

职责：

- 学习或规则化 3D 压缩表示。
- 维护 octree/统一树结构。
- 维护 VQ codebook 与 `main_code`。
- 导出版本化 scene token。
- 可选把 token 解码回 UDF、occupancy、surface、RGB。

不负责：

- Qwen instruction 数据格式。
- 驾驶决策 loss。
- 语言模板。
- VLM prompt 或 LoRA 训练。

### 3D VLM / Qwen

职责：

- 消费 tokenizer 的版本化 token。
- 学习 3D 与文本对齐。
- 做空间理解、3D 生成、风险判断和驾驶规划。
- 生成合法的 token 编辑或补全结果。

不负责：

- VQVAE codebook 训练。
- 点云到 octree 的构建细节。
- UDF/RGB decoder 的内部结构。

### 稳定接口

第一版稳定接口是文件级 API：

- `scene_tokens_compact_latest.json`
- `scene_tokens_llm_sequence_latest.json`
- `scene_token_bundle.json`

后续如果需要更高效训练，可以再增加 binary/Arrow/WebDataset 形式，但语义字段必须与 JSON schema 保持一致。

这个边界允许两侧互相替换：

- 更强 tokenizer 可以直接喂给同一个 Qwen 任务体系。
- 更强 VLM 可以直接消费同一批 token 数据。
- Qwen 任务失败可以反向诊断 tokenizer，但不能让 VLM 代码直接侵入 tokenizer 内部。

## 输入输出统一形式

### 输入

默认输入使用当前 pipeline 的 LLM 序列出口：

- `scene_tokens_llm_sequence_latest.json`

序列由两层组成：

- `INSTANCE_HEADER`：实例语义、位置、朝向、尺寸、node 数、code scheme。
- `INSTANCE_NODE_V2`：`split_flag / child_mask / main_code`。

训练 Qwen 时建议把结构字段和 code 字段都变成显式 special tokens：

```text
<SCENE_START>
<EGO x=0 y=0 yaw=0>
<TASK navigation_or_question>
<INST semantic=car cx=... cy=... cz=... yaw=... size=... nodes=...>
<NODE sf=7 cm=... code=VQ_01234>
...
<SCENE_END>
```

第一版可以先用文本化离散 token；当任务跑通后，再考虑把 `main_code` 接到可学习 embedding 表，保留结构字段文本化。

### 输出

输出分成四类：

- 自然语言：caption、解释、问答。
- 结构化 JSON：空间关系、风险列表、决策计划。
- 3D token：补全、预测、编辑后的 `INSTANCE_HEADER + INSTANCE_NODE`。
- 混合输出：先给驾驶决策，再给支撑该决策的关键 3D evidence。

## 任务族 1：3D Token 与文字对齐

目标：让 Qwen 知道 3D token 对应什么物体、几何、空间关系和驾驶语义。

### 1.1 场景 caption

输入：完整 scene token。

输出：从自车视角描述场景。

示例目标：

```text
前方约 18 米有一辆车，左侧为人行道，右前方有交通灯，当前车道前方无明显障碍。
```

自动构造方式：

- 从 scene layer 读取实例类别、位置、距离、相对方位。
- 用模板生成 caption。
- 后续可用更强模型或人工规则改写成自然语言。

### 1.2 实例 grounding

输入：scene token + 文本指令。

问题：

```text
找出前方最近的车辆。
```

输出：

```json
{"instance_id": 12, "semantic": "car", "distance_m": 14.6, "relative_position": "front"}
```

能力点：

- 文字短语和 3D 实例对齐。
- 相对方位、距离、类别检索。

### 1.3 几何属性问答

问题：

- 哪个物体最高？
- 前方车辆大概多宽？
- 交通灯杆是细长结构还是块状结构？
- 该障碍物是否可能侵入行驶空间？

监督来源：

- `box_size_xyz`
- node 分布
- split_flag 形态
- 语义类别

## 任务族 2：空间理解

目标：从 token 中学习 3D 空间结构，而不是只记语义标签。

### 2.1 相对位置关系

任务：

- 判断 A 在 B 的前/后/左/右。
- 判断 A 是否位于自车当前车道内。
- 判断某个目标是否靠近路沿、车道线、人行横道。

输出：

```json
{
  "subject": "pedestrian_3",
  "relation": "left_front_of",
  "object": "ego",
  "distance_m": 9.2
}
```

### 2.2 距离与尺度估计

任务：

- 最近车辆距离是多少？
- 前方 30 米内有几个动态目标？
- 某障碍物高度是否高于可安全碾压阈值？

评估：

- 数值误差允许一定容差，例如距离 `±1m`、角度 `±10°`。

### 2.3 可通行空间与占用理解

任务：

- 判断自车前方是否有 free corridor。
- 判断左转区域是否被动态目标占用。
- 找出当前最可能碰撞的区域。

监督来源：

- Bench2Drive/CARLA 语义与实例。
- 由道路、车道线、动态目标 bbox 自动生成 BEV occupancy/free-space 标签。

### 2.4 遮挡与不可见区域推理

任务：

- 大车后方是否存在潜在遮挡风险？
- 路口右侧视野是否被建筑或车辆遮挡？
- 当前看不见但需要减速观察的区域在哪里？

这类任务会迫使模型学习驾驶经验，而不仅是几何重建。

## 任务族 3：3D 生成

目标：让 Qwen 不只消费 3D token，也能生成或修改 3D token。

### 3.1 Node token 补全

输入：

- 随机 mask 一部分 `main_code`
- 或 mask 某个实例的局部 node 子树

输出：

- 被 mask 的 `main_code` 或完整子树。

训练价值：

- 强化 codebook 的局部语义。
- 检验 Qwen 是否理解同一物体内部的几何上下文。

### 3.2 实例补全

输入：

- partial observation 的实例 token
- 类别提示，例如 `car / pedestrian / traffic_cone`

输出：

- 更完整的实例 node token。

训练价值：

- 学习形状先验。
- 对自驾场景中的遮挡物体特别重要。

### 3.3 场景 token infilling

输入：

- 删除某个区域内的实例或地面局部 token。

输出：

- 合理补全该区域的 scene layer 与 instance layer。

示例：

```text
补全路口右前方被遮挡区域中可能存在的交通参与者。
```

注意：

- 这不是要求凭空幻想唯一真值，而是学习合理假设与不确定性表达。
- 输出应允许 `possible_objects` 和置信度。

### 3.4 未来场景预测

输入：

- 当前帧或短历史 scene token。

输出：

- `t+1s / t+2s` 的 scene layer。
- 第一版先预测实例 pose，不预测完整 node 几何。

训练价值：

- 学习动态物体行为。
- 与驾驶规划直接相关。

### 3.5 指令式 3D 场景编辑

输入：

```text
把前方车辆向右移动 1 米，并保持其形状不变。
```

输出：

- 修改后的 `INSTANCE_HEADER`。
- instance node 保持不变。

能力点：

- 区分 scene layer 的 where 与 instance layer 的 what。
- 对齐语言控制和 3D token 操作。

## 任务族 4：驾驶常识与经验

目标：让模型把 3D 空间与驾驶语义连起来。

### 4.1 风险识别

任务：

- 找出当前最危险的 3 个对象。
- 判断是否存在鬼探头风险。
- 判断是否需要减速。

输出：

```json
{
  "risk_level": "high",
  "risks": [
    {
      "type": "occluded_pedestrian_risk",
      "location": "right_front",
      "evidence": ["large_vehicle_blocks_view", "crosswalk_nearby"]
    }
  ]
}
```

### 4.2 交通规则与场景常识

任务：

- 红灯时是否可以直行？
- 接近人行横道且右侧有遮挡时应如何处理？
- 前方施工锥桶占据车道时是否应变道？

训练数据来源：

- 规则模板。
- Bench2Drive 场景标签。
- 人工构造的少量高质量 instruction 数据。

### 4.3 行为预测

任务：

- 预测前车是否可能减速。
- 预测行人是否可能进入车道。
- 预测旁车是否可能切入。

监督来源：

- 多帧轨迹。
- 交通灯、车道线、距离、相对速度。

## 任务族 5：驾驶决策与规划

目标：让模型输出可验证的驾驶动作，而不是只给解释。

### 5.1 高层决策

输出动作空间：

- `keep_lane`
- `slow_down`
- `stop`
- `yield`
- `lane_change_left`
- `lane_change_right`
- `turn_left`
- `turn_right`

输出格式：

```json
{
  "decision": "slow_down",
  "target_speed_mps": 3.0,
  "reason": "右前方人行横道附近存在遮挡行人风险",
  "evidence_instances": [5, 9, 12]
}
```

### 5.2 轨迹规划草案

输出：

- 未来 2-4 秒 ego waypoints。
- 每个 waypoint 包含 `x, y, speed`。

第一版不要求 Qwen 直接输出低层控制，只输出可检查的中高层规划。

### 5.3 反事实决策

任务：

- 如果前车突然刹停，应如何规划？
- 如果行人进入车道，应如何规划？
- 如果右侧车辆切入，应如何规划？

这类任务对常识和经验学习很关键，也能让 3D 表示覆盖危险边界条件。

## 推荐训练阶段

## 生成任务验证路线

生成任务先分成两大类：

- 静态细节生成任务：检验模型是否理解物体几何、外观、部件语义和驾驶相关细节。
- 动态未来生成任务：检验模型是否理解场景随时间变化、动态物体意图和未来风险。

第一版不把所有生成任务同时铺开，而是按“从局部 token 语法到驾驶相关动态预测”的顺序验证。每一步都要明确它验证什么能力、失败时反向诊断哪部分编码。

### 验证顺序总览

| 顺序 | 小类任务 | 所属大类 | 主要验证能力 | 失败时优先检查 |
|------|----------|----------|--------------|----------------|
| 1 | Masked local node infilling | 静态细节生成 | node code 是否承载局部几何/外观语义 | VQ codebook、node 粒度、tree grammar |
| 2 | Attribute-conditioned local editing | 静态细节生成 | 车门、车灯、轮胎角、交通灯等属性是否与局部 token 对齐 | 局部部件 token、RGB/外观编码、属性标注 |
| 3 | Object footprint / obstacle contour generation | 静态细节生成 | 障碍物轮廓、BEV footprint、可碰撞边界是否可恢复 | 几何 decoder、bbox/footprint schema、LOD |
| 4 | Future scene layer prediction | 动态未来生成 | 动态物体未来 pose、速度、yaw 是否可预测 | scene layer、轨迹字段、动态历史 |
| 5 | Multi-modal risk future generation | 动态未来生成 | 多可能未来、遮挡风险、交互风险是否可表达 | 不确定性建模、风险标签、交通常识 |

### 1. Masked local node infilling

输入：

- 完整实例或场景 token。
- 随机 mask 某些 `INSTANCE_NODE_V2.main_code`。
- 可选 mask 某个局部子树。

输出：

- 被 mask 的 `main_code`。
- 或完整补全的局部 node 子树。

验证目标：

- 模型是否学到同一物体内部的局部几何上下文。
- `main_code` 是否不仅是压缩编号，而具有可预测的局部语义。
- 统一树结构是否足够让 VLM 通过上下文恢复局部几何/外观。

为什么排第一：

- 标签最容易自动构造，不需要额外人工标注。
- 直接检查 3D token 语法与 codebook 是否可被 Qwen 学习。
- 是后续属性编辑、轮廓生成和未来预测的底座。

主要指标：

- masked code accuracy / top-k accuracy。
- tree grammar validity。
- decoder 重建后的局部 UDF/Chamfer/F-score。
- 按类别、node depth、node size 分组统计。

### 2. Attribute-conditioned local editing

输入：

```text
<car instance tokens> + condition: brake_light=on
<car instance tokens> + condition: left_door=open
<traffic_light tokens> + condition: state=red
```

输出：

- 编辑后的局部 node token。
- 或局部属性相关 node ids / bbox / token delta。

重点属性：

- `door=open/closed`
- `headlight=on/off`
- `brake_light=on/off`
- `turn_signal=left/right/off`
- `wheel_angle=left/right/straight`
- `traffic_light=red/yellow/green`
- `barrier=fallen/upright`
- `cone=standing/tilted`

验证目标：

- 模型是否能把驾驶语义属性定位到局部 3D 区域。
- 模型是否能区分外观变化、局部几何变化和全局 pose 不变。
- 车灯、车门、轮胎角度等细节是否能进入 3D token 表示。

为什么排第二：

- 它建立在 node infilling 的局部 token 能力上。
- 它比整物体生成更聚焦，能直接检验细粒度语义。
- 它连接物体外观细节与驾驶含义，例如刹车灯亮意味着前车可能减速。

主要指标：

- 属性分类/识别准确率。
- 编辑区域 localization accuracy。
- 非目标区域 token 保持率。
- 解码后局部几何/外观变化是否符合属性。

### 3. Object footprint / obstacle contour generation

输入：

- 稀疏或 partial 的障碍物 token。
- 可选类别提示，例如 `fallen_barrier / cone / debris / unknown_obstacle`。

输出：

- 完整或更稳健的 obstacle contour。
- BEV footprint polygon。
- 可碰撞 bbox / occupied cells。
- 可选补全后的 instance node tokens。

验证目标：

- 模型是否理解障碍物几何轮廓，而不是只识别类别。
- 生成结果是否能服务碰撞判断和可通行空间判断。
- 对异形障碍物、倒伏物体、施工物体是否能给出保守边界。

为什么排第三：

- 它从局部部件细节走向驾驶安全几何。
- 输出可以用 BEV footprint 简化验证，先不要求完整 mesh 高保真。
- 它直接服务自动驾驶中的可通行空间与碰撞风险。

主要指标：

- footprint IoU。
- contour Chamfer / Hausdorff distance。
- collision boundary recall。
- conservative safety recall，即宁可略大也不能漏掉风险边界。

### 4. Future scene layer prediction

输入：

- 当前帧 scene token。
- 或短历史 scene token。

输出：

```json
{
  "t_plus_1s": [
    {"instance_id": 12, "center_xyz": [..], "yaw": 0.1, "velocity": [..]}
  ],
  "t_plus_2s": [...]
}
```

第一版只预测 scene layer：

- `center_xyz`
- `yaw`
- `velocity`
- `valid/existence`

暂不要求预测完整 instance node 几何。

验证目标：

- 模型是否能从当前空间布局和短历史中预测动态物体未来。
- 模型是否学到车辆、行人、自行车等不同交通参与者的运动先验。
- 预测是否能为驾驶规划提供可用输入。

为什么排第四：

- 在静态几何、属性、轮廓任务稳定后，再进入时间维度。
- 先预测 scene layer，避免一开始要求完整未来几何生成。
- 这是从 3D token 走向驾驶规划的关键桥。

主要指标：

- ADE / FDE。
- yaw error。
- velocity error。
- collision prediction recall。
- 按类别和距离分桶的预测误差。

### 5. Multi-modal risk future generation

输入：

- 当前或短历史 scene token。
- 可选 ego action 条件，例如 `ego_keep_lane / ego_slow_down / ego_change_left`。
- 可选遮挡区域提示。

输出：

```json
{
  "futures": [
    {
      "mode": "pedestrian_waits",
      "probability": 0.55,
      "risk_level": "low",
      "future_scene_layer": [...]
    },
    {
      "mode": "pedestrian_crosses",
      "probability": 0.35,
      "risk_level": "high",
      "future_scene_layer": [...]
    }
  ]
}
```

验证目标：

- 模型是否知道未来不是单一答案。
- 模型是否能表达遮挡风险、交互风险和低概率高风险事件。
- 模型是否能把生成的未来与风险判断、驾驶决策连起来。

为什么排第五：

- 它依赖前四类能力：token 语法、局部语义、障碍边界、未来 pose。
- 它最接近驾驶常识与决策规划，但监督也最难。
- 第一版应作为高价值挑战任务，而不是最先启动的任务。

主要指标：

- minADE / minFDE。
- risk recall。
- mode diversity。
- probability calibration。
- ego-action conditioned consistency。

### 顺序与依赖关系

```text
Masked node infilling
  -> Attribute-conditioned editing
  -> Obstacle contour / footprint generation
  -> Future scene layer prediction
  -> Multi-modal risk future generation
```

这条顺序背后的逻辑是：

- 先验证 3D token 本身能被读写。
- 再验证局部 token 能承载驾驶相关细节语义。
- 再验证几何轮廓能支持安全边界。
- 再验证动态物体未来变换。
- 最后验证多可能未来、遮挡风险和驾驶交互。

### 两大类任务的关系

静态细节生成解决“物体是什么、长什么样、哪些细节有驾驶语义”：

- masked local node infilling
- attribute-conditioned local editing
- object footprint / obstacle contour generation

动态未来生成解决“场景接下来怎么变、哪些变化影响风险和规划”：

- future scene layer prediction
- multi-modal risk future generation

这五个任务共同构成第一版生成式 3D 理解验证链路。

### 单物体起步策略

第一阶段先从单个物体做静态细节生成，而不是直接进入完整驾驶场景。

原因：

- object-level 数据更容易从 Objaverse++ 和程序化合成中构造。
- 局部 mask、属性编辑、footprint GT 都更可控。
- 可以先诊断 VQVAE `main_code` 是否具备局部几何、外观和语义可预测性。
- 不会被场景布局、动态历史、驾驶规则等因素干扰。

对应实验设计详见：

- `docs/object_level_static_generation_experiments.md`

### Stage A：3D token 读写预训练

目标：

- 让 Qwen 熟悉 3D token 语法。

任务：

- scene token next-token prediction
- masked node code infilling
- tree structure consistency prediction

### Stage B：3D-token/text 对齐

目标：

- 让 token 与语言中的物体、位置、几何、关系对齐。

任务：

- caption
- grounding
- spatial QA
- geometry QA

### Stage C：生成式空间理解

目标：

- 通过生成任务提升 3D 编码和空间推理。

任务：

- node 补全
- instance partial-to-complete
- scene infilling
- future scene prediction

### Stage D：驾驶任务微调

目标：

- 学习驾驶常识、风险判断和规划。

任务：

- risk QA
- behavior prediction
- high-level decision
- waypoint planning
- counterfactual planning

### Stage E：联合训练与偏好对齐

目标：

- 让模型在理解、生成、决策之间保持一致。

任务：

- 同一场景同时输出 caption、risk、decision、future tokens。
- 对安全决策做 preference tuning。
- 对生成结果做 VQVAE decode 后的几何一致性过滤。

## 数据构建策略

### 自动生成标签

可从现有 3D token 与 Bench2Drive/CARLA 数据自动得到：

- 实例类别
- 实例距离
- 相对方位
- bbox 尺寸
- 车道占用
- 近距离风险
- 遮挡关系近似
- 短时轨迹

### 模板生成 instruction

先用模板保证标签正确：

```text
问题：前方 20 米内有几辆车？
答案：{"count": 3, "instances": [12, 15, 18]}
```

模板任务足够多后，再做语言改写，增强自然语言多样性。

### 人工高质量小集

需要人工或强模型辅助构建的小集：

- 复杂遮挡风险
- 驾驶经验解释
- 反事实规划
- 多目标权衡
- 违反常识的生成样本筛除

## 关键评估

### 3D 编码质量

- VQ codebook usage
- token length
- reconstruction UDF/Chamfer/F-score
- partial-to-complete consistency

### 语言对齐

- grounding accuracy
- spatial QA accuracy
- distance/angle numeric error
- caption factuality

### 生成能力

- token validity
- tree grammar validity
- decoded geometry quality
- scene relation consistency
- uncertainty calibration

### 驾驶能力

- collision rate
- off-road rate
- red-light violation
- route progress
- comfort
- risk recall
- decision explanation faithfulness

## 对 3D 编码的反向促进

这些任务会暴露当前 3D tokenizer 的不足：

- 如果 spatial QA 做不好，说明 scene header 的位置、朝向、尺度量化或序列组织有问题。
- 如果 shape semantics probe 做不好，说明 node code 太局部，缺少 instance-level 汇聚 token。
- 如果 partial-to-complete 做不好，说明 VQ code 没学到类别级形状先验。
- 如果 planning 做不好，说明 token 缺少车道、可通行区域、动态历史或交通灯状态。
- 如果 Qwen 生成 token 经常非法，说明树语法需要更强约束或分层生成器。

因此 Qwen 任务不是下游展示，而是 3D 编码迭代的诊断器。

## 第一版落地建议

优先做一个最小闭环：

1. 用 `scene_tokens_llm_sequence_latest.json` 构造 token 文本序列。
2. 给 Qwen 增加结构 special tokens 和 `VQ_xxxxx` code tokens。
3. 先训练 LoRA，不全量微调 4B。
4. 先做四个任务：
   - scene caption
   - spatial QA
   - masked node code infilling
   - high-level driving decision
5. 每个样本同时保存：
   - 输入 scene token
   - 输出 text/JSON/token
   - 自动验证脚本需要的 label
6. 通过评估结果反推是否需要改 VQVAE codebook、scene header、LOD 或时序字段。
