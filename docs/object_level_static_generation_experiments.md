# 单物体几何/外观/语义细节生成任务设计

## 目标

第一阶段先不从完整自动驾驶场景开始，而是从单个物体研究 3D token 的生成式理解能力。这样数据更容易构造，GT 更清晰，也能更直接诊断 VQVAE tokenizer 的局部 code 是否真的承载了物体几何、外观和语义细节。

本阶段聚焦“任务一：物体几何/外观/语义细节相关生成任务”，对应生成任务验证路线中的前三个静态细节任务：

1. `masked local node infilling`
2. `attribute-conditioned local editing`
3. `object footprint / obstacle contour generation`

## 数据来源

### A. Objaverse++ 真实物体数据

使用当前已构建的数据：

- `data/objaverse_pp_task_relevant`
- `data/objaverse_pp_task_relevant_val`

优先类别：

- 车辆：`car_(automobile)`、`bus_(vehicle)`、`ambulance`、`pickup_truck`、`motorcycle`、`bicycle`
- 交通设施：`traffic_light`、`stop_sign`、`street_sign`、`signboard`
- 障碍/道路相关：`cone` 后续补充、`barrier` 后续补充、`garbage_truck`、`trailer_truck`
- 室内/机器人泛化：`chair`、`desk`、`cabinet`、`sofa`、`bed`

用途：

- 真实几何分布。
- 类别级 shape semantics。
- object-level VQVAE/tokenizer 训练和验证。

限制：

- 车门开关、车灯亮灭、轮胎角度等属性大多没有显式标注。
- 纹理/材质可能不稳定。
- 多数资产不是成对状态变化数据。

### B. 程序化合成可控物体

为了研究细粒度属性，必须补一批程序化数据。第一版不追求真实渲染，追求可控几何和标签。

建议生成：

- 简化车辆：车体、四轮、车灯、车门板、后视镜。
- 简化交通灯：灯杆、灯箱、红黄绿灯面。
- 简化障碍物：路锥、倒伏路锥、长方体障碍、圆柱障碍、倒伏标志牌。
- 简化门/板状物：打开/关闭角度可控。

可控属性：

- `door_state`: `closed / left_open / right_open`
- `door_angle_deg`: `0 / 30 / 60 / 90`
- `headlight_state`: `off / on`
- `brake_light_state`: `off / on`
- `turn_signal_state`: `off / left / right`
- `wheel_angle_deg`: `-30 / 0 / 30`
- `traffic_light_state`: `red / yellow / green`
- `obstacle_pose`: `upright / tilted / fallen`

用途：

- 属性条件编辑。
- 局部部件定位。
- 反事实编辑 GT。
- 验证 VLM 是否理解“局部变化，整体不变”。

### C. Partial observation 数据

从完整物体构造 partial 输入：

- 随机视角裁剪。
- 按相机方向保留可见半边。
- 按 bbox 区域删除局部部件。
- 按 octree node mask 删除局部子树。
- 加入点云稀疏化与噪声。

用途：

- local node infilling。
- partial-to-complete object token 补全。
- 障碍物轮廓恢复。

## 标准样本格式

每个 object-level 样本建议包含以下文件：

```text
sample_id/
  object.ply
  object_tokens.json
  object_tokens_llm_sequence.json
  object_meta.json
  masks/
    mask_000.json
  tasks/
    task_000.json
```

### object_meta.json

```json
{
  "sample_id": "car_000123",
  "category": "car",
  "source": "objaverse_pp|procedural",
  "split": "train|val|test",
  "attributes": {
    "door_state": "closed",
    "headlight_state": "off",
    "brake_light_state": "on",
    "wheel_angle_deg": 0
  },
  "bbox": {
    "center_xyz": [0.0, 0.0, 0.0],
    "size_xyz": [4.5, 1.8, 1.6],
    "yaw": 0.0
  }
}
```

### task_000.json

```json
{
  "task_type": "masked_local_node_infilling",
  "input_tokens": "object_tokens_llm_sequence.json",
  "mask": {
    "node_ids": [12, 13, 14],
    "mask_mode": "main_code"
  },
  "target": {
    "node_tokens": [
      {"node_id": 12, "main_code": 345},
      {"node_id": 13, "main_code": 812}
    ]
  },
  "eval": {
    "metrics": ["top1_code_acc", "top5_code_acc", "tree_validity", "decoded_chamfer"]
  }
}
```

## 实验 1：Masked Local Node Infilling

### 目标

验证 Qwen/3D VLM 是否能根据同一物体的上下文补全被 mask 的局部 `main_code` 或局部子树。

### 数据构造

输入数据：

- Objaverse++ object-level tokens。
- 程序化物体 tokens。

mask 策略：

- `random_node_mask`: 随机 mask 15%-30% node code。
- `subtree_mask`: mask 一个局部子树。
- `part_region_mask`: 对程序化物体 mask 已知部件区域，如车灯、车门、轮胎。
- `depth_aware_mask`: 更高概率 mask 细层 node，检验细节恢复。

训练输入：

```text
<OBJECT_START category=car>
<HEADER size=...>
<NODE sf=7 cm=... code=VQ_0123>
<NODE sf=7 cm=... code=<MASK>>
...
<OBJECT_END>
```

训练输出：

```text
<FILL node_id=15 code=VQ_0456>
<FILL node_id=16 code=VQ_0789>
```

### 实验设置

Baseline：

- `frequency baseline`: 同类别同 depth 最常见 code。
- `nearest neighbor`: 用未 mask token 检索相似 object 后填充。
- `Qwen-LoRA`: 文本化 token 输入输出。

评估：

- `top1 / top5 code accuracy`
- `subtree exact match`
- `tree grammar validity`
- `decoded local Chamfer / F-score`
- `per category / per depth / per part` 分桶结果

### 预期诊断

- code accuracy 低：VQ code 可能缺少可预测语义，或 token 序列上下文不足。
- tree validity 低：需要 grammar-constrained decoding。
- 几何指标差但 code accuracy 高：decoder 或 VQ codebook 局部几何表达不足。

## 实验 2：Attribute-Conditioned Local Editing

### 目标

验证模型能否根据属性指令修改局部 3D token，同时保持非目标区域不变。

### 数据构造

第一版优先用程序化数据，因为属性 GT 可控。

配对样本：

```text
car_closed_door -> car_left_door_open_60deg
car_light_off -> car_headlight_on
traffic_light_red -> traffic_light_green
cone_upright -> cone_fallen
```

每个配对样本需要保存：

- source tokens
- target tokens
- changed node ids
- unchanged node ids
- attribute delta
- decoded source/target PLY

### 任务形式

输入：

```text
<OBJECT_START category=car>
...
<EDIT attribute=left_door state=open angle=60deg>
```

输出：

```json
{
  "changed_nodes": [21, 22, 23],
  "new_tokens": [
    {"node_id": 21, "main_code": 501},
    {"node_id": 22, "main_code": 777}
  ],
  "unchanged_policy": "keep_other_nodes"
}
```

### 属性优先级

第一批建议：

1. `traffic_light_state`: 最容易做颜色/语义控制。
2. `brake_light/headlight`: 局部外观变化。
3. `door_open`: 局部几何变化。
4. `wheel_angle`: 小几何变化，较难。
5. `cone_fallen/barrier_fallen`: 几何姿态变化，和驾驶风险强相关。

### 评估

- `attribute success rate`: 用规则或 probe 判断属性是否变对。
- `changed region recall/precision`: 预测变化 node 是否覆盖 GT changed nodes。
- `unchanged token preservation`: 非目标 node 是否保持不变。
- `decoded geometry delta`: 几何变化是否局部且符合属性。
- `semantic consistency`: 类别不应被改坏。

### 预期诊断

- 属性成功但改动过大：VLM 缺少局部编辑约束。
- 非目标区域变化多：需要 delta-format 输出，而不是重写全物体。
- 属性无法学会：现有 tokenizer 可能没有给对应部件足够 LOD 或 RGB 表达。

## 实验 3：Object Footprint / Obstacle Contour Generation

### 目标

验证模型是否能从 partial object tokens 生成可用于驾驶安全的轮廓和占用边界。

这里不追求完整漂亮 mesh，优先验证：

- BEV footprint
- collision boundary
- conservative occupied area

### 数据构造

输入：

- 完整物体 token。
- partial/cropped 物体 token。
- 类别提示或 unknown obstacle。

GT：

- 从完整点云投影到 BEV 得到 footprint polygon 或 occupancy grid。
- 从 mesh/point cloud 计算 2D convex hull / alpha shape。
- 对障碍物适当膨胀，得到 conservative safety footprint。

适合类别：

- 路锥
- 倒伏路锥
- 标志牌
- 护栏/路障
- 掉落箱体
- 车辆和两轮车的 footprint

### 输出形式

第一版输出 JSON，比直接输出完整 token 更容易评估：

```json
{
  "footprint_polygon": [[-1.2, -0.4], [1.1, -0.4], [1.2, 0.5], [-1.1, 0.5]],
  "confidence": 0.82,
  "conservative": true
}
```

第二版再要求输出补全后的 object tokens，并通过 decoder 恢复 surface。

### 评估

- `footprint IoU`
- `boundary Chamfer`
- `Hausdorff distance`
- `occupied recall`
- `free-space false negative rate`
- `conservative risk recall`

对于自动驾驶，漏检风险边界比略微扩大边界更严重，所以第一版主指标应偏向 recall。

### 预期诊断

- footprint IoU 低：shape prior 不足或 partial token 信息不足。
- occupied recall 低：模型对安全边界不够保守。
- unknown obstacle 表现差：类别条件依赖过强，需要更多形状多样性。

## Partial Octree 与 Full Octree 结构不一致问题

### 问题

在 object-level 几何/纹理补全中，训练输入通常是 partial point cloud，监督目标是完整 object。由于当前 octree/统一树是根据点云分布自适应构建的，partial 输入和 full GT 很可能生成不同树结构：

- partial 点云只覆盖可见表面，节点分布偏向可见侧。
- full object 点云覆盖完整表面，节点分布更均衡。
- partial bbox 如果由可见点计算，center、size、yaw 也可能偏离完整物体。
- partial tree 的 `split_flag / child_mask / path_code / node_count` 不一定能与 full tree 一一对应。

如果直接要求模型学习：

```text
partial node i -> full node i
```

会导致明显学习困难，因为 `node i` 在两棵树里可能代表完全不同的空间区域。

### 结论

这是一个真实风险。第一版不应把 partial-to-complete 设计成简单的逐 node token 对齐任务，而应显式处理结构不一致。

### 推荐处理方式

#### 方案 A：固定 canonical scaffold

训练补全任务时，使用同一个 canonical bbox 和同一套空间 scaffold 来编码 partial 与 full：

- bbox 使用完整物体 bbox，或训练时的 oracle bbox。
- partial points 只填充可见区域。
- full points 用同一 scaffold 生成监督。

优点：

- partial/full node 有明确空间对应。
- 最适合验证模型是否能从可见区域补全不可见区域。

缺点：

- 使用完整 bbox 是 oracle，和真实自驾部署不完全一致。

用途：

- 作为 upper-bound 实验。
- 验证 VQVAE code 和 VLM 是否具备补全能力。

#### 方案 B：partial-derived bbox + full target in partial frame

部署更接近真实自驾：

- bbox 只由 partial 点云估计。
- full object GT 变换到 partial-derived local frame。
- 输出完整 object token 或 footprint。

优点：

- 更接近真实自驾，因为自驾中通常没有准确完整 bbox。

缺点：

- partial bbox 的尺度和中心偏差会带来额外难度。
- 模型同时要学 bbox 修正、形状补全和 token 生成。

用途：

- 作为 realistic 实验。
- 需要和方案 A 分开汇报，不要混在一起。

#### 方案 C：不做逐 node 对齐，改用几何场监督

把 partial tokens 作为条件，输出 full object representation，但监督不要求 node index 对齐，而是在连续空间查询上评估：

- UDF / occupancy query
- surface points
- BEV footprint
- Chamfer / F-score

优点：

- 避免 partial tree 与 full tree 的逐节点错位。
- 更适合不同 tokenizer/tree 结构之间的比较。

缺点：

- 如果输出仍是离散 token，需要额外 decoder 或 validator。

用途：

- 作为主要几何补全评估方式。

#### 方案 D：输出 full-tree tokens，但输入保留 partial-tree tokens 的空间字段

如果 VLM 输入 partial tree、输出 full tree，输入 token 不能只包含 compact DFS 序列。需要保留更多空间锚点：

- `path_code`
- `level`
- `node_center_local`
- `node_size_local`
- `split_flag`
- `child_mask`
- `observed_ratio`
- `visibility / partial_mask`

否则当 partial/full 树结构不同，模型很难知道每个输入 token 对应哪个空间区域。

### 推荐实验对照

第一版应显式做三组对照：

| 实验 | 输入 bbox/tree | 监督 | 目的 |
|------|----------------|------|------|
| Oracle scaffold | full bbox + shared scaffold | full tokens / full geometry | 上限能力，排除 bbox/tree mismatch |
| Realistic partial | partial bbox + partial tree | full geometry / footprint | 接近自驾部署 |
| Rich spatial tokens | partial bbox + partial tree + spatial fields | full tokens / geometry | 验证空间字段能否缓解树不一致 |

### 当前优先设置：Realistic partial

当前决定第一版直接进入 `Realistic partial` 设置，而不是先做 oracle scaffold。原因是该设置更接近后续自驾部署：

- 输入端没有完整 object bbox，只能从 partial 点云估计 bbox。
- 输入 octree 由 partial 点云构建。
- GT 端由完整 object 点云构建完整 object token。
- 模型需要从 partial token 自回归生成补全后的 full object token。
- 生成 token 再通过 tokenizer decoder 解码成点云、UDF、surface 或颜色，用于几何/外观检查。

这会比 oracle scaffold 更难，但验证结果更有实际价值。

#### Realistic partial 训练样本

每个样本包含：

```text
partial_points.ply
partial_bbox.json
partial_tokens.json
partial_tokens_llm_sequence.json
full_points.ply
full_bbox.json
full_tokens.json
full_tokens_llm_sequence.json
```

其中：

- `partial_bbox` 只由 partial 点云计算。
- `partial_tokens` 由 partial bbox/local frame 下的 partial point cloud 构建。
- `full_tokens` 由完整 object 点云构建，可使用 full bbox/local frame。
- 如果 partial/full local frame 不一致，需要在样本元数据中保存 `partial_to_full_transform`，用于 decode 后统一评估。

#### LLM 自回归 token 补全方案

输入：

```text
<TASK object_completion>
<PARTIAL_OBJECT category=car bbox_source=partial>
<PARTIAL_HEADER ...>
<PARTIAL_NODE ...>
...
<TARGET_FULL_OBJECT>
```

输出：

```text
<FULL_HEADER ...>
<FULL_NODE ...>
<FULL_NODE ...>
...
<FULL_OBJECT_END>
```

训练方式：

- 用 teacher forcing 做 next token prediction。
- 目标序列是完整 object 的 token 序列。
- 输入序列是 partial object token 加任务提示。

这条路线是合理的，因为它把补全任务转成标准自回归序列建模，和 Qwen/LLM 的训练范式一致。

#### Token 监督还是点云/颜色 loss

第一版建议：

1. `token-level supervision` 作为主训练目标。
2. `decoded geometry/color metrics` 作为主要评估。
3. `decoded geometry/color loss` 作为第二阶段可选辅助目标。

原因：

- LLM 直接输出的是离散 token，最自然的训练目标是 token cross-entropy。
- 如果直接把点云/颜色 loss 反传到 LLM，需要 decoder 可微、生成 token 可微或使用 soft token/expected embedding，工程复杂度明显上升。
- token loss 可以稳定训练语法和 full object token 分布。
- 几何/颜色 loss 更适合作为评估，或者在后续用 differentiable decoder + soft token 做辅助。

#### 推荐 loss 组合

第一阶段：

```text
L = L_token_ce
```

评估时：

```text
decode(pred_tokens) -> pred_points / pred_surface / pred_rgb
compare with full_points
```

评估指标：

- token exact / top-k accuracy
- tree grammar validity
- decoded Chamfer / F-score
- UDF / occupancy error
- RGB MSE / LPIPS-like feature distance if available
- footprint IoU for driving geometry

第二阶段可选：

```text
L = L_token_ce + lambda_geo * L_decoded_geometry + lambda_rgb * L_decoded_rgb
```

但第二阶段需要解决离散 token 的梯度问题，可选方法：

- 使用 softmax over codebook，再由 decoder 解码 expected code embedding。
- 使用 Gumbel-Softmax / straight-through estimator。
- 只对一个轻量 token refiner 训练 decoded loss，不直接端到端微调 Qwen。

#### 关键风险

- partial tree 与 full tree 不一致时，LLM 必须学会从 partial sequence 生成另一棵 full tree，而不是简单填空。
- 如果 `partial_tokens_llm_sequence` 过于 compact，缺少空间锚点，模型会很难知道 partial node 对应物体哪个区域。
- 因此 Realistic partial 输入建议使用 richer token，而不是只用 v2 compact node：
  - `level`
  - `path_code`
  - `node_center_local`
  - `node_size_local`
  - `observed_ratio`
  - `partial_bbox`
  - `category`

#### 是否在输出侧也预测 partial rich spatial fields

可以，而且这可能是第一阶段很有帮助的辅助任务。它的作用不是补全物体，而是让模型先学会“读懂并重述 partial octree 结构”。

建议把输出拆成两个 block：

```text
<PARTIAL_STRUCTURE_RECON>
  predict/restate partial header
  predict/restate partial node level/path/center/size/observed_ratio
<FULL_COMPLETION>
  predict full header
  predict full node structure
  predict full main_code
```

其中：

- `PARTIAL_STRUCTURE_RECON` 是自编码式结构理解任务。
- `FULL_COMPLETION` 是真正的 partial-to-full 补全任务。

这样做的好处：

- 让模型先对输入 partial tree 的空间结构建立稳定表示。
- 可以检查模型是否真正读懂 `level / path_code / node_center / node_size / observed_ratio`，而不是只记 token 序列模式。
- 对小数据尤其有帮助，因为 partial structure reconstruction 的标签完全自动、无噪声。
- 可以作为 curriculum：先训练结构重述，再训练 full completion，最后联合训练。

注意事项：

- partial rich fields 可以作为输出辅助监督，但 full rich fields 不能放进输入侧。
- partial reconstruction loss 不应压过 full completion loss，否则模型可能学会复读 partial，而不学补全。
- 第一版建议给 partial structure loss 较小权重，或先单独预训练再联合。

推荐 loss：

```text
L = L_full_token_ce
  + alpha * L_partial_structure_ce
```

其中：

- `L_full_token_ce` 是主目标。
- `L_partial_structure_ce` 只监督 partial rich spatial fields 的重述。
- `alpha` 第一版可设为 `0.1 ~ 0.3`，或采用先预训练后关闭的 curriculum。

更推荐的训练顺序：

1. `partial structure reconstruction`：只让模型重述 partial rich spatial fields。
2. `full token completion`：输入 partial rich fields，输出 full rich fields + full code。
3. `joint objective`：少量 partial structure loss + 主要 full completion loss。

#### 推荐第一版实验

第一版只做 object-level geometry/color completion：

- 数据：80 train / 20 test Objaverse++ object-level 数据。
- partial 构造：每个完整 object 生成 4-8 个视角 partial。
- 输入：partial-derived bbox + partial octree rich token。
- 输出：full object token。
- 训练：token CE。
- 验证：decode token 后用 geometry/color/footprint 指标评估。

如果这个设置下 token CE 学得动、decode 几何也有提升，就说明该路线对后续自驾有价值。

### 与自驾扩展的关系

自驾场景中通常只能从 partial point cloud 或检测结果估计 bbox，因此不能长期依赖完整 object bbox。合理路线是：

1. 先用 oracle scaffold 建立 object completion 上限。
2. 再引入 partial-derived bbox，测量性能下降。
3. 再用 bbox jitter、partial bbox augmentation、spatial token fields 缓解下降。
4. 最后在自驾场景中把补全目标从“完整精美 mesh”收敛为“驾驶可用几何”：footprint、占用边界、碰撞风险区域和关键局部部件。

## 数据规模建议

### Smoke

- 10 类。
- 每类 10 个对象。
- 每个对象 5 个 mask/edit/partial 变体。
- 总计约 500 task samples。

目标：

- 跑通数据格式、Qwen 输入输出、validator 和评估脚本。

### Small

- 30 类。
- 每类 50 个对象或程序化变体。
- 每个对象 10 个任务变体。
- 总计约 15,000 task samples。

目标：

- 初步比较 baseline 与 Qwen-LoRA。
- 观察 token 语义是否可学。

### Medium

- 50-100 类。
- 每类 100-500 个对象或变体。
- 总计 100k-500k task samples。

目标：

- 正式训练 object-level 3D VLM 静态细节能力。

## 推荐实验顺序

### Step 1：只做 Masked Local Node Infilling

原因：

- 数据全自动。
- 不依赖属性标注。
- 最能直接诊断 VQ codebook 是否适合 VLM 学习。

通过标准：

- Qwen-LoRA 显著超过 frequency baseline 和 nearest neighbor baseline。
- top-k code accuracy 随类别和局部上下文增加而提升。
- 解码后的局部几何质量优于随机填充。

### Step 2：程序化 Attribute Editing

原因：

- 真实数据缺少车灯/车门等细粒度属性 GT。
- 程序化数据可控，能清楚区分 changed/unchanged nodes。

通过标准：

- 属性变更成功。
- 非目标区域 token preservation 高。
- decoded delta 与属性变化区域一致。

### Step 3：Partial-to-Footprint / Contour

原因：

- 直接连接驾驶安全几何。
- 输出 footprint JSON 可稳定评估。

通过标准：

- occupied recall 高。
- footprint IoU 超过几何 baseline。
- 对 unknown obstacle 不完全崩溃。

### Step 4：混合真实 Objaverse++ 与程序化数据

原因：

- 程序化数据提供属性 GT。
- Objaverse++ 提供真实形状分布。

通过标准：

- 在真实 Objaverse++ val 上 masked infilling 和 footprint 仍有提升。
- 在程序化 val 上属性编辑稳定。
- 不同数据源的 token schema 保持一致。

## 与 Tokenizer 的关系

本阶段仍保持模块解耦：

- VQVAE tokenizer 只负责输出 object tokens 和可选 decoder。
- object-level task builder 读取 tokenizer 输出，生成 VLM 任务样本。
- Qwen/3D VLM 训练不进入 tokenizer 代码。
- 评估脚本可调用 tokenizer decoder，但必须通过稳定 token schema。

如果实验失败，按以下方向诊断 tokenizer：

- 局部 infilling 学不会：检查 codebook size、node 粒度、instance-level context token。
- 属性编辑学不会：检查对应部件是否被切到足够细，RGB/外观是否进入 `main_code`。
- footprint 生成不稳：检查 bbox/scale 量化、低层 node 是否保留轮廓边界。

## 第一版产物

建议第一版实现以下产物：

- `object_generation_tasks_manifest.jsonl`
- `masked_node_infilling.jsonl`
- `attribute_editing.jsonl`
- `footprint_generation.jsonl`
- `object_task_eval_config.json`
- 三个评估脚本入口：
  - `eval_masked_node_infilling`
  - `eval_attribute_editing`
  - `eval_footprint_generation`
