# 3DSceneTokenizer 最终设计方案

## 1. 目标

本项目的目标是为自动驾驶场景构建一套面向 `VLA / LLM` 的 3D 场景编码方案。该方案需要同时满足：

- 表达自车坐标系下的场景空间关系
- 表达实例级几何与纹理信息
- 支持结构化压缩，尽量缩短 token 序列
- 让同类实例在不同场景位置下保持相对稳定的局部表示
- 为未来的时序扩展、生成任务和下游理解任务预留接口

当前默认实现先聚焦单帧路径，优先把单帧场景编码、实例编码、训练和评估链路跑通。

## 2. 输入定义

当前上游输入固定为单帧 `PLY` 文件。每帧 `PLY` 至少包含以下字段：

- `x`
- `y`
- `z`
- `red`
- `green`
- `blue`
- `instance`
- `semantic`

约定如下：

- `instance` 为数值型实例 ID
- `semantic` 为数值型语义 ID
- 地面区域同样允许拥有数值型 `instance_id`

## 3. 坐标系与实例局部坐标

### 3.1 自车坐标系

- `x`：车头前向
- `y`：自车左侧
- `z`：地面朝上

### 3.2 实例位姿与包围盒

每个实例先在自车坐标系下估计一个与地面平行的 `OBB`。当前只保留：

- `center_xyz`
- `size_xyz`
- `yaw`

其中：

- `yaw` 定义为实例朝向轴相对自车坐标系 `X` 轴的偏角
- 当前不建模 `roll / pitch`

### 3.3 实例局部坐标系

每个实例点云会变换到实例局部坐标系下编码。实例编码不做尺寸归一化，保持真实物理尺度。

## 4. 场景表示

整体场景采用两层表示。

### 4.1 Scene Layer

`scene layer` 负责表达实例在自车坐标系下的全局信息，包括：

- `semantic_id`
- `center_xyz`
- `yaw`
- `box_size_xyz`

### 4.2 Instance Layer

`instance layer` 负责表达实例在局部坐标系下的结构与外观，包括：

- 局部层级结构
- 几何编码
- 纹理编码
- 学习式节点编码

当前单帧实现中，时序信息暂不进入主编码链路。

## 5. 实例层结构：统一自适应树

### 5.1 基本原则

实例层统一采用一种“带 `split_flag` 的自适应树”，而不是维护完全独立的 `Octree` 和 `Quadtree` 两套结构。

### 5.2 split_flag

`split_flag` 的位定义如下：

- bit0：沿 `X` 轴切分
- bit1：沿 `Y` 轴切分
- bit2：沿 `Z` 轴切分

因此：

- `0b111`：标准 3D 切分
- `0b011`：仅沿 `XY` 切分，适合地面
- `0b101`：仅沿 `XZ` 切分
- `0b110`：仅沿 `YZ` 切分

### 5.3 child_mask

每个节点额外记录 `child_mask`，表示哪些 child slot 被显式物化成了子节点。

`child_mask` 的含义是：

- 只表达结构拓扑
- 不直接表达 free-space
- 不直接表达未观测
- 不直接表达物理上为空

也就是说，`child_mask` 的 `0 bit` 只表示该 child slot 当前没有被展开成独立节点。

## 6. 节点是否继续切分：最终准则

这是当前方案中最关键的更新之一。节点切分不再只依赖简单的 `occupied_extent / node_size`，而是升级为三段式决策：

1. `hard gating`
2. `complexity scoring`
3. `axis selection`

### 6.1 Hard Gating

先用硬约束过滤掉无意义切分：

- 达到 `max_depth` 时停止
- 点数小于 `min_points_per_node` 时停止
- 语义优先级决定 `min_depth`
- 语义优先级和距离共同决定 `max_depth`

这一步负责给不同类别和不同距离分配 token 预算。

### 6.2 Complexity Scoring

通过 gating 后，再计算局部复杂度。当前最终实现中，局部几何复杂度由三部分组成：

- `extent_score`
- `occupancy_score`
- `plane_residual_score`

#### extent_score

活动切分轴上的占据范围占节点尺寸的比例。它衡量该节点在当前可切分方向上是否“铺满”了足够大的空间。

#### occupancy_score

候选子槽位中被点云实际占据的比例。它衡量局部分布是否已经足够复杂，值得继续分配更多节点。

#### plane_residual_score

点到最佳拟合平面的归一化残差。它衡量该节点是否只是简单平面，还是具有更强的 3D 几何起伏。

#### 当前几何复杂度组合

当前实现采用加权组合：

- `extent_weight = 0.45`
- `occupancy_weight = 0.35`
- `plane_residual_weight = 0.20`

即：

`geom_score = 0.45 * extent_score + 0.35 * occupancy_score + 0.20 * plane_residual_score`

此外还保留：

- `rgb_score`

用于在几何复杂度不足时补充外观驱动的细化能力。

### 6.3 Axis Selection

是否切分与沿哪些轴切分是一起决定的。当前 `split_flag` 的选择依据为：

- 语义显式覆盖
- `occupied_extent`
- `axis_std`

具体规则是：

- 如果语义被明确指定为 `XY / XZ / YZ`，优先使用语义覆盖
- 否则根据局部点云在三个轴上的 `extent` 与 `std` 估计各向异性
- 若最弱轴信号显著低于阈值，则退化为两轴切分
- 若三个方向都足够显著，则采用 `XYZ`

这使得：

- 体状实例更容易走 `XYZ`
- 近平面实例即使没有语义强制覆盖，也能自动退化到 `XY / XZ / YZ`

### 6.4 当前切分顺序

当前最终规则可以概括为：

1. 先由语义和距离决定深度预算
2. 再由局部几何复杂度和 RGB 复杂度决定值不值得继续切
3. 最后由局部各向异性决定沿哪些轴切

这是一个 `budget-aware + error-aware + anisotropy-aware` 的切分准则，借鉴了 OAT 的“局部复杂度驱动 token 分配”思想，但适配了自动驾驶 partial RGB 点云和统一树结构。

## 7. 语义策略表：深度与 split_flag 的统一配置

当前方案进一步把 `depth` 和 `split_flag` 合并成了一张统一的语义策略表，而不是分别由优先级集合和单独的切分规则控制。

### 7.1 统一策略对象

每个语义类对应一个 `SemanticOctreePolicy`，至少包含：

- `tag_name`
- `min_depth`
- `max_depth_by_distance`
- `preferred_split_flag`
- `lock_preferred_split`
- `priority`

### 7.2 与距离的融合方式

运行时先根据实例中心点计算距离段：

- `near`
- `mid`
- `far`

再按 `semantic_id` 查询策略表，从中读取：

- `min_depth`
- `max_depth_by_distance[distance_bin]`
- `preferred_split_flag`

因此当前最大深度的确定方式是：

- 先按语义查策略
- 再按距离段取该语义在对应距离下的 `max_depth`

### 7.3 split_flag 的使用方式

`split_flag` 不再只由几何规则决定，而是遵循：

- 语义先验
- 几何统计
- 可选锁定

具体逻辑：

- 若 `lock_preferred_split=true`，直接使用语义表中的 `preferred_split_flag`
- 若 `lock_preferred_split=false`，则用语义先验和局部几何推断融合得到最终 `split_flag`

### 7.4 当前默认 CARLA 语义策略

当前代码中已内置一份基于 CARLA 语义表的默认策略：

- 地面相关：
  - `Roads / SideWalks / RoadLine / Ground` 默认 `XY`
- 体状目标：
  - `Car / Pedestrian / Bicycle / TrafficLight` 等默认 `XYZ`
- 背景结构：
  - `Building / Wall / Fence / GuardRail` 当前先保守设为 `XYZ`，再允许几何退化

当前实现提供：

- `build_default_carla_semantic_policies()`
- `OctreeBuildConfig.with_default_carla_semantics()`

## 8. 节点展开顺序

节点序列化采用：

- `preorder DFS`

child slot 顺序采用：

- `active_axes_binary`

这两条约定保证 compact token 在不依赖 `path_code` 的情况下仍可恢复树拓扑。

## 9. 节点编码

每个节点当前支持三类信息：

- 结构编码：`split_flag + child_mask`
- 规则式编码：`geom_code + rgb_code`
- 学习式编码：`learned_code`

### 8.1 规则式编码

- `geom_code`：基于节点密度、几何复杂度、层级等规则量化得到
- `rgb_code`：基于节点 RGB 均值与方差量化得到

### 8.2 学习式编码

学习式节点编码由节点级 `VQ-VAE` 产生，对外表现为：

- `learned_code`

当前系统允许规则式编码和学习式编码并行输出，便于调试和对比。

## 10. 学习式实例编码器

### 9.1 总体结构

当前实例编码器采用节点级 `PointNet + VQ + Decoder` 结构。

编码器组成：

- `PointNetEncoder`
- `pre_quant`
- `VectorQuantizer`
- `PointCloudDecoder`
- `UDFDecoder`

### 9.2 输入形式

训练时支持两种 `sample unit`：

- `instance`
- `node`

当前默认推荐使用：

- `node`

即以统一树的节点作为训练样本。

### 9.3 训练目标

当前训练目标为：

- `xyz reconstruction`
- `rgb reconstruction`
- `vq loss`
- `udf loss`

### 9.4 UDF 监督

当前 `UDF` 监督采用基于 partial 点云的 observed UDF baseline：

- 在节点局部 `bbox` 中采样查询点
- 用查询点到局部点云最近邻的距离作为监督值
- 对距离进行截断

该方案的优点是：

- 不依赖 watertight mesh
- 不依赖传感器射线
- 适合当前 partial point cloud 条件

## 11. Token 导出层

当前 pipeline 同时导出多层表示，以兼顾调试和下游消费。

### 10.1 完整调试版

文件：

- `scene_tokens.json`

特点：

- 保留调试友好字段
- 保留 `parent_id / child_index / path_code`
- 适合人工排查和树结构核查

### 10.2 紧凑结构版 v1

文件：

- `scene_tokens_compact.json`

特点：

- 去掉调试字段
- 保留结构恢复所需核心字段

### 10.3 线性序列版 v1

文件：

- `scene_tokens_llm_sequence.json`

格式：

- `SCENE_START`
- 多个 `POSE`
- 每个实例：
  - `INSTANCE_START`
  - 多个 `INSTANCE_NODE`
  - `INSTANCE_END`
- `SCENE_END`

### 10.4 紧凑结构版 v2

文件：

- `scene_tokens_compact_v2.json`

这是当前推荐的 compact 结构表示。

#### v2 的关键变化

- 用 `INSTANCE_HEADER` 取代显式 `POSE + INSTANCE_START` 拆分
- 节点 token 不再显式保留：
  - `level`
  - `num_points`
  - `code_source`
- 节点主码收敛成单一 `main_code`

#### v2 header

每个实例的 `INSTANCE_HEADER` 当前包含：

- `semantic_id`
- `center_xyz_q`
- `yaw_q`
- `box_size_xyz_q`
- `node_count`
- `main_code_scheme`

其中：

- `center_xyz_q`：位置量化结果
- `yaw_q`：朝向量化结果
- `box_size_xyz_q`：尺寸量化结果

### 10.5 LLM 线性序列版 v2

文件：

- `scene_tokens_llm_sequence_v2.json`

这是当前推荐的 LLM 输入候选格式。

格式：

- `SCENE_START`
- 多个 `INSTANCE_HEADER`
- 每个实例对应多个 `INSTANCE_NODE_V2`
- `SCENE_END`

其中 `INSTANCE_NODE_V2` 仅保留：

- `split_flag`
- `child_mask`
- `main_code`

### 10.6 默认推荐出口

当前 pipeline 还会生成：

- `scene_tokens_compact_latest.json`
- `scene_tokens_llm_sequence_latest.json`
- `scene_token_bundle.json`

当前默认约定：

- `scene_tokens_compact_latest.json` 指向 `v2 compact`
- `scene_tokens_llm_sequence_latest.json` 指向 `v2 LLM sequence`

`scene_token_bundle.json` 记录推荐读取的文件名。

## 12. 调试输出

为了检查树结构与实例表达，当前 pipeline 还会导出以下中间结果：

- `scene_instance_bboxes.ply`
- `instance_<id>_bbox.json`
- `instance_<id>_octree_nodes.jsonl`

其中场景 bbox 和节点 bbox 采用统一几何表示：

- `8` 个顶点
- `6` 个四边形面

## 13. 评估方案

当前评估体系已经实现，并配套评估脚本。

### 12.1 已实现的指标

#### 重建指标

- `xyz_mse`
- `rgb_mse`
- `chamfer_l2`
- `udf_smooth_l1`

#### 码本指标

- `code_count`
- `used_code_count`
- `usage_rate`
- `entropy_bits`
- `perplexity`

#### 压缩指标

- `sample_count`
- `avg_input_points`
- `latent_tokens_per_sample`
- `points_per_token_ratio`

### 12.2 评估脚本

脚本：

- `scripts/evaluate_instance_tokenizer.py`

输出：

- `evaluation_metrics.json`
- `evaluation_summary.md`
- `evaluation_metrics.csv`

### 12.3 模板

结果模板：

- `templates/evaluation_results_template.md`
- `templates/evaluation_results_template.csv`

## 14. 当前默认推荐使用方式

### 13.1 训练

默认推荐：

- 使用 `node` 模式训练节点级 tokenizer
- 使用当前 `UDF` 辅助监督

### 13.2 导出

默认推荐：

- 调试使用：`scene_tokens.json`
- 下游读取使用：`scene_tokens_llm_sequence_latest.json`

### 13.3 评估

默认推荐：

- 用 `scripts/evaluate_instance_tokenizer.py` 统一导出实验结果
- 所有实验统一保存 JSON、Markdown 和 CSV 三份报告

## 15. 方案边界

当前最终方案明确聚焦于：

- 单帧场景
- partial RGB point cloud
- 节点级统一树编码
- 节点级 `VQ-VAE`
- 面向 LLM 的压缩结构序列

以下内容不属于当前最终方案的已完成部分：

- 动态物体时序编码主链路
- free-space / visibility 的可靠恢复
- TSDF / SDF 监督
- 下游 VLA 任务自动评估

这些内容保留为后续扩展方向，但不属于当前默认实现。
