# 3dVAE 最终设计方案

## 1. 目标

本项目的目标是为自动驾驶场景构建一种面向 VLA / LLM 的 3D 场景编码方案。该方案应同时满足以下要求：

- 能表达自车坐标系下的整场景空间关系
- 能表达实例级几何与纹理信息
- 支持结构化压缩，控制 token 长度
- 对同类实例在不同场景位置下保持较稳定的局部表示
- 为后续时序扩展、生成任务和下游理解任务预留接口

当前默认实现先聚焦单帧路径，优先把单帧场景编码、实例编码、训练和评估闭环跑通。

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

每个实例点云会变换到实例局部坐标系下编码。实例编码不做尺寸归一化，保留物理世界真实尺度。

## 4. 场景表示

整场景采用两层表示。

### 4.1 Scene Layer

`scene layer` 负责表达实例在自车坐标系下的全局信息，包括：

- 实例类别 `semantic_id`
- 实例位置 `center_xyz`
- 实例朝向 `yaw`
- 实例尺度 `box_size_xyz`

### 4.2 Instance Layer

`instance layer` 负责表达实例在局部坐标系下的结构与外观，包括：

- 局部层级结构
- 几何编码
- 纹理编码
- 学习式节点编码

当前单帧实现中，时序信息暂不进入编码主链路。

## 5. 实例层结构：统一自适应树

### 5.1 基本原则

实例层统一采用一种“带 `split_flag` 的自适应树”表达，而不是维护完全独立的 `Octree` 与 `Quadtree` 两套结构。

### 5.2 split_flag

`split_flag` 的位定义如下：

- bit0：沿 `X` 轴切分
- bit1：沿 `Y` 轴切分
- bit2：沿 `Z` 轴切分

因此：

- `0b111`：标准 3D 切分
- `0b011`：仅沿 `XY` 切分，适合地面类
- `0b101`：仅沿 `XZ` 切分
- `0b110`：仅沿 `YZ` 切分

### 5.3 child_mask

每个节点额外记录 `child_mask`，表示哪些 child slot 被显式物化成了子节点。

`child_mask` 的含义是：

- 仅表达结构拓扑
- 不直接表达 free-space
- 不直接表达未观测
- 不直接表达物理为空

也就是说，`child_mask` 的 0 bit 只表示该 child slot 当前没有被展开成独立节点。

### 5.4 切分顺序

当前切分决策采用规则链，而不是加权总分：

1. 语义优先级
2. 几何复杂度
3. RGB 变化

### 5.5 节点展开顺序

节点序列化采用：

- `preorder DFS`

child slot 顺序采用：

- `active_axes_binary`

这两条约定决定了 compact token 在不依赖 `path_code` 的情况下仍可恢复树拓扑。

## 6. 节点编码

每个节点当前支持三类信息：

- 结构编码：`split_flag + child_mask`
- 规则式编码：`geom_code + rgb_code`
- 学习式编码：`learned_code`

### 6.1 规则式编码

- `geom_code`：基于节点密度、几何复杂度、层级等规则量化得到
- `rgb_code`：基于节点 RGB 均值与方差量化得到

### 6.2 学习式编码

学习式节点编码由节点级 `VQ-VAE` 产生，对外表现为：

- `learned_code`

当前系统允许规则式编码和学习式编码并行输出，便于调试和对比。

## 7. 学习式实例编码器

### 7.1 总体结构

当前实例编码器采用节点级 `PointNet + VQ + Decoder` 结构。

编码器组成：

- `PointNetEncoder`
- `pre_quant`
- `VectorQuantizer`
- `PointCloudDecoder`
- `UDFDecoder`

### 7.2 输入形式

训练时支持两种 sample unit：

- `instance`
- `node`

当前默认推荐使用：

- `node`

也就是以统一树的节点作为训练样本。

### 7.3 训练目标

当前训练目标为：

- `xyz reconstruction`
- `rgb reconstruction`
- `vq loss`
- `udf loss`

### 7.4 UDF 监督

当前 `UDF` 监督采用基于 partial 点云的 observed UDF baseline：

- 在节点局部 bbox 中采样查询点
- 用查询点到局部点云最近邻的距离作为监督值
- 对距离进行截断

该方案的优点是：

- 不依赖 watertight mesh
- 不依赖传感器射线
- 适合当前 partial point cloud 条件

## 8. Token 导出层

当前 pipeline 同时导出多层表示，以兼顾调试和下游消费。

### 8.1 完整调试版

文件：

- `scene_tokens.json`

特点：

- 保留调试友好字段
- 保留 `parent_id / child_index / path_code`
- 适合人工排查和树结构核查

### 8.2 紧凑结构版 v1

文件：

- `scene_tokens_compact.json`

特点：

- 去掉调试字段
- 保留结构恢复所需核心字段

### 8.3 线性序列版 v1

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

### 8.4 紧凑结构版 v2

文件：

- `scene_tokens_compact_v2.json`

这是当前推荐的 compact 结构表示。

#### v2 的关键变化

- 用 `INSTANCE_HEADER` 的信息取代显式 `POSE + INSTANCE_START` 拆分
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

### 8.5 LLM 线性序列版 v2

文件：

- `scene_tokens_llm_sequence_v2.json`

这是当前推荐的 LLM 输入候选格式。

格式：

- `SCENE_START`
- 多个 `INSTANCE_HEADER`
- 每个实例对应多个 `INSTANCE_NODE_V2`
- `SCENE_END`

其中：

`INSTANCE_NODE_V2` 仅保留：

- `split_flag`
- `child_mask`
- `main_code`

### 8.6 默认推荐出口

当前 pipeline 还会生成：

- `scene_tokens_compact_latest.json`
- `scene_tokens_llm_sequence_latest.json`
- `scene_token_bundle.json`

当前默认约定：

- `scene_tokens_compact_latest.json` 指向 `v2` compact
- `scene_tokens_llm_sequence_latest.json` 指向 `v2` LLM sequence

`scene_token_bundle.json` 记录推荐读取的文件名。

## 9. 调试输出

为了检查树结构与实例表达，当前 pipeline 还会导出以下中间结果：

- `scene_instance_bboxes.ply`
- `instance_<id>_bbox.json`
- `instance_<id>_octree_nodes.jsonl`

其中场景 bbox 与节点 bbox 采用统一几何表示：

- `8` 个顶点
- `6` 个四边形面

## 10. 评估方案

当前评估体系已实现，并配套评估脚本。

### 10.1 已实现的指标

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

### 10.2 评估脚本

脚本：

- `scripts/evaluate_instance_tokenizer.py`

输出：

- `evaluation_metrics.json`
- `evaluation_summary.md`
- `evaluation_metrics.csv`

### 10.3 模板

结果模板：

- `templates/evaluation_results_template.md`
- `templates/evaluation_results_template.csv`

## 11. 当前默认推荐使用方式

### 11.1 训练

默认推荐：

- 使用 `node` 模式训练节点级 tokenizer
- 使用当前 `UDF` 辅助监督

### 11.2 导出

默认推荐：

- 调试使用：`scene_tokens.json`
- 下游读取使用：`scene_tokens_llm_sequence_latest.json`

### 11.3 评估

默认推荐：

- 用 `scripts/evaluate_instance_tokenizer.py` 统一导出实验结果
- 所有实验统一保存 JSON、Markdown 和 CSV 三份报告

## 12. 方案边界

当前最终方案明确聚焦于：

- 单帧场景
- partial RGB point cloud
- 节点级统一树编码
- 节点级 VQ-VAE
- 面向 LLM 的压缩结构序列

以下内容不属于当前最终方案的已完成部分：

- 动态物体时序编码主链路
- free-space / visibility 的可靠恢复
- TSDF / SDF 监督
- 下游 VLA 任务自动评估

这些内容保留为后续扩展方向，但不属于当前默认实现。
