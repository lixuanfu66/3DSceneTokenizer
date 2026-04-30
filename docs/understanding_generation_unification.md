# 理解生成合一 3D 多模态研究框架

## 核心问题

当前项目的研究目标可以进一步收敛为：

> 面向自动驾驶具身智能，构建一种 3D 表示，使同一套 token/latent 同时支持空间理解与 3D 生成，并利用生成任务反向验证和增强空间理解能力。

这里的“理解生成合一”不是把理解模型和生成模型简单拼在一起，而是让二者共享同一种结构化 3D 表示：

- 理解侧读取场景层 pose、实例层 octree node、fused VAE/VQVAE code，回答物体几何、空间关系、语义和可供性问题。
- 生成侧根据同一套结构化表示重建、补全、预测或编辑 3D 场景。
- 如果表示真的理解了空间与形状，它应当能在受控条件下生成出几何一致、语义合理、空间关系正确的 3D 内容。

## 为什么 3D 生成可以成为 3D 理解的能力证明

监督式 3D 理解数据很难构建，尤其是自动驾驶具身场景中需要的能力：

- 物体局部几何是否被理解，例如车体、车轮、路沿、交通灯杆、障碍物形状。
- 形状与语义是否绑定，例如“像车的几何”不应被编码成普通背景块。
- 可见局部是否能推出合理整体，例如遮挡车辆、被截断的道路边界。
- 空间关系是否正确，例如车道线、路沿、车辆、行人的相对位置。
- 任务相关细节是否保留，例如刹车灯、转向灯、车轮方向、地面风险。

这些监督标签难以穷举标注。但生成任务提供了另一种检验方式：

- 如果模型能从压缩 token 重建局部 UDF/occupancy/surface，说明 token 中保留了几何理解所需信息。
- 如果模型能补全被遮挡或稀疏观测的形状，说明它学到了类别条件下的形状先验。
- 如果模型能在 scene layer 条件下生成空间一致的实例布局，说明它学到了物体间关系。
- 如果模型能根据语义、距离和任务条件调整 LOD，说明它学到了任务相关的信息分配。

因此，“能生成出来”不等于全部理解，但在严格约束和可量化评估下，可以作为理解能力的强 probe。

## 与当前 3dVAE 设计的契合点

当前设计天然适合这条路线：

- `scene layer` 显式表达物体在哪里：`semantic_id / center_xyz / yaw / box_size_xyz / trajectory`。
- `instance layer` 表达物体本地长什么样：统一树结构、node bbox、fused latent/code。
- `split_flag` 统一 Octree/Quadtree/退化切分，适合把地面、平面物体和体状物体放进同一语法。
- Octree Node VAE/VQVAE 让每个 node 既是压缩理解 token，也是可解码的生成单元。
- BFS/coarse-to-fine 序列天然适合自回归 LLM/多模态模型从粗到细地产生 3D 场景。

这使得项目可以避免“先做一个纯理解编码，再另做一个生成模型”的割裂路线。

## 建议的研究假设

### H1：结构化 3D 生成是空间理解的强监督替代

在缺少密集人工理解标签时，节点级 UDF/occupancy/surface reconstruction、实例补全和场景布局生成可以作为空间理解的自监督或弱监督训练信号。

### H2：统一树 token 能桥接理解与生成

同一套 `split_flag + child topology + node code + scene pose` 表示可以同时服务：

- 几何重建
- 形状语义 probe
- 场景关系理解
- 未来场景预测
- 3D 场景生成

### H3：生成质量必须被理解指标约束

不能只看生成表面是否好看，还要检查：

- 语义一致性
- 空间关系一致性
- 物理尺度一致性
- 驾驶任务相关细节是否保留
- token budget 是否可控

## 第一阶段实验路线

### 1. Object-level geometry-only VAE

目标：证明当前 node-level VAE 在干净对象数据上能学到稳定几何。

数据：

- Objaverse++ strict quality subset
- 先使用 geometry-only
- 每个对象采样 surface PLY，归一化或保留可控尺度

评估：

- node UDF MAE/RMSE
- surface reconstruction Chamfer/F-score
- token 数、depth 分布、node size 分布
- 类别条件下的线性 probe 或 kNN retrieval

### 2. Shape semantics probe

目标：验证 latent/code 是否包含形状语义，而不只是局部几何压缩。

方式：

- 用冻结 node/instance representation 做类别分类或检索。
- 比较规则 token、VAE latent、VQVAE code 的语义可分性。
- 检查生成/重建质量与语义 probe 是否同向提升。

### 3. Partial-to-complete 形状补全

目标：把自动驾驶 partial observation 与生成能力连接起来。

方式：

- 从 Objaverse 对象中模拟局部可见点云。
- 输入 partial node tokens，要求生成完整或更完整的 object-level surface/UDF。
- 检查补全是否保持类别形状先验和局部观测一致。

### 4. Scene-level spatial relation probe

目标：从 object-level 走回自动驾驶场景。

方式：

- 使用 Bench2Drive 场景层 token。
- 在已知实例 token 的情况下预测或重建 scene layer 中的相对空间关系。
- 检查是否能回答前后、左右、距离、可通行区域、风险区域等空间问题。

## 后续模型形态

理想的统一模型可以分成三层：

1. 3D tokenizer：把点云/mesh 编成 scene + instance + node tokens。
2. 3D generative decoder：从 tokens 解码 UDF/occupancy/RGB/surface。
3. 3D multimodal reasoner：在同一 token 序列上做问答、预测、规划和生成控制。

当前仓库正在实现第 1 层和第 2 层的基础；第 3 层需要等 token 稳定、评估可靠后再接入。

## 当前决策

- 把“生成验证理解”作为项目主线之一，而不是附属可视化功能。
- 短期优先在 Objaverse++ 高质量对象上验证 geometry-only 生成和形状语义。
- Bench2Drive 暂时作为 scene-level integration 与自动驾驶空间关系验证，不作为早期高精度几何收敛的唯一依据。
- 后续每次生成相关实验都必须同时记录生成指标和理解指标。

## 生成任务验证顺序

生成任务分成两大类：

- 静态细节生成：验证物体几何、外观、部件语义和驾驶相关细节。
- 动态未来生成：验证动态场景变化、交通参与者未来行为和风险演化。

第一版建议按以下五个小类逐步验证：

| 顺序 | 小类任务 | 所属大类 | 验证重点 |
|------|----------|----------|----------|
| 1 | Masked local node infilling | 静态细节生成 | node code 是否能表达局部几何/外观语义 |
| 2 | Attribute-conditioned local editing | 静态细节生成 | 车门、车灯、轮胎角、交通灯等属性是否与局部 3D token 对齐 |
| 3 | Object footprint / obstacle contour generation | 静态细节生成 | 障碍物轮廓、BEV footprint、碰撞边界是否可恢复 |
| 4 | Future scene layer prediction | 动态未来生成 | 动态物体未来 pose、yaw、velocity 是否可预测 |
| 5 | Multi-modal risk future generation | 动态未来生成 | 多可能未来、遮挡风险、交互风险是否可表达 |

这条顺序的依赖关系是：

```text
局部 token 读写
  -> 局部属性语义编辑
  -> 障碍物安全轮廓生成
  -> 动态物体未来状态预测
  -> 多模态风险未来生成
```

排序理由：

- 第 1 步最容易自动构造标签，先验证 VQ codebook 和 tree grammar 是否能被 VLM 学会。
- 第 2 步把局部几何/外观 token 与驾驶语义细节绑定，例如车门、刹车灯、转向灯、交通灯状态。
- 第 3 步把物体几何理解转成驾驶可用的安全边界，不追求最漂亮 mesh，而追求轮廓和碰撞边界可靠。
- 第 4 步进入时间维度，先预测 scene layer 的 pose/yaw/velocity，不急着生成完整未来几何。
- 第 5 步处理最接近真实驾驶决策的多可能未来和低概率高风险事件，应放在前四步稳定后做。
