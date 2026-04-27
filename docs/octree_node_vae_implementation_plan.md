# Octree Node VAE / VQVAE 实现方案

## 1. 目标

这一版模型的目标不是替换自适应八叉树，而是把当前 `learned_code` 从简单 `PointNet + VQ` 升级为可训练稳定、可逐步离散化的 node-level 几何残差编码器。

核心原则：

1. 八叉树继续负责结构、LOD 和 token 长度
2. 每个 node 默认只产生一个 learned geometry code
3. 复杂区域优先通过继续切分 node 获得更多 token
4. learned latent 表达当前 node 相对 parent/ancestor 的几何残差
5. 训练顺序是 `Geometry VAE -> Geometry+RGB fused VAE -> fused VQVAE`
6. LLM/AR 序列保留结构 token，新增 coarse-to-fine BFS 版本

## 2. 当前基线需要替换的部分

旧版 `PointNet + VQ` 实现已经移除，当前主实现是：

- 模型：[octree_node_vae.py](/D:/code/3dVAE/src/threedvae/models/octree_node_vae.py:94)
- VQ 模型：[octree_node_vqvae.py](/D:/code/3dVAE/src/threedvae/models/octree_node_vqvae.py:44)
- 损失：[octree_node_losses.py](/D:/code/3dVAE/src/threedvae/train/octree_node_losses.py:18)
- node dataset：[dataset.py](/D:/code/3dVAE/src/threedvae/data/dataset.py:92)
- learned code 接口：[octree_node_encoder.py](/D:/code/3dVAE/src/threedvae/tokenizer/octree_node_encoder.py:23)

当前问题：

- `mean pooling` 会过早丢失局部结构
- 单个 VQ code 在连续几何还没学稳时就被离散化
- fixed point reconstruction 难收敛，且和 octree 结构目标不完全一致
- RGB 与 geometry 早融合，容易拖累几何表示

## 3. 总体架构

新模型名建议：

- `OctreeNodeVAE`
- `OctreeNodeVQVAE`

新增文件建议：

- `src/threedvae/models/octree_node_vae.py`
- `src/threedvae/models/octree_node_vqvae.py`
- `src/threedvae/train/octree_node_losses.py`
- `src/threedvae/train/octree_node_trainer.py`
- `src/threedvae/tokenizer/octree_node_encoder.py`
- `scripts/train_octree_node_vae.py`
- `scripts/train_octree_node_vqvae.py`

旧模型：

- `PointNetVQTokenizer` 相关实现不再保留，避免训练和导出路径分叉

## 4. 数据接口设计

### 4.1 TreeNodePointCloudDataset 增量字段

当前 node sample 已经有：

- `xyz`
- `rgb`
- `points`
- `query_xyz`
- `query_udf`
- `level`
- `split_flag`
- `child_index`
- `parent_id`

需要新增或显式整理：

- `node_center_local`: `[3]`
- `node_size_local`: `[3]`
- `node_depth`: scalar，等价于 `level`
- `node_feature`: `[F_node]`
- `query_occ`: `[Q, 1]`
- `query_rgb`: `[Q_rgb, 3]`，Stage C 才需要
- `query_rgb_mask`: `[Q_rgb, 1]`，Stage C 才需要

第一阶段可以只增加：

- `node_center_local`
- `node_size_local`
- `query_occ`

### 4.2 query_occ 构造

当前 `build_udf_queries()` 已经输出 observed UDF。第一版可以从 UDF 派生 occupancy-like 监督：

```python
query_occ = (query_udf <= near_surface_threshold).float()
```

推荐默认：

- `near_surface_threshold = min(0.03, 0.2 * udf_truncation)`
- `udf_truncation = 0.25`

注意：这不是 watertight occupancy，而是 `observed near-surface occupancy`。命名上建议叫 `observed_occ`，避免误解。

### 4.3 geometry input feature

Stage A 输入不要使用 RGB。

每个点的几何特征建议：

```text
point_feature = [
  xyz_local,                # [3]
  xyz_normalized_to_node,   # [3]
  xyz_relative_to_center,   # [3]
]
```

第一版可用 `9` 维。

后续可加：

- local PCA normal `[3]`
- distance to node center `[1]`
- semantic embedding 不进入点特征，放 node condition

### 4.4 node condition feature

每个 node 的条件特征：

```text
node_condition = [
  node_center_local,        # [3]
  log_node_size,            # [3]
  level_normalized,         # [1]
  split_flag_bits,          # [3]
  child_index_bits,         # [3]
  semantic_embedding,       # [D_sem]
]
```

第一版可不用 semantic embedding，直接把 `semantic_id` 过 `nn.Embedding(num_semantics, semantic_dim)`。

## 5. Stage A：Geometry Octree Node VAE

### 5.1 模型配置

建议 dataclass：

```python
@dataclass(slots=True)
class OctreeNodeVAEConfig:
    num_points: int = 128
    queries_per_node: int = 128
    point_feature_dim: int = 9
    hidden_dim: int = 256
    latent_dim: int = 128
    node_condition_dim: int = 32
    semantic_vocab_size: int = 256
    semantic_dim: int = 16
    encoder_layers: int = 2
    decoder_layers: int = 2
    num_attention_heads: int = 4
    kl_weight: float = 1e-4
```

### 5.2 forward 输入

```python
forward(
    xyz: Tensor,              # [B, N, 3]
    query_xyz: Tensor,        # [B, Q, 3]
    node_center: Tensor,      # [B, 3]
    node_size: Tensor,        # [B, 3]
    level: Tensor,            # [B]
    split_flag: Tensor,       # [B]
    child_index: Tensor,      # [B]
    semantic_id: Tensor,      # [B]
    parent_latent: Tensor | None = None,  # [B, D]
)
```

### 5.3 forward 输出

```python
@dataclass(slots=True)
class OctreeNodeVAEOutput:
    latent: Tensor            # [B, D], absolute z_node
    residual_latent: Tensor   # [B, D], z_node - z_parent if parent exists
    mu: Tensor                # [B, D]
    logvar: Tensor            # [B, D]
    udf: Tensor               # [B, Q, 1]
    occ_logits: Tensor        # [B, Q, 1]
    decoder_features: Tensor  # [B, Q, H], for RGB Stage C
```

### 5.4 Encoder

第一版不需要完整 Perceiver，实现一个轻量 query encoder 即可：

1. `GeometryPointStem`
   - 输入 `[B, N, 9]`
   - 输出 `[B, N, H]`

2. `NodeConditionEncoder`
   - 输入 node metadata
   - 输出 node query `[B, 1, H]`

3. `CrossAttentionPooling`
   - query: node query `[B, 1, H]`
   - key/value: point features `[B, N, H]`
   - 输出 pooled node feature `[B, H]`

4. `Self/MLP refinement`
   - 第一版可以先用 MLP
   - 后续扩展 sibling/ancestor attention

5. `mu/logvar heads`
   - `Linear(H, D)` 两个头

### 5.5 Parent-relative residual

训练数据是独立 node batch，第一版不强依赖 parent batch 同时存在。

实现分两步：

#### Stage A.1：absolute latent

先训练：

```text
z_node = reparameterize(mu, logvar)
residual_latent = z_node
```

这一步让模型先收敛。

#### Stage A.2：residual latent

构造 batch 时可加入 parent sample 或离线 parent latent cache：

```text
z_node = parent_latent + residual_latent
```

第一版代码接口先预留 `parent_latent=None`，不强制实现 cache。

### 5.6 Decoder

输入：

- `query_xyz`
- `z_node`
- `node condition`

特征：

```text
query_feature = [
  query_xyz,
  query_xyz_normalized_to_node,
  query_xyz_relative_to_center,
  z_node broadcast,
  node_condition broadcast,
]
```

第一版 decoder 用 MLP 就够：

- `Linear(input, H)`
- `SiLU`
- residual MLP blocks
- `udf_head`
- `occ_head`

后续可升级成 query cross-attention decoder。

### 5.7 Geometry loss

```python
loss = (
    udf_weight * smooth_l1(pred_udf, target_udf)
    + occ_weight * binary_cross_entropy_with_logits(occ_logits, observed_occ)
    + kl_weight * kl_loss(mu, logvar)
)
```

默认权重：

- `udf_weight = 1.0`
- `occ_weight = 0.5`
- `kl_weight` 从 `0` warmup 到 `1e-4` 或 `1e-3`

KL warmup：

```text
first 10% steps: linear 0 -> target_kl_weight
afterwards: target_kl_weight
```

## 6. Stage B：Geometry + RGB Fused Octree Node VAE

Stage B 的目标是把 RGB 纳入同一个 node latent，而不是永久维护独立的几何 code 和颜色 code。

训练策略是：

1. 先加载 Stage A 的 geometry VAE checkpoint
2. 初始化 RGB encoder stem 和 color decoder head
3. 先冻结或半冻结 geometry encoder/decoder，让 RGB 分支学会读同一个 latent 空间
4. 再联合微调，输出一个 fused node latent

### 6.1 fused latent 定义

最终每个 node 仍然只有一个 latent：

```text
z_node = z_fused
```

这个 latent 同时服务两个 decoder：

```text
geometry_decoder(z_node, query_xyz) -> observed_occ, udf
color_decoder(z_node, query_xyz) -> rgb
```

也就是说，RGB 训练阶段可以有独立 RGB stem，但导出的 node token 不是 `geom_code + color_code` 两个主码，而是一个统一的 `main_code`。

### 6.2 Encoder fusion 结构

建议 encoder 分成三段：

```text
GeometryPointStem(xyz features) -> geo_point_features
RGBPointStem(rgb + xyz features) -> rgb_point_features
NodeConditionEncoder(metadata) -> node_query
FusedCrossAttention(node_query, geo_point_features, rgb_point_features) -> fused_feature
LatentHead(fused_feature) -> mu, logvar
```

第一版可以用最简单的 late fusion：

```python
geo_feature = geo_encoder(...)
rgb_feature = rgb_encoder(...)
fused_feature = fusion_mlp(torch.cat([geo_feature, rgb_feature, node_condition], dim=-1))
mu = mu_head(fused_feature)
logvar = logvar_head(fused_feature)
```

后续再升级为 gated fusion：

```python
gate = sigmoid(gate_mlp([geo_feature, rgb_feature, node_condition]))
fused_feature = geo_feature + gate * rgb_projection(rgb_feature)
```

这样做的好处是几何仍然是主干，RGB 以可控方式进入同一个 latent。

### 6.3 RGB decoder

RGB decoder 不在空域随机 query 上训练，只在 observed / near-surface query 上训练。

输入：

- `query_xyz`
- `z_node`
- optional geometry decoder feature
- node condition

输出：

- `rgb_pred`: `[B, Q_rgb, 3]`

监督：

```python
rgb_loss = masked_l1_or_mse(rgb_pred, target_rgb, query_rgb_mask)
```

建议：

- `rgb_weight = 0.05 ~ 0.1`
- RGB loss 只在 near-surface / observed point 附近计算
- geometry loss 继续保持主导，避免 RGB 噪声破坏结构 latent

### 6.4 Stage B loss

```python
loss = (
    udf_weight * udf_loss
    + occ_weight * observed_occ_loss
    + rgb_weight * rgb_loss
    + kl_weight * kl_loss
)
```

默认：

- `udf_weight = 1.0`
- `occ_weight = 0.5`
- `rgb_weight = 0.05`
- `kl_weight` 继承 Stage A，并可重新 warmup

### 6.5 Stage B checkpoint

Stage B checkpoint 是最终连续 VAE 的主 checkpoint，包含：

- geometry point stem
- RGB point stem
- fusion module
- shared latent head
- geometry decoder
- color decoder

后续 VQVAE 应该从 Stage B checkpoint 初始化，而不是从纯几何 Stage A checkpoint 初始化。

## 7. Stage C：Fused Octree Node VQVAE

Stage C 不改变 node 数量和树结构，只把 Stage B 的 fused continuous latent 量化。

### 7.1 模型关系

`OctreeNodeVQVAE` 复用 Stage B fused VAE 的 encoder/decoder：

```text
fused_encoder -> residual_latent_continuous -> vector_quantizer -> residual_latent_quantized -> geometry/color decoders
```

### 7.2 residual quantization

如果有 parent latent：

```text
residual = z_node_continuous - z_parent
residual_q, code_index, vq_loss = VQ(residual)
z_node_q = z_parent + residual_q
```

如果第一版没有 parent latent：

```text
residual = z_node_continuous
z_node_q = residual_q
```

### 7.3 codebook

建议：

- `codebook_size = 4096` 起步
- `embedding_dim = latent_dim`
- `commitment_cost = 0.25`

当前项目训练样本可能不大，`16384` 容易利用率低。先用 `1024/4096` 做实验更稳。

### 7.4 loss

```python
loss = geometry_recon_loss + rgb_weight * rgb_loss + vq_weight * vq_loss
```

默认：

- `vq_weight = 1.0`
- 如果 collapse 明显，降低到 `0.25`

### 7.5 checkpoint 迁移

训练 VQVAE 时加载 Stage B checkpoint：

- encoder weights
- decoder weights
- RGB branch weights
- fusion weights
- config

新增：

- vector quantizer weights

## 8. Token 和序列化

### 8.1 learned node code

现有接口：

- `NodeCodeProvider.encode_node(xyz_local, rgb) -> int`

建议新增接口：

```python
class NodeCodeProvider(Protocol):
    def encode_node(self, xyz_local: np.ndarray, rgb: np.ndarray) -> int:
        ...

class RichNodeCodeProvider(Protocol):
    def encode_node_record(self, node_sample: NodeEncodingInput) -> LearnedNodeCode:
        ...
```

第一版为了兼容，可以继续返回单个 int：

- continuous VAE：用 k-means/post-quant code 或临时 hash code
- fused VQVAE：返回 unified codebook index

最终导出的 `learned_code` 表示同一个 fused node latent，应该能够同时解码 geometry 和 RGB。

### 8.2 schema 扩展

建议新增：

```python
@dataclass(slots=True)
class LearnedNodeCode:
    main_code: int
    codebook_name: str = "octree_node_fused_vq"
    is_residual: bool = True
    decodes_geometry: bool = True
    decodes_rgb: bool = True
```

但第一版可以暂不改 schema，只把 VQVAE index 放入现有 `learned_code`。

### 8.3 DFS 与 BFS 双序列

保留：

- `scene_tokens_llm_sequence_latest.json`
- 当前 DFS / preorder 版本

新增：

- `scene_tokens_llm_sequence_bfs.json`
- `sequence_format = "instance_header_then_bfs_nodes_v1"`

BFS token：

```text
SCENE_START
INSTANCE_HEADER
INSTANCE_NODE_BFS { level, split_flag, child_mask, main_code }
INSTANCE_NODE_BFS ...
SCENE_END
```

BFS 用于 LLM/AR 训练，DFS 用于紧凑解析。

## 9. 训练脚本设计

### 9.1 train_octree_node_vae.py

参数：

```text
--ply-dir / --train-ply-dir / --val-ply-dir
--out
--points-per-node
--queries-per-node
--udf-truncation
--near-surface-threshold
--hidden-dim
--latent-dim
--semantic-dim
--epochs
--batch-size
--learning-rate
--kl-weight
--kl-warmup-ratio
--device
--include-leaf-only
```

默认：

- `points_per_node=128`
- `queries_per_node=128`
- `hidden_dim=256`
- `latent_dim=128`
- `batch_size=32` 如果显存允许

### 9.2 train_octree_node_vqvae.py

新增参数：

```text
--vae-checkpoint
--codebook-size
--commitment-cost
--vq-weight
```

默认：

- `codebook_size=4096`
- `commitment_cost=0.25`

## 10. 评估指标

Geometry VAE：

- `udf_smooth_l1`
- `observed_occ_bce`
- `kl_loss`
- `near_surface_precision`
- `near_surface_recall`

VQVAE：

- 上述 geometry 指标
- `vq_loss`
- `used_code_count`
- `usage_rate`
- `entropy_bits`
- `perplexity`

Token/structure：

- `avg_nodes_per_instance`
- `avg_tokens_per_instance`
- `points_per_token_ratio`
- `tokens_by_semantic`
- `tokens_by_depth`

RGB：

- `masked_rgb_l1`
- `masked_rgb_mse`

## 11. 实现顺序

### Step 1：数据字段扩展

修改：

- `src/threedvae/data/dataset.py`

添加：

- node center/size
- observed occupancy target
- collate 字段

测试：

- 更新 `tests/test_node_dataset.py`

### Step 2：Geometry VAE 模型

新增：

- `src/threedvae/models/octree_node_vae.py`
- `src/threedvae/train/octree_node_losses.py`

测试：

- forward shape test
- loss finite test
- CPU tiny batch backward test

### Step 3：Geometry VAE trainer/script

新增：

- `src/threedvae/train/octree_node_trainer.py`
- `scripts/train_octree_node_vae.py`

复用：

- 当前 dataset split 逻辑
- checkpoint manifest 风格

### Step 4：导出 learned geom code

新增：

- `src/threedvae/tokenizer/octree_node_encoder.py`

先支持 continuous checkpoint 的临时 code：

- nearest centroid 或 simple projection bin

VQVAE 完成后切到 codebook index。

### Step 5：Geometry VQVAE

新增：

- `src/threedvae/models/octree_node_vqvae.py`
- `scripts/train_octree_node_vqvae.py`

功能：

- load fused VAE checkpoint
- train VQ bottleneck
- export `learned_code`

### Step 6：BFS LLM sequence

修改：

- `src/threedvae/scene/serializer.py`
- `src/threedvae/data/schema.py`

新增：

- `build_llm_token_sequence_bfs_v1`

测试：

- BFS token count equals node count
- parent before child
- sequence can recover node ordering with child masks

### Step 7：RGB fusion

新增：

- RGB dataset query/mask
- `RGBPointStem`
- `FusedNodeEncoder`
- `OctreeNodeColorDecoder`
- RGB loss

训练：

- load geometry VAE checkpoint
- train RGB stem + color decoder first
- then joint finetune fused encoder with low RGB weight
- VQVAE should quantize the fused latent, not a separate color latent

## 12. 推荐第一版最小闭环

最小可用实现只做：

1. 扩展 node dataset：`node_center_local`, `node_size_local`, `query_occ`
2. 实现 `OctreeNodeVAE`
3. 用 `UDF + observed_occ + KL` 训练 geometry
4. 保留现有 token 导出不变
5. 训练确认 geometry loss 能下降

这个闭环完成后，再做 VQVAE 和 BFS 序列。

第二个闭环应该先做 RGB fused VAE，再做 fused VQVAE。这样最终离散 code 天然同时承载几何和颜色。

## 13. 不建议第一版做的事

1. 不做每 node 多 token
2. 不直接训练 RGB + geometry 早融合
3. 不要求 watertight occupancy
4. 不强制 parent latent cache 一次到位
5. 不重新引入旧 PointNetVQTokenizer 路线

## 14. 最终形态

最终模型应该形成一个统一主码：

### 14.1 Fused node token

```text
INSTANCE_HEADER:
  semantic_id
  center_xyz_q
  yaw_q
  box_size_xyz_q
  node_count
  main_code_scheme = "octree_node_fused_residual_vq"

INSTANCE_NODE:
  split_flag
  child_mask
  main_code
```

其中：

- `main_code` 是 residual VQ codebook index
- `main_code` 解码后得到 fused latent
- fused latent 同时输入 geometry decoder 和 color decoder
- geometry decoder 输出 observed occupancy / UDF
- color decoder 输出 near-surface RGB

这保持了 LLM 看到的主序列仍然是结构化八叉树，同时 learned code 从规则量化升级为可训练的几何+颜色统一残差码。
