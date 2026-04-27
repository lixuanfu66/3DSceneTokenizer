# 3DSceneTokenizer

用于自动驾驶场景单帧点云编码实验的最小代码骨架。

当前已实现：

- 单帧 `PLY` 解析
- 实例聚合
- ground-aligned OBB 估计
- 实例局部坐标变换
- 规则式 octree 构建
- 基础实例离散编码
- 场景级 token 打包
- 场景级 bbox mesh `PLY` 导出
- 节点级调试记录导出接口
- octree node VAE/VQVAE tokenizer 训练代码骨架

## 运行单帧 pipeline

```bash
set PYTHONPATH=%CD%\src
python .\scripts\export_scene_tokens.py --ply D:\path\to\frame.ply --out D:\path\to\outputs
```

输出目录会包含：

- `scene_instance_bboxes.ply`
- `instance_<id>_bbox.json`
- `instance_<id>_octree_nodes.jsonl`
- `scene_tokens.json`

## 训练 octree node VAE/VQVAE tokenizer

训练代码基于 PyTorch，默认不会随基础依赖自动安装。

```bash
pip install -e .[train]
set PYTHONPATH=%CD%\src
python .\scripts\train_octree_node_vae.py --ply-dir D:\path\to\ply_dir --out D:\path\to\vae_outputs
python .\scripts\train_octree_node_vqvae.py --vae-checkpoint D:\path\to\vae_outputs\best.pt --ply-dir D:\path\to\ply_dir --out D:\path\to\vqvae_outputs
```

训练输出目录会包含：

- `checkpoint_epoch_*.pt`
- `history.json`
- `trainer_config.json`
