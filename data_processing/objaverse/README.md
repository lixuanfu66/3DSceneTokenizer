# Objaverse 数据处理

这个目录保存 3dVAE object-level 实验使用的 Objaverse / Objaverse++ 数据准备流程。目标是把“质量筛选、类别筛选、资产下载、点云转换、结果检查”整理成一套可以在其他机器上复现的脚本和清单。

## 目录内容

- `prepare_objaversepp_quality_ply.py`：主脚本。读取 Objaverse++ 质量标注，可选结合 Objaverse LVIS 类别标注，下载原始 GLB，采样彩色点云，并导出 3dVAE 可读的 PLY。
- `prepare_objaverse_smoke_ply.py`：较早的 Smithsonian / Objaverse-XL 小样本 smoke test 路径。
- `build_objaverse_split_manifests.py`：从已准备好的数据中生成可迁移的 train/val object list。
- `validate_objaverse_ply_split.py`：检查某个 split 的 PLY 是否能被当前 3dVAE loader 读取。
- `manifests/`：轻量级对象清单，不包含 GLB 或 PLY 大文件，适合提交、同步和在其他机器上复现。

## 当前推荐划分

当前用于自动驾驶 / 室内机器人 object-level 实验的清单：

- 训练集：`manifests/objaversepp_task_relevant_train_objects.json`
- 验证集：`manifests/objaversepp_task_relevant_val_objects.json`
- 总览：`manifests/objaversepp_task_relevant_splits.json`

当前远端数据路径：

```text
train: /data/l00821447/3DSceneTokenizer/data/objaverse_pp_task_relevant/ply
val:   /data/l00821447/3DSceneTokenizer/data/objaverse_pp_task_relevant_val/ply
```

当前本地镜像路径：

```text
train: data/objaverse_pp_task_relevant/ply
val:   data/objaverse_pp_task_relevant_val/ply
```

## 筛选规则

基础质量过滤来自 Objaverse++：

```text
score >= 3
style in {realistic, scanned}
density in {mid, high}
排除 scene / multi-object / transparent / single-color
```

任务相关类别来自 Objaverse LVIS 标注，当前使用：

```text
--category-preset driving,indoor
```

它覆盖自动驾驶常见对象，例如车辆、自行车、摩托车、交通灯、停止牌、路牌、行人等；也覆盖室内机器人常见对象，例如椅子、桌子、办公桌、床、沙发、柜子、书架、键盘、鼠标、显示器、笔记本、电话、台灯等。

## 在其他机器上复现

先安装数据处理依赖：

```bash
pip install pandas pyarrow trimesh DracoPy
```

准备训练集：

```bash
PYTHONPATH=src python data_processing/objaverse/prepare_objaversepp_quality_ply.py \
  --out-dir data/objaverse_pp_task_relevant \
  --category-preset driving,indoor \
  --per-category-count 2 \
  --count 0 \
  --points-per-object 50000 \
  --normalize-unit-box \
  --require-color-variance \
  --min-color-std 3.0
```

准备验证集：

```bash
PYTHONPATH=src python data_processing/objaverse/prepare_objaversepp_quality_ply.py \
  --out-dir data/objaverse_pp_task_relevant_val \
  --category-preset driving,indoor \
  --per-category-count 2 \
  --start-index 2 \
  --count 0 \
  --target-success-count 20 \
  --points-per-object 50000 \
  --normalize-unit-box \
  --require-color-variance \
  --min-color-std 3.0
```

导出的 PLY schema 为：

```text
x y z red green blue instance semantic
```

文件名会带 LVIS 类别，便于检索，例如：

```text
000015_car__automobile_25c6874d173a4f6ca63e150bbd505686.ply
000084_traffic_light_8e6be3d334f84619a97eb90c5b20369b.ply
000025_desk_df1aa375f775420b90a20c1eaec9e016.ply
```

## 检查数据

检查训练集：

```bash
PYTHONPATH=src python data_processing/objaverse/validate_objaverse_ply_split.py \
  --ply-dir data/objaverse_pp_task_relevant/ply \
  --manifest data_processing/objaverse/manifests/objaversepp_task_relevant_train_objects.json
```

检查验证集：

```bash
PYTHONPATH=src python data_processing/objaverse/validate_objaverse_ply_split.py \
  --ply-dir data/objaverse_pp_task_relevant_val/ply \
  --manifest data_processing/objaverse/manifests/objaversepp_task_relevant_val_objects.json
```

检查脚本会验证：

- PLY 文件是否存在
- 是否能被 `load_ply_frame` 读取
- 点数是否为 `50000`
- RGB、semantic、instance 字段长度是否和点数一致
- 各类别数量统计

## 注意事项

- `data/` 目录被 `.gitignore` 忽略，只适合放下载后的 GLB、PLY 和中间 metadata。
- `manifests/` 中的 JSON/CSV 清单是轻量文件，适合在机器之间同步。
- 有些候选会因为 RGB 近似常量、资产过大、下载失败或 GLB 里不是可采样三角 mesh 而被跳过。
- 远端 A3 机器当前无法直接访问 Hugging Face，之前采用本机下载后同步到远端的方式落地数据。
