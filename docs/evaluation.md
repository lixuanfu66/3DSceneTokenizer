# 评估说明

## 当前已实现的评估范围

当前 `VQ-VAE` 评估脚本会输出三类核心指标：

- 重建指标
  - `xyz_mse`
  - `rgb_mse`
  - `chamfer_l2`
  - `udf_smooth_l1`（仅 node 模式）
- 码本指标
  - `code_count`
  - `used_code_count`
  - `usage_rate`
  - `entropy_bits`
  - `perplexity`
- 压缩指标
  - `sample_count`
  - `avg_input_points`
  - `latent_tokens_per_sample`
  - `points_per_token_ratio`

## 评估脚本

脚本路径：

- `.\scripts\evaluate_instance_tokenizer.py`

示例：

```powershell
set PYTHONPATH=%CD%\src
python .\scripts\evaluate_instance_tokenizer.py `
  --checkpoint D:\path\to\best.pt `
  --ply-dir D:\path\to\eval_ply `
  --out D:\path\to\eval_outputs `
  --sample-unit node
```

## 输出文件

评估输出目录会包含：

- `evaluation_metrics.json`
- `evaluation_summary.md`
- `evaluation_metrics.csv`

## 模板

结果表模板位于：

- `.\templates\evaluation_results_template.md`
- `.\templates\evaluation_results_template.csv`

## 当前边界

- 当前脚本重点覆盖“重建 + 码本 + 压缩”三类指标。
- 稳定性、检索、线性 probe、下游 VLA 任务评估还没有自动化实现。
- `UDF` 指标基于 partial 点云最近邻距离构造的 observed UDF baseline，不是严格的 `TSDF/SDF` 评估。
