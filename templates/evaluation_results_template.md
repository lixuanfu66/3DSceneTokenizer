# 实验结果模板

## 实验信息

- 实验名：
- checkpoint：
- 数据集：
- sample_unit：
- 备注：

## 重建指标

| 指标 | 数值 | 说明 |
|------|------|------|
| xyz_mse |  | 点坐标重建误差 |
| rgb_mse |  | 颜色重建误差 |
| chamfer_l2 |  | 点云双向最近邻误差 |
| udf_smooth_l1 |  | 节点级截断 UDF 误差 |

## 码本指标

| 指标 | 数值 | 说明 |
|------|------|------|
| code_count |  | 码本总大小 |
| used_code_count |  | 实际使用的码数量 |
| usage_rate |  | 使用率 |
| entropy_bits |  | 码分布熵 |
| perplexity |  | 感知有效码数量 |

## 压缩指标

| 指标 | 数值 | 说明 |
|------|------|------|
| sample_count |  | 评估样本数 |
| avg_input_points |  | 平均输入点数 |
| latent_tokens_per_sample |  | 每样本 latent token 数 |
| points_per_token_ratio |  | 点数/Token 压缩比 |

## 观察结论

- 
- 
- 

## 后续问题

- 
- 
- 
