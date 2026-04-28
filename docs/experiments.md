# Experiment Log

This document records improvement experiments in a stable structure. Each new experiment should include:

- Improvement goal
- Scheme design
- Experiment design
- Implementation
- Results
- Final conclusion
- Follow-up decision

The current validation set is intentionally small (`00015` to `00020`) and the source point clouds are noisy. Conclusions below should therefore be treated as engineering evidence plus theory-informed decisions, not as final generalization proof.

## EXP-001: Ascend 910B/C VAE Smoke Training

### Improvement Goal

Verify that Octree Node VAE training can run end-to-end on the Ascend A3 server with 910B/C NPUs and `torch_npu`.

### Scheme Design

Use the existing VAE training pipeline with a very small model and one epoch, then adapt only the minimum code required for NPU execution.

### Experiment Design

- Data: six debug PLY files, `00015` to `00020`.
- Environment: Docker container `claude_sz_1`, conda env `3dvae`.
- Device: `npu:8`.
- Model: `hidden_dim=64`, `latent_dim=32`.
- Training: `epochs=1`, `batch_size=4`, `points_per_node=32`, `queries_per_node=32`.

### Implementation

- Imported `torch_npu` opportunistically in `src/threedvae/utils/torch_compat.py`.
- Moved `_int_bits` feature generation to CPU before transferring the small bit feature tensor to NPU.
- Installed missing CANN/TE Python dependencies in the remote conda env.

### Results

- NPU probe succeeded: `torch_npu` available, `torch.npu.device_count() == 16`.
- Training finished successfully.
- Final tracked validation loss: `0.057689368877691605`.
- Checkpoints generated: `best.pt`, `latest.pt`, `checkpoint_epoch_1.pt`.

### Final Conclusion

The training pipeline can run on 910B/C after explicit `torch_npu` import and avoiding integer bit shifts on NPU.

### Follow-Up Decision

Keep NPU compatibility changes in the development branch and use the remote Docker environment for further training/debugging.

## EXP-002: Six-Frame VAE Overfit

### Improvement Goal

Check whether the current Octree Node VAE can memorize the six-frame debug set and produce meaningful UDF reconstruction signals.

### Scheme Design

Turn the setup into an overfit experiment by using all six frames for training, disabling KL regularization, and removing weight decay.

### Experiment Design

- Data: six PLY files, all used as training data.
- Training: `epochs=30`, `batch_size=64`.
- Model: `hidden_dim=128`, `latent_dim=64`.
- Sampling: `points_per_node=32`, `queries_per_node=32`.
- Loss: `kl_weight=0.0`, `occ_weight=0.05`.
- Device: `npu:8`.

### Implementation

Used `scripts/train_octree_node_vae.py` and evaluated `best.pt` with a temporary deterministic evaluation script.

### Results

Training curve:

| Metric | Epoch 1 | Epoch 30 |
|---|---:|---:|
| total_loss | `0.04729586285816518` | `0.025301645787445124` |
| udf_loss | `0.016270606598156868` | `0.0021992473229334153` |
| occ_loss | `0.6205051158437666` | `0.46204796176872504` |

Deterministic checkpoint evaluation on the same six frames:

| Metric | Value |
|---|---:|
| node samples | `9642` |
| query count | `308544` |
| UDF MAE | `0.04922004034474192` |
| UDF RMSE | `0.06598562407420061` |
| occupancy accuracy | `0.7853661066168844` |

### Final Conclusion

The model learns a usable UDF signal and can partially overfit the six-frame set, but it does not collapse to near-zero error. Occupancy is weaker than UDF and should remain auxiliary for now.

### Follow-Up Decision

Continue using UDF as the primary reconstruction metric. Investigate query density, input point count, and deterministic AE behavior later.

## EXP-003: Per-Frame Reconstruction Error Export

### Improvement Goal

Make VAE reconstruction analysis easier by exporting one PLY per frame and recording per-frame token count.

### Scheme Design

Evaluate the overfit checkpoint on query points, transform query coordinates back to ego/frame coordinates, and export one error-colored PLY per frame.

### Experiment Design

- Checkpoint: `outputs/vae_overfit_6ply_npu/best.pt`.
- Query setup: `queries_per_node=32`.
- Export: one PLY per frame plus JSON metrics.
- Token count definition: one token per octree node.

### Implementation

Created a temporary per-frame evaluation/export script. PLY fields included `pred_udf`, `target_udf`, `abs_error`, `pred_occ_prob`, `instance_id`, `node_id`, and `semantic_id`.

### Results

| Frame | Tokens / Nodes | Query Count | MAE | RMSE | Occ Accuracy |
|---|---:|---:|---:|---:|---:|
| `00015` | `1757` | `56224` | `0.04586305623576172` | `0.06338984818123257` | `0.7993205748434832` |
| `00016` | `1687` | `53984` | `0.046830298545923264` | `0.06405408113318595` | `0.794550237107291` |
| `00017` | `1603` | `51296` | `0.04589544966605132` | `0.06317835961273291` | `0.7991071428571429` |
| `00018` | `1318` | `42176` | `0.0542469914633244` | `0.06959734675068265` | `0.7719793247344461` |
| `00019` | `1581` | `50592` | `0.04708431982640725` | `0.06400944827336807` | `0.7915480708412397` |
| `00020` | `1696` | `54272` | `0.05630146032878124` | `0.07178785678457289` | `0.7534271816037735` |
| Total | `9642` | `308544` | - | - | - |

### Final Conclusion

The per-frame export is useful for debugging, but query points are not reconstructed surface points. They visualize UDF error on sampled evaluation locations.

### Follow-Up Decision

Design a separate dense surface reconstruction query pipeline.

## EXP-004: Training Query Strategy Ablation

### Improvement Goal

Understand whether increasing training query count or using layered query sampling improves overfit/reconstruction behavior.

### Scheme Design

Compare fixed `uniform` query sampling against `layered` sampling. Use the same model and training schedule, then evaluate all checkpoints on the same `uniform q=128` distribution.

### Experiment Design

- Data: six PLY files, all used as training data.
- Training: `epochs=12`, `batch_size=64`.
- Model: `hidden_dim=128`, `latent_dim=64`.
- Loss: `kl_weight=0.0`, `occ_weight=0.05`.
- Evaluation distribution: `uniform / queries_per_node=128`.

Experiment groups:

| Group | queries_per_node | query_strategy | Notes |
|---|---:|---|---|
| A | `32` | `uniform` | low-query baseline |
| B | `128` | `uniform` | denser supervision |
| C | `128` | `layered` | first layered version with expensive sparse hard mining |
| C-fast | `128` | `layered` | cheaper edge-biased hard sampling |

### Implementation

- Added `query_strategy` support to `TreeNodePointCloudDataset`.
- Added `--query-strategy` to VAE/VQVAE training scripts.
- Added `tests/test_query_sampling.py`.

### Results

Final training metrics:

| Group | total_loss | udf_loss | occ_loss |
|---|---:|---:|---:|
| A | `0.02623406584245085` | `0.002343879013464151` | `0.4778037290304702` |
| B | `0.02635727111028125` | `0.0023526183070479243` | `0.48009304771360184` |
| C | `0.026352301239967346` | `0.0017574721737610584` | `0.491896574860377` |
| C-fast | `0.026122533537398112` | `0.001666428664674526` | `0.4891220921317473` |

Unified evaluation metrics:

| Group | UDF MAE | UDF RMSE | smooth L1 | occ_accuracy |
|---|---:|---:|---:|---:|
| A | `0.0517743440231726` | `0.06793399240288212` | `0.0023075136723291626` | `0.7641543831673927` |
| B | `0.051449388766333266` | `0.06820015037408847` | `0.002325630253302565` | `0.7715990263949388` |
| C | `0.06014100482443561` | `0.08231158400919268` | `0.003387598446795392` | `0.7164083566687409` |
| C-fast | `0.05839194018324983` | `0.0812703193407691` | `0.003302432397466243` | `0.7273808597801286` |

### Final Conclusion

`layered` sampling lowers training-distribution UDF loss but performs worse under a shared `uniform q=128` evaluation distribution, indicating distribution shift. `uniform q=128` does not dramatically beat `uniform q=32` on this tiny/noisy dataset, but is theoretically safer for larger nodes and downstream dense surface export.

### Follow-Up Decision

Use `query_strategy=uniform` and `queries_per_node=128` as the current default training supervision setup. Keep `layered` as an experiment option only.

## EXP-005: Dense Surface Query Export

### Improvement Goal

Improve visual reconstruction quality by separating surface reconstruction queries from training/evaluation queries.

### Scheme Design

Generate dense candidate query points per octree node, run the VAE decoder, and keep predicted near-surface points using `pred_udf` threshold plus a small top-k fallback.

### Experiment Design

- Checkpoint: overfit `best.pt`.
- Candidate count: `512` per node.
- Threshold: `pred_udf <= 0.03`.
- Fallback: `topk_per_node=4`, `topk_max_udf=0.08`.
- Strategies:
  - `hybrid`: 45% bbox uniform + 40% observed-near + 15% split-plane.
  - `uniform`: 100% bbox uniform.

### Implementation

Added `scripts/export_octree_vae_surface.py` to export predicted surface PLY files and `surface_export_metrics.json`.

### Results

| Strategy | total_candidates | total_kept_points | Notes |
|---|---:|---:|---|
| `hybrid512` | `4936704` | `1242151` | higher near-surface hit rate |
| `uniform512` | `4936704` | `1087099` | more objective field probing |

### Final Conclusion

Dense reconstruction queries should be independent of training `queries_per_node`. `hybrid512` is better for visual debugging; `uniform512` is better for unbiased field inspection.

### Follow-Up Decision

Use `hybrid512` for quick visual surface reconstruction and keep `uniform512` as a stricter comparison mode.

## EXP-006: Node-Size Adaptive Surface Candidate Count

### Improvement Goal

Improve reconstruction density in large nodes without oversampling every node equally.

### Scheme Design

Scale per-node candidate count by node bbox surface area:

```text
raw = candidates_per_node * sqrt(surface_area / reference_node_area)
```

Then bucket and clamp the result to keep batch execution tractable.

### Experiment Design

- Base strategy: `hybrid`.
- Base candidates: `512`.
- Threshold: `pred_udf <= 0.03`.
- Fallback: `topk_per_node=4`, `topk_max_udf=0.08`.
- Compare fixed count, aggressive adaptive, gentle adaptive, and quality-oriented adaptive.

### Implementation

Updated `scripts/export_octree_vae_surface.py`:

- Added `--adaptive-candidates`.
- Added `--min-candidates-per-node`, `--max-candidates-per-node`, `--reference-node-area`, and `--candidate-bucket-size`.
- Batched nodes by candidate-count bucket to preserve dense tensors.

### Results

| Strategy | Parameters | total_candidates | total_kept_points | Result |
|---|---|---:|---:|---|
| fixed hybrid512 | `512/node` | `4936704` | `1242151` | balanced baseline |
| adaptive aggressive | `min=128 max=2048 ref_area=1.0` | `13433472` | `1421175` | too many candidates, limited gain |
| adaptive gentle | `min=128 max=1024 ref_area=4.0` | `6771584` | `765694` | undersamples small/medium nodes |
| adaptive quality | `min=512 max=2048 ref_area=4.0` | `11980544` | `1649892` | densest useful export |

### Final Conclusion

Adaptive candidate count is useful when it preserves a strong per-node lower bound. Reducing small/medium nodes below 512 hurts surface coverage. The useful version is quality-oriented: keep 512 as the floor and only add candidates for large nodes.

### Follow-Up Decision

Use fixed `hybrid512` for fast debugging and adaptive `min512/ref4/max2048` when inspecting reconstruction quality in large nodes.
