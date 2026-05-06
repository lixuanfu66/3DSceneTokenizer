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

## EXP-007: Objaverse-XL Smithsonian Smoke Data

### Improvement Goal

Check whether a small amount of object-level Objaverse-XL data can be downloaded and converted into the same PLY schema used by the current 3dVAE pipeline.

### Scheme Design

Use the Objaverse-XL Hugging Face metadata for the Smithsonian subset, download a few CC0 GLB assets, sample mesh surfaces, normalize each object to a unit box, and export `x y z red green blue instance semantic` PLY files.

### Experiment Design

- Source metadata: `allenai/objaverse-xl`, `smithsonian/smithsonian.parquet`.
- Data count: 3 objects.
- Samples: 50,000 surface points per object.
- Output schema: same PLY fields as Bench2Drive debug data.
- Pipeline check: run `scripts/export_scene_tokens.py` on each exported PLY.

### Implementation

Added `scripts/prepare_objaverse_smoke_ply.py`.

The Smithsonian GLBs use `KHR_draco_mesh_compression`, so plain `trimesh.load()` produced invalid all-zero geometry. The script now parses GLB chunks, decodes Draco primitives with `DracoPy`, builds `trimesh.Trimesh` objects, samples surfaces, and writes compatible PLY files.

### Results

Generated files:

- PLY directory: `data/objaverse_xl_smoke/ply`
- Manifest: `data/objaverse_xl_smoke/manifest.json`
- Scene token smoke outputs: `data/objaverse_xl_smoke/scene_tokens_smoke`

Validation summary:

| Object | Points | XYZ range after normalization | Semantic | Instance |
|---|---:|---|---:|---:|
| `Pan_troglodytes_verus_Talus_Left` | 50000 | within unit box | 100 | 1 |
| `Blue_Crab` | 50000 | within unit box | 100 | 2 |
| `George_Washington` | 50000 | within unit box | 100 | 3 |

`python -m unittest tests.test_bench2drive_rgbd tests.test_ply_loader` passed, and `export_scene_tokens.py` succeeded for all three PLY files.

### Final Conclusion

Objaverse-XL can provide cleaner instance-level geometry for validating VAE convergence before returning to noisy Bench2Drive scenes. The current smoke data is suitable for geometry-only VAE debugging.

### Follow-Up Decision

Scale this path from 3 objects to 100-1000 objects, prioritize geometry-only VAE convergence first, then add better texture/color sampling for fused RGB experiments.

## EXP-008: Objaverse++ Quality-Filtered Object Smoke Data

### Improvement Goal

Use Objaverse++ quality annotations to select cleaner Objaverse objects for object-level VAE convergence validation.

### Scheme Design

Filter Objaverse++ UID annotations with a strict quality rule, map selected UIDs to original Objaverse GLB paths via `object-paths.json.gz`, download the assets, sample mesh surfaces, normalize each object to a unit box, and export the same PLY schema used by the current 3dVAE pipeline.

### Experiment Design

- Annotation source: `cindyxl/ObjaversePlusPlus`, `annotated_800k.json`.
- Asset source: `allenai/objaverse`, `object-paths.json.gz` plus `glbs/.../*.glb`.
- Strict filter: `score>=3`, `style in {realistic, scanned}`, `density in {mid, high}`, and not scene/multi-object/transparent/single-color.
- Smoke count: 20 objects.
- Points per object: 50,000.
- Validation: remote PLY loader check for all 20 objects and remote scene token export for the first 5 objects.

### Implementation

Added `scripts/prepare_objaversepp_quality_ply.py`.

The remote host and container could not reach Hugging Face directly (`Network is unreachable`), so the first smoke batch was downloaded and converted locally, then synchronized to `/data/l00821447/3DSceneTokenizer/data/objaverse_pp_quality` on the remote debug machine.

### Results

The strict filter produced `79221` candidate objects from `789195` annotated Objaverse entries.

Generated remote data:

- `/data/l00821447/3DSceneTokenizer/data/objaverse_pp_quality/assets`
- `/data/l00821447/3DSceneTokenizer/data/objaverse_pp_quality/ply`
- `/data/l00821447/3DSceneTokenizer/data/objaverse_pp_quality/selected_quality_manifest.json`
- `/data/l00821447/3DSceneTokenizer/data/objaverse_pp_quality/prepared_quality_manifest.json`

Validation:

| Check | Result |
|---|---:|
| Prepared assets | `20/20` |
| PLY files readable remotely | `20` |
| Points per object | `50000` |
| Scene token exports | first `5` succeeded |

### Final Conclusion

Objaverse++ gives a much better object-level training pool than random Objaverse sampling. The strict subset is large enough for the next 100-1000 object geometry-only VAE training run.

### Follow-Up Decision

Use this strict Objaverse++ subset as the next object-level VAE data source. Keep Bench2Drive for later scene-level fine-tuning and integration.

## EXP-009: Task-Relevant Objaverse++ Object Data

### Improvement Goal

Bias object-level VAE training data toward autonomous-driving and indoor-robotics objects instead of generic high-quality Objaverse assets.

### Scheme Design

Intersect Objaverse++ quality annotations with Objaverse LVIS category annotations. Use explicit category whitelists for driving and indoor objects, sample a small balanced set per category, require non-constant RGB, and put the category name into each output file name.

### Experiment Design

- Quality source: Objaverse++ `annotated_800k.json`.
- Category source: Objaverse `lvis-annotations.json.gz`.
- Asset path source: Objaverse `object-paths.json.gz`.
- Category preset: `driving,indoor`.
- Sampling: up to 2 candidates per LVIS category.
- Color filter: mean RGB channel std >= `3.0`.
- Points per object: 50,000.
- Remote validation: read all PLY files and run scene token export on representative categories.

### Implementation

Updated `scripts/prepare_objaversepp_quality_ply.py`:

- Added LVIS category presets.
- Added `--per-category-count`.
- Added category-prefixed asset/PLY names.
- Added `--require-color-variance`.

### Results

Generated data:

- Local: `data/objaverse_pp_task_relevant`
- Remote: `/data/l00821447/3DSceneTokenizer/data/objaverse_pp_task_relevant`

Summary:

| Check | Result |
|---|---:|
| Strict quality + category candidates | `406` |
| Balanced selected candidates | `92` |
| Successful colored PLY files | `80` |
| Points per PLY | `50000` |
| Remote loader check | `80/80` |
| Remote scene-token smoke export | `6/6` representative categories |

Representative exported categories include `car_(automobile)`, `traffic_light`, `stop_sign`, `person`, `desk`, `sofa`, `lamp`, `computer_keyboard`, `motorcycle`, and multiple truck/bus variants.

### Final Conclusion

The task-relevant filtered subset is a better next object-level VAE dataset than generic high-quality Objaverse sampling because it matches the downstream autonomous-driving and indoor-robotics domains while preserving basic quality and color requirements.

### Follow-Up Decision

Use `data/objaverse_pp_task_relevant/ply` for the next geometry/RGB object-level VAE smoke training. Later expand per category and relax filtering only for rare safety-critical classes such as `person` and traffic signs.

## EXP-010: Task-Relevant Objaverse++ Validation Split

### Improvement Goal

Create a held-out validation/test split from the same task-relevant Objaverse++ candidate pool, without reusing the first training candidates.

### Scheme Design

Use the same quality, category, and color filters as the training split, but start from the third candidate in each category. Continue through candidates until 20 successful colored PLY files are produced.

### Experiment Design

- Source pool: strict quality + `driving,indoor` LVIS category candidates.
- Split rule: `--start-index 2` with `--per-category-count 2`.
- Target success count: 20.
- Points per object: 50,000.
- Remote validation: read all PLY files and export scene tokens for representative samples.

### Implementation

Updated `scripts/prepare_objaversepp_quality_ply.py` with `--target-success-count`.

### Results

Generated data:

- Local: `data/objaverse_pp_task_relevant_val`
- Remote: `/data/l00821447/3DSceneTokenizer/data/objaverse_pp_task_relevant_val`

Summary:

| Check | Result |
|---|---:|
| Candidate rows | `64` |
| Attempted rows | `21` |
| Successful colored PLY files | `20` |
| Points per PLY | `50000` |
| Remote loader check | `20/20` |
| Remote scene-token smoke export | `3/3` representative samples |

### Final Conclusion

The project now has a small but clean task-relevant object-level train/validation pair: 80 training PLY files and 20 held-out validation PLY files.

### Follow-Up Decision

Use `data/objaverse_pp_task_relevant/ply` for training and `data/objaverse_pp_task_relevant_val/ply` for validation in the next object-level VAE run.

## EXP-011: Object-Level Adaptive Octree Preset

### Improvement Goal

Evaluate whether the existing CARLA semantic octree policy is reasonable for high-quality complete object-level data, and define a better object-level preset if needed.

### Scheme Design

Compare two policies on the same 80/20 task-relevant Objaverse++ split:

- `carla`: existing scene-level policy table. Unknown semantic `100` falls back to a shallow object/region policy.
- `object`: a dedicated complete-object preset where semantic `100` uses object-priority splitting up to depth `4`.

### Experiment Design

- Train set: `data/objaverse_pp_task_relevant/ply`, 80 PLY files.
- Validation set: `data/objaverse_pp_task_relevant_val/ply`, 20 PLY files.
- Points per PLY: 50,000.
- Evaluation script: `scripts/evaluate_octree_dataset.py`.
- Metrics: node/token count, max depth, leaf ratio, semantic histogram.

### Implementation

Added `build_default_object_semantic_policies()` and `OctreeBuildConfig.with_default_object_semantics()` in `src/threedvae/octree/split_policy.py`.

Added `--octree-preset {carla,object}` to the octree evaluation/training scripts so the same data can be evaluated under both policies.

### Results

CARLA preset:

| Split | Objects | Total Tokens | Mean Tokens/Object | Median | Max Depth |
|---|---:|---:|---:|---:|---:|
| train80 | 80 | 4,245 | 53.06 | 57.0 | 2 |
| val20 | 20 | 1,049 | 52.45 | 51.5 | 2 |

Object preset:

| Split | Objects | Total Tokens | Mean Tokens/Object | Median | Max Depth |
|---|---:|---:|---:|---:|---:|
| train80 | 80 | 109,382 | 1,367.28 | 1,337.5 | 4 |
| val20 | 20 | 27,465 | 1,373.25 | 1,234.0 | 4 |

### Final Conclusion

The CARLA preset is too coarse for complete object-level data because semantic `100` is treated as an unknown fallback and stops at depth `2`. The object preset is much more suitable for complete mesh-derived objects because it exposes shape detail at depth `4`, with about 1.3k nodes per object.

### Follow-Up Decision

Use `--octree-preset object` for object-level VAE training and keep `--octree-preset carla` for Bench2Drive/self-driving partial scene data.

## EXP-012: Object-Level Geometry VAE

### Improvement Goal

Train a geometry-only VAE on the 80 high-quality task-relevant objects and evaluate whether it generalizes to the 20 held-out object-level samples.

### Scheme Design

Use the object octree preset from EXP-011. Train a node-level geometry VAE with UDF reconstruction, soft occupancy auxiliary loss, and a small KL weight. Keep RGB disabled to isolate geometric capacity.

### Experiment Design

- Train set: 80 object PLY files.
- Validation set: 20 object PLY files.
- Octree preset: `object`.
- Points per node: `64`.
- Queries per node: `64`.
- Query strategy: `uniform`.
- Model: hidden dim `128`, latent dim `64`, attention heads `4`.
- Training: `8` epochs, batch size `128`, learning rate `5e-4`, weight decay `1e-4`, KL weight `1e-5`, occupancy weight `0.05`.
- Device: Ascend NPU `npu:8`.
- Checkpoint evaluation: deterministic validation pass over all `27,465` val nodes and `1,757,760` query points.

### Implementation

Updated `scripts/train_octree_node_vae.py` to accept `--octree-preset object`.

Added `scripts/evaluate_octree_vae_checkpoint.py` for deterministic checkpoint evaluation on a PLY directory.

### Results

Training summary:

| Metric | Epoch 1 | Epoch 8 |
|---|---:|---:|
| train total loss | 0.024038 | 0.019222 |
| train UDF loss | 0.003749 | 0.000021 |
| train occupancy loss | 0.405724 | 0.383773 |
| val total loss | 0.019361 | 0.018669 |
| val UDF loss | 0.000032 | 0.000017 |
| val occupancy loss | 0.386474 | 0.372755 |

Deterministic validation checkpoint metrics:

| Metric | Value |
|---|---:|
| val nodes | 27,465 |
| val queries | 1,757,760 |
| UDF MAE | 0.003310 |
| UDF RMSE | 0.005911 |
| UDF max abs | 0.304473 |
| Occupancy accuracy | 0.988322 |

### Final Conclusion

The geometry-only VAE converges cleanly on high-quality object-level data. The held-out UDF MAE is about `0.0033` in the normalized object coordinate scale, and the occupancy auxiliary head reaches about `98.8%` accuracy. This is a much healthier convergence signal than the earlier noisy partial self-driving point cloud experiments.

### Follow-Up Decision

Use this geometry-only run as the object-level baseline. Next geometry improvements should focus on reconstruction sampling/export quality, longer training, and capacity ablations rather than diagnosing basic convergence.

## EXP-013: Object-Level Geometry+Color VAE

### Improvement Goal

Train a fused geometry+color VAE on the same object-level split and evaluate whether a single node latent can jointly encode UDF geometry and RGB.

### Scheme Design

Start from the geometry-only setup in EXP-012 and enable RGB point fusion plus RGB query prediction. Keep the same latent size and training schedule so the cost of adding color is visible.

### Experiment Design

- Train/validation data: same 80/20 object split.
- Octree preset: `object`.
- Points per node: `64`.
- Queries per node: `64`.
- Query strategy: `uniform`.
- Model: hidden dim `128`, latent dim `64`, attention heads `4`.
- RGB options: `--use-rgb-fusion --predict-rgb --rgb-weight 1.0`.
- Training: `8` epochs, batch size `128`, learning rate `5e-4`, weight decay `1e-4`, KL weight `1e-5`, occupancy weight `0.05`.
- Device: Ascend NPU `npu:8`.

### Implementation

Used the same `scripts/train_octree_node_vae.py` path with RGB fusion and prediction enabled. Evaluated with `scripts/evaluate_octree_vae_checkpoint.py`, which reports RGB MAE/MSE/PSNR when the checkpoint contains an RGB head.

### Results

Training summary:

| Metric | Epoch 1 | Epoch 8 |
|---|---:|---:|
| train total loss | 0.055255 | 0.031915 |
| train UDF loss | 0.003695 | 0.000027 |
| train occupancy loss | 0.411341 | 0.393254 |
| train RGB loss | 0.030965 | 0.012194 |
| val total loss | 0.037084 | 0.030980 |
| val UDF loss | 0.000044 | 0.000025 |
| val occupancy loss | 0.388947 | 0.385227 |
| val RGB loss | 0.017534 | 0.011666 |

Deterministic validation checkpoint metrics:

| Metric | Value |
|---|---:|
| val nodes | 27,465 |
| val queries | 1,757,760 |
| UDF MAE | 0.003889 |
| UDF RMSE | 0.007036 |
| UDF max abs | 0.233344 |
| Occupancy accuracy | 0.987787 |
| RGB MAE | 0.061589 |
| RGB MSE | 0.011695 |
| RGB PSNR | 19.32 dB |

### Final Conclusion

The fused geometry+color VAE also converges, and RGB MSE drops substantially from the first epoch. Geometry is slightly worse than the geometry-only baseline, which is expected because the same latent capacity now carries both geometry and appearance. The current color quality is usable as a first fused baseline but not yet final.

### Follow-Up Decision

Keep geometry-only as the clean geometry baseline and use geometry+color as the first fused baseline. Next fused-model experiments should test longer training, larger latent/hidden capacity, and staged RGB weighting so geometry does not regress while color improves.

## EXP-014: RGB VAE Test-Set Surface Export

### Improvement Goal

Export visual PLY reconstructions from the geometry+color VAE on the 20 held-out object-level test samples so predicted RGB point clouds can be compared against the original colored PLY files.

### Scheme Design

Use the RGB VAE checkpoint from EXP-013. Query each object octree node with dense hybrid candidates, keep surface candidates according to predicted UDF, and color each kept point with the model-predicted RGB instead of UDF debug colors.

### Experiment Design

- Checkpoint: `outputs/object_level_geo_rgb_vae_e8/best.pt`.
- Test set: `data/objaverse_pp_task_relevant_val/ply`.
- Octree preset: `object`.
- Points per node: `64`.
- Candidate queries per node: `512`.
- Candidate strategy: `hybrid`.
- Color mode: `predicted-rgb`.
- First export threshold: `pred_udf < 0.03`.
- Final visual export threshold: `pred_udf < 0.003`, with `topk_per_node=16` and `topk_max_udf=0.02`.

### Implementation

Updated `scripts/export_octree_vae_surface.py`:

- Added `--octree-preset`.
- Added `--color-mode predicted-rgb`.
- Uses `rgb_query_xyz` so RGB prediction is evaluated on the same candidate surface points.

### Results

The loose `pred_udf < 0.03` export kept `14,012,310 / 14,062,080` candidate points, or about `99.6%`. This is too dense for surface visualization and indicates that the RGB VAE UDF output is under-calibrated for naive thresholding.

The stricter export produced:

| Metric | Value |
|---|---:|
| Test objects | 20 |
| Total octree nodes/tokens | 27,465 |
| Total candidate queries | 14,062,080 |
| Kept visual points | 4,169,763 |

Local output:

`outputs/object_level_geo_rgb_vae_e8/val20_pred_rgb_surface_hybrid512_udf0003_top16`

### Final Conclusion

The RGB VAE can export colored reconstructions for every held-out object, but the UDF head is not yet well calibrated as a direct surface filter. A stricter threshold plus per-node top-k fallback gives more usable visual PLY files for manual inspection.

### Follow-Up Decision

Use the strict export for visual comparison now. For the next model iteration, add explicit UDF calibration/surface sampling metrics and consider training/exporting with a surface-focused query distribution.

## EXP-015: Low-Point Voxel Pruning

### Improvement Goal

Remove unreliable low-point voxels before they become octree tokens, and ensure parent `child_mask` flags exactly match the materialized child node list.

### Scheme Design

Move sparse voxel pruning into octree construction instead of only filtering after tokenization:

- Stop subdivision when a node has fewer than `10` points.
- When splitting a parent, drop any child with fewer than `4` points.
- Recompute the parent `child_mask` only from children that survive this pruning.
- Keep `--min-node-points` as an extra train/eval/export filter for explicit leaf-token experiments.

### Experiment Design

- Dataset: 20 held-out task-relevant Objaverse++ object PLY files.
- Octree preset: `object`.
- BBox export: leaf-only and occupied-only.
- Reconstruction export: RGB VAE checkpoint, leaf-only, `min_node_points=4`, `512` hybrid candidates per node, `pred_udf < 0.003`, `topk_per_node=16`.
- Verification: remote full unittest suite.

### Implementation

Updated:

- `OctreeBuildConfig.min_points_per_leaf`
- `OctreeBuildConfig.with_default_object_semantics()` defaults:
  - `min_points_per_node=10`
  - `min_points_per_leaf=4`
- `build_instance_octree()` now prunes children with fewer than `min_points_per_leaf` points before setting parent `child_mask`.
- VAE train/eval/export scripts now expose `--min-node-points`.
- Added tests for low-count split stopping, object preset defaults, and child-mask pruning consistency.

### Results

Tests:

| Check | Result |
|---|---:|
| Remote targeted unittest | `13` tests OK |
| Remote full unittest | `47` tests OK |

Leaf occupied bbox comparison:

| Export | Leaf Occupied Nodes |
|---|---:|
| Before construct-time child pruning | 22,098 |
| After construct-time child pruning | 21,417 |

New local bbox output:

`outputs/object_level_geo_rgb_vae_e8/val20_octree_leaf_occupied_minleaf4_node_bboxes`

New local RGB reconstruction output:

`outputs/object_level_geo_rgb_vae_e8/val20_pred_rgb_surface_leaf_minleaf4_hybrid512_udf0003_top16`

Reconstruction export summary:

| Metric | Value |
|---|---:|
| Leaf tokens/nodes | 21,417 |
| Candidate queries | 10,965,504 |
| Kept visual points | 3,826,501 |

### Final Conclusion

The sparse voxel handling now happens at the structural octree level. Low-count children are no longer silently removed after tokenization, so parent `child_mask` and the actual node list remain consistent. This also reduces unreliable leaf tokens and candidate reconstruction points.

### Follow-Up Decision

Use construct-time pruning for all object-level runs. Retrain the RGB VAE with `--include-leaf-only --min-node-points 4` so the checkpoint distribution matches the new leaf-token reconstruction path.

## EXP-016: Leaf-Min4 Layered RGB VAE Retraining

### Improvement Goal

Reduce false positive reconstructed points inside leaf voxels by matching the VAE training distribution to the new leaf-only octree token distribution and by adding node-bbox volume/hard query supervision.

### Scheme Design

The previous RGB VAE was trained on all nodes with `uniform` queries sampled around the observed point bbox. Surface export queried the full octree node bbox, creating a distribution mismatch. This experiment retrains with:

- `--include-leaf-only`
- `--min-node-points 4`
- object preset with construct-time sparse child pruning
- `query_strategy=layered`
- `queries_per_node=128`

### Experiment Design

- Train set: 80 task-relevant Objaverse++ object PLY files.
- Validation set: 20 held-out object PLY files.
- Model: RGB-fused VAE, hidden dim `128`, latent dim `64`.
- Training: 8 epochs, batch size `128`, learning rate `5e-4`.
- Export: `256` hybrid candidates per leaf node, `pred_udf < 0.003`, `topk_per_node=4`, predicted RGB colors.

### Implementation

No new model architecture was introduced. The experiment uses the new structural pruning and existing layered query sampling, plus the newly exposed `--include-leaf-only` and `--min-node-points` options.

### Results

Checkpoint:

`outputs/object_level_geo_rgb_vae_leaf_min4_layered_e8/best.pt`

Validation metrics:

| Metric | Value |
|---|---:|
| sample_count | 21,417 |
| query_count | 2,741,376 |
| UDF MAE | 0.005828 |
| UDF RMSE | 0.009292 |
| Occupancy accuracy | 0.924896 |
| RGB MAE | 0.061976 |
| RGB MSE | 0.010682 |
| RGB PSNR | 19.71 dB |

Visual export:

`outputs/object_level_geo_rgb_vae_leaf_min4_layered_e8/val20_pred_rgb_surface_leaf_min4_hybrid256_udf0003_top4`

| Export | Kept Points |
|---|---:|
| Old leaf/min4 export with old checkpoint | 3,826,501 |
| New leaf/min4 layered checkpoint export | 88,462 |

Most frames now keep only the top-k fallback points; `pred_udf < 0.003` threshold points are often zero.

### Final Conclusion

Layered training substantially reduces voxel-filling false positives during visual export, but the model is now conservative and UDF scale is not well calibrated for direct thresholding. Visual quality is still limited because the decoder is trained as a per-node field with weak global surface consistency and the export is still candidate-selection based.

### Follow-Up Decision

Next steps should focus on UDF calibration and reconstruction quality rather than more voxel pruning: increase training duration/capacity, add stronger full-node negative supervision, export by calibrated per-node quantile/top-k, and compute visual reconstruction metrics such as Chamfer/F-score before deciding the final surface extraction rule.

## EXP-017: Calibrated Surface Query Supervision

### Improvement Goal

Improve UDF thresholdability so surface reconstruction can use a direct `pred_udf < threshold` rule instead of Top-K or spatial fallback.

### Scheme Design

Add a new `query_strategy=calibrated_surface` with explicit query mixture:

- Covered surface positives from spatially distributed anchors on node points.
- Near-surface band queries with controlled offsets.
- Full node-bbox volume negatives.
- Hard negatives on split planes, node boundaries, and near-surface bands.

This keeps the same VAE architecture and changes only the supervision query distribution.

### Experiment Design

- Data: 80 train / 20 validation task-relevant Objaverse++ object PLY files.
- Octree: object preset, leaf-only, `min_node_points=4`.
- Model: RGB-fused VAE, hidden dim `128`, latent dim `64`.
- Training: 8 epochs, `queries_per_node=128`, `query_strategy=calibrated_surface`.
- Export: no Top-K fallback, `topk_per_node=0`.
- Thresholds tested: `0.003`, `0.005`, `0.01`.

### Implementation

Updated `src/threedvae/data/dataset.py` with `build_calibrated_surface_udf_queries()`.

Updated train/eval scripts to accept `calibrated_surface`.

Added a sampling unit test in `tests/test_query_sampling.py`.

### Results

Validation metrics:

| Metric | Value |
|---|---:|
| sample_count | 21,417 |
| query_count | 2,741,376 |
| UDF MAE | 0.006423 |
| UDF RMSE | 0.014231 |
| Occupancy accuracy | 0.956074 |
| RGB MAE | 0.059363 |
| RGB MSE | 0.010476 |
| RGB PSNR | 19.80 dB |

Threshold-only exports:

| Threshold | Top-K | Kept Points | Retention |
|---:|---:|---:|---:|
| 0.003 | 0 | 2,536,473 | 23.13% |
| 0.005 | 0 | 5,589,402 | 50.97% |
| 0.010 | 0 | 9,194,812 | 83.85% |

Local outputs:

- `outputs/object_level_geo_rgb_vae_leaf_min4_calibrated_e8/val20_pred_rgb_surface_threshold_0p003`
- `outputs/object_level_geo_rgb_vae_leaf_min4_calibrated_e8/val20_pred_rgb_surface_threshold_0p005`
- `outputs/object_level_geo_rgb_vae_leaf_min4_calibrated_e8/val20_pred_rgb_surface_threshold_0p01`

### Final Conclusion

The calibrated query supervision makes direct threshold export possible without Top-K. However, `0.005` and especially `0.01` retain too many candidate points and are likely too loose. `0.003` is the most useful first visual candidate, but the high retention rate still suggests that the near-surface/negative separation is not yet sharp enough.

### Follow-Up Decision

Inspect `0.003` visually first. If it still fills volume, increase the volume/hard negative ratio or add distance-bucket weighted UDF loss before changing the decoder architecture.

## EXP-018: Weighted UDF Loss for Near-Surface Reconstruction

### Improvement Goal

Improve direct threshold reconstruction quality after EXP-017 showed that `0.005` was visually thick and partly missing, while looser thresholds became noisy. The goal is to make the decoder predict smaller and better-separated UDF values near real object surfaces.

### Scheme Design

Keep the calibrated surface query distribution, but replace plain Smooth L1 UDF regression with distance-bucket weighted Smooth L1:

- `target_udf < 0.003`: weight `8`
- `target_udf < 0.01`: weight `4`
- `target_udf < 0.03`: weight `2`
- farther queries: weight `1`

Theoretical reason: direct threshold export depends most on ranking and calibration around the near-surface band, not on average accuracy over all node-volume samples. The weighted loss makes near-surface errors more expensive while keeping volume negatives in training.

### Experiment Design

- Data: 80 train / 20 validation task-relevant Objaverse++ object PLY files.
- Octree: object preset, leaf-only, `min_node_points=4`.
- Query strategy: `calibrated_surface`.
- Model: RGB-fused VAE, hidden dim `256`, latent dim `128`.
- Training: 12 epochs, batch size `96`, `queries_per_node=192`, NPU `npu:8`.
- Export: threshold-only, no Top-K fallback, `864` hybrid candidates per leaf node.
- Thresholds tested: `0.005`, `0.006`, `0.007`.

### Implementation

Updated:

- `src/threedvae/train/octree_node_losses.py`
- `src/threedvae/train/octree_node_trainer.py`
- `scripts/train_octree_node_vae.py`
- `scripts/train_octree_node_vqvae.py`
- `tests/test_octree_node_vae.py`

Added `--udf-loss-mode bucket_weighted_smooth_l1` and bucket weight/threshold CLI options. Added a unit test to confirm the weighted UDF loss prioritizes near-surface errors.

### Results

Checkpoint:

`outputs/object_level_geo_rgb_vae_leaf_min4_calibrated_weighted_h256_l128_e12/best.pt`

Validation metrics:

| Metric | EXP-017 | EXP-018 |
|---|---:|---:|
| sample_count | 21,417 | 21,417 |
| query_count | 2,741,376 | 4,112,064 |
| UDF MAE | 0.006423 | 0.002895 |
| UDF RMSE | 0.014231 | 0.006646 |
| Occupancy accuracy | 0.956074 | 0.967987 |
| RGB MAE | 0.059363 | 0.050671 |
| RGB MSE | 0.010476 | 0.008475 |
| RGB PSNR | 19.80 dB | 20.72 dB |

Threshold-only exports:

| Threshold | Kept Points | Retention |
|---:|---:|---:|
| 0.005 | 10,348,867 | 55.93% |
| 0.006 | 11,269,327 | 60.90% |
| 0.007 | 11,916,050 | 64.40% |

Local outputs:

- `outputs/object_level_geo_rgb_vae_leaf_min4_calibrated_weighted_h256_l128_e12/val20_pred_rgb_surface_threshold_0p005`
- `outputs/object_level_geo_rgb_vae_leaf_min4_calibrated_weighted_h256_l128_e12/val20_pred_rgb_surface_threshold_0p006`
- `outputs/object_level_geo_rgb_vae_leaf_min4_calibrated_weighted_h256_l128_e12/val20_pred_rgb_surface_threshold_0p007`

### Final Conclusion

Weighted UDF loss substantially improves measured UDF and RGB reconstruction on the validation queries. This is theoretically consistent with the goal because the loss now emphasizes the threshold-critical near-surface region. However, the exported point count is much higher than EXP-017 at the same threshold, so visual inspection is still required: lower numerical UDF loss can make more candidates pass the threshold, and that is only useful if the accepted points are actually concentrated on surfaces rather than thickened voxel interiors.

### Follow-Up Decision

Use the new weighted checkpoint as the current best candidate. Compare `0.005 / 0.006 / 0.007` visually. If the surface is still thick, next improve candidate placement and training target sharpness instead of only changing threshold: adaptive candidates by node size, stronger near-zero surface anchor coverage, and optional Chamfer/F-score evaluation against validation points.

## EXP-019: RGB VQVAE Training Smoke and Validation

### Improvement Goal

Walk through the VQVAE training path on the current object-level RGB VAE setup before making further VAE architecture changes. The goal is to confirm that continuous node latents can be quantized into discrete code indices while preserving enough UDF/RGB reconstruction quality for later tokenizer work.

### Scheme Design

Use the best RGB VAE checkpoint from EXP-018 as initialization, then replace the continuous latent path with a vector quantizer:

- Encoder produces `mu`.
- `mu` is quantized by nearest codebook entry.
- Decoder receives the quantized latent.
- Reconstruction losses remain UDF, occupancy, and RGB.
- VQ loss is added with commitment cost `0.25`.

One important code change was added before training: initialize the codebook from pretrained encoder latents sampled from the training set. This avoids starting from a tiny random codebook that is far away from the pretrained VAE latent distribution.

### Experiment Design

- Data: 80 train / 20 validation task-relevant Objaverse++ object PLY files.
- Octree: object preset, leaf-only, `min_node_points=4`.
- Initialization: `outputs/object_level_geo_rgb_vae_leaf_min4_calibrated_weighted_h256_l128_e12/best.pt`.
- Codebook size: `1024`.
- Query strategy: `calibrated_surface`.
- Training: 8 epochs, batch size `96`, `queries_per_node=192`, NPU `npu:8`.
- Loss: weighted UDF loss from EXP-018, RGB weight `1.0`, VQ weight `1.0`.
- Export: threshold-only, no Top-K fallback, `threshold=0.006`, predicted RGB.

### Implementation

Updated:

- `scripts/train_octree_node_vqvae.py`
- `scripts/evaluate_octree_vae_checkpoint.py`
- `scripts/export_octree_vae_surface.py`

Added:

- `--init-codebook-from-data`
- `--codebook-init-max-samples`
- VQVAE-aware checkpoint loading for eval/export.
- Codebook usage metrics in checkpoint evaluation: `used_code_count`, `used_code_ratio`, `code_perplexity`.

### Results

Smoke run:

- `outputs/_smoke_vqvae_cb128_e1`
- 1 epoch completed.
- VQVAE checkpoint loading and codebook metrics verified.
- `codebook_size=128`, `used_code_count=13`, `code_perplexity=6.90`.

Formal checkpoint:

`outputs/object_level_geo_rgb_vqvae_leaf_min4_calibrated_weighted_cb1024_e8/best.pt`

Validation metrics:

| Metric | VAE EXP-018 | VQVAE EXP-019 |
|---|---:|---:|
| sample_count | 21,417 | 21,417 |
| query_count | 4,112,064 | 4,112,064 |
| UDF MAE | 0.002895 | 0.003983 |
| UDF RMSE | 0.006646 | 0.008736 |
| Occupancy accuracy | 0.967987 | 0.965797 |
| RGB MAE | 0.050671 | 0.080894 |
| RGB MSE | 0.008475 | 0.013443 |
| RGB PSNR | 20.72 dB | 18.72 dB |
| Codebook size | n/a | 1,024 |
| Used codes | n/a | 279 |
| Used code ratio | n/a | 27.25% |
| Code perplexity | n/a | 101.12 |

Threshold-only export:

| Threshold | Kept Points | Retention |
|---:|---:|---:|
| 0.006 | 13,178,067 | 71.22% |

Local output:

- `outputs/object_level_geo_rgb_vqvae_leaf_min4_calibrated_weighted_cb1024_e8/val20_pred_rgb_surface_threshold_0p006`

### Final Conclusion

The VQVAE training, evaluation, and RGB point-cloud export path is now functionally complete. Quantization causes expected reconstruction degradation compared with the continuous VAE, especially in RGB PSNR. Codebook usage is nonzero and not fully collapsed, but `279/1024` used codes and perplexity `101` indicate that codebook utilization is still modest.

### Follow-Up Decision

Use EXP-019 as the baseline VQVAE. Before optimizing visual quality, first improve discrete-token health: compare smaller codebooks such as `256` or `512`, consider longer training, and track code usage/perplexity alongside reconstruction metrics. If RGB quality remains weak, train geometry-only VQVAE first, then add RGB decoding.

## EXP-020: EMA VQVAE with Codebook Size 512

### Improvement Goal

Reduce VQVAE block artifacts and color patches by improving codebook health. The user observed that the baseline VQVAE point cloud was usable but had stronger block traces, more noise, and color patches that looked like one block sharing one color.

### Scheme Design

Test two controlled changes against EXP-019:

- Reduce `codebook_size` from `1024` to `512`.
- Replace gradient-updated VQ with EMA vector quantization.

The EMA codebook is updated as an online moving average of encoder latents assigned to each code. This should make code vectors behave more like stable cluster centers and can improve code usage stability. The codebook is still initialized from pretrained VAE encoder latents.

### Experiment Design

- Data: 80 train / 20 validation task-relevant Objaverse++ object PLY files.
- Octree: object preset, leaf-only, `min_node_points=4`.
- Initialization: EXP-018 RGB VAE checkpoint.
- Quantizer: EMA VQ, `codebook_size=512`, `ema_decay=0.99`, commitment cost `0.25`.
- Query strategy: `calibrated_surface`.
- Training: 8 epochs, batch size `96`, `queries_per_node=192`, NPU `npu:8`.
- Export: threshold-only, no Top-K fallback, `threshold=0.006`, predicted RGB.

### Implementation

Updated:

- `src/threedvae/models/octree_node_vqvae.py`
- `scripts/train_octree_node_vqvae.py`
- `tests/test_octree_node_vae.py`

Added:

- `EMAVectorQuantizer`
- `--quantizer-type ema`
- `--ema-decay`
- EMA buffer synchronization when initializing the codebook from data.

### Results

Smoke run:

- `outputs/_smoke_vqvae_ema_cb64_e1`
- 1 epoch completed on NPU.
- EMA update path verified.

Formal checkpoint:

`outputs/object_level_geo_rgb_vqvae_leaf_min4_calibrated_weighted_ema_cb512_e8/best.pt`

Validation metrics:

| Metric | Standard VQ cb1024 | EMA VQ cb512 |
|---|---:|---:|
| sample_count | 21,417 | 21,417 |
| query_count | 4,112,064 | 4,112,064 |
| UDF MAE | 0.003983 | 0.003967 |
| UDF RMSE | 0.008736 | 0.007997 |
| Occupancy accuracy | 0.965797 | 0.968290 |
| RGB MAE | 0.080894 | 0.075320 |
| RGB MSE | 0.013443 | 0.012294 |
| RGB PSNR | 18.72 dB | 19.10 dB |
| Codebook size | 1,024 | 512 |
| Used codes | 279 | 151 |
| Used code ratio | 27.25% | 29.49% |
| Code perplexity | 101.12 | 85.63 |

Threshold-only export:

| Threshold | Standard VQ cb1024 | EMA VQ cb512 |
|---:|---:|---:|
| 0.006 kept points | 13,178,067 | 13,305,063 |

Local output:

- `outputs/object_level_geo_rgb_vqvae_leaf_min4_calibrated_weighted_ema_cb512_e8/val20_pred_rgb_surface_threshold_0p006`

### Final Conclusion

EMA cb512 improves reconstruction metrics over the standard cb1024 VQVAE: better UDF RMSE, occupancy accuracy, RGB MAE/MSE/PSNR, and a slightly higher used-code ratio. It does not reduce exported point count at threshold `0.006`; the visual decision depends on whether color patches and block boundaries are softer in the exported PLY.

### Follow-Up Decision

Use EMA cb512 as the stronger VQVAE baseline if visual block artifacts improve. If color patches remain obvious, the next change should be geometry VQ plus continuous/residual RGB decoding, because a single node-level discrete latent is still likely to quantize color too coarsely.

## EXP-021: EMA VQVAE with Codebook Size 256

### Improvement Goal

Test whether the low effective codebook usage in EXP-020 means the codebook should be smaller. If the model only needs about 80-100 effective codes, a `256`-entry EMA codebook might improve utilization and reduce block artifacts without hurting reconstruction too much.

### Scheme Design

Keep the EXP-020 EMA quantizer and training setup, changing only:

- `codebook_size=512` -> `codebook_size=256`

All other settings remain the same: VAE latent initialization, EMA decay `0.99`, commitment cost `0.25`, VQ weight `1.0`, calibrated surface queries, and RGB reconstruction.

### Experiment Design

- Data: 80 train / 20 validation task-relevant Objaverse++ object PLY files.
- Octree: object preset, leaf-only, `min_node_points=4`.
- Initialization: EXP-018 RGB VAE checkpoint.
- Quantizer: EMA VQ, `codebook_size=256`, `ema_decay=0.99`, commitment cost `0.25`.
- Training: 8 epochs, batch size `96`, `queries_per_node=192`, NPU `npu:8`.
- Evaluation: validation metrics plus code usage/perplexity.

### Results

Checkpoint:

`outputs/object_level_geo_rgb_vqvae_leaf_min4_calibrated_weighted_ema_cb256_e8/best.pt`

Validation metrics:

| Metric | EMA VQ cb512 | EMA VQ cb256 |
|---|---:|---:|
| sample_count | 21,417 | 21,417 |
| query_count | 4,112,064 | 4,112,064 |
| UDF MAE | 0.003967 | 0.004492 |
| UDF RMSE | 0.007997 | 0.009908 |
| Occupancy accuracy | 0.968290 | 0.965182 |
| RGB MAE | 0.075320 | 0.080558 |
| RGB MSE | 0.012294 | 0.013307 |
| RGB PSNR | 19.10 dB | 18.76 dB |
| Codebook size | 512 | 256 |
| Used codes | 151 | 63 |
| Used code ratio | 29.49% | 24.61% |
| Code perplexity | 85.63 | 40.63 |

### Final Conclusion

Simply shrinking the EMA codebook to `256` does not improve utilization or reconstruction. Effective code usage drops from perplexity `85.63` to `40.63`, and UDF/RGB metrics also degrade. This suggests the current low utilization is not just caused by an oversized codebook; the encoder/decoder and node-level RGB/geometric coupling are still concentrating samples into a small set of codes.

### Follow-Up Decision

Do not replace the current EMA cb512 baseline with cb256. The next experiment should keep `codebook_size=512` and lower commitment/VQ weights to test whether weaker quantization pressure can recover code diversity without sacrificing reconstruction.

## EXP-022: EMA cb512 with Lower Commitment and VQ Weight

### Improvement Goal

Improve codebook utilization while keeping the stronger EMA cb512 capacity from EXP-020. EXP-021 showed that simply shrinking the codebook to 256 hurts both reconstruction and effective code usage, so this experiment keeps `codebook_size=512` and weakens the quantization constraint.

### Scheme Design

Compare against EXP-020 and change only:

- `commitment_cost=0.25` -> `0.1`
- `vq_weight=1.0` -> `0.5`

All other key settings remain unchanged: EMA quantizer, `codebook_size=512`, VAE latent initialization, calibrated surface query supervision, and RGB reconstruction.

### Experiment Design

- Data: 80 train / 20 validation task-relevant Objaverse++ object PLY files.
- Octree: object preset, leaf-only, `min_node_points=4`.
- Initialization: EXP-018 RGB VAE checkpoint.
- Quantizer: EMA VQ, `codebook_size=512`, `ema_decay=0.99`, commitment cost `0.1`.
- VQ loss weight: `0.5`.
- Training: 8 epochs, batch size `96`, `queries_per_node=192`, NPU `npu:8`.
- Export: threshold-only, no Top-K fallback, `threshold=0.006`, predicted RGB.

### Results

Checkpoint:

`outputs/object_level_geo_rgb_vqvae_leaf_min4_calibrated_weighted_ema_cb512_c01_vq05_e8/best.pt`

Validation metrics:

| Metric | EMA cb512 c0.25 vq1.0 | EMA cb512 c0.1 vq0.5 |
|---|---:|---:|
| sample_count | 21,417 | 21,417 |
| query_count | 4,112,064 | 4,112,064 |
| UDF MAE | 0.003967 | 0.003647 |
| UDF RMSE | 0.007997 | 0.007537 |
| Occupancy accuracy | 0.968290 | 0.967001 |
| RGB MAE | 0.075320 | 0.072097 |
| RGB MSE | 0.012294 | 0.011482 |
| RGB PSNR | 19.10 dB | 19.40 dB |
| Codebook size | 512 | 512 |
| Used codes | 151 | 372 |
| Used code ratio | 29.49% | 72.66% |
| Code perplexity | 85.63 | 202.45 |

Threshold-only export:

| Threshold | EMA cb512 c0.25 vq1.0 | EMA cb512 c0.1 vq0.5 |
|---:|---:|---:|
| 0.006 kept points | 13,305,063 | 12,872,519 |

Local output:

- `outputs/object_level_geo_rgb_vqvae_leaf_min4_calibrated_weighted_ema_cb512_c01_vq05_e8/val20_pred_rgb_surface_threshold_0p006`

### Final Conclusion

Lowering the commitment cost and VQ weight is clearly beneficial in this setting. It substantially improves codebook usage (`151 -> 372` used codes, perplexity `85.63 -> 202.45`) while also improving UDF RMSE and RGB PSNR. This suggests the previous EMA cb512 model was over-constrained by the quantization loss, causing excessive code concentration.

### Follow-Up Decision

Use `EMA cb512 c0.1 vq0.5` as the current best VQVAE baseline. The next decision should be visual: inspect whether color block artifacts are reduced in the exported PLY. If block artifacts remain despite healthier code usage, the next structural change should be geometry VQ plus continuous/residual RGB decoding.
