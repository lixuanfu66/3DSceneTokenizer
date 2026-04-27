# Evaluation Notes

The legacy fixed-point `PointNetVQTokenizer` evaluator has been removed with the old tokenizer stack.

Current checkpoints should be inspected through:

- `history.json` from `scripts/train_octree_node_vae.py`
- `history.json` from `scripts/train_octree_node_vqvae.py`
- codebook metrics from the VQVAE trainer outputs once a dedicated evaluator is added

The next evaluator should target the current model family:

- geometry: `udf_loss`, `observed_occ_bce`
- RGB: masked near-surface RGB loss
- VQ: used code count, entropy, perplexity
- structure: tokens per instance, tokens by depth, tokens by semantic class
