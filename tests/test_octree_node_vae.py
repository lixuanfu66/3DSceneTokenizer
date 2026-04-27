from __future__ import annotations

import unittest

from threedvae.models.octree_node_vae import OctreeNodeVAE
from threedvae.models.octree_node_vqvae import OctreeNodeVQVAE
from threedvae.train.octree_node_losses import octree_node_vae_loss
from threedvae.utils.torch_compat import HAS_TORCH, require_torch


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class OctreeNodeVAETest(unittest.TestCase):
    def test_octree_node_vae_forward_and_backward(self) -> None:
        torch, _, _, _, _ = require_torch()
        batch = _make_batch(torch)
        model = OctreeNodeVAE(
            num_points=8,
            hidden_dim=32,
            latent_dim=16,
            semantic_vocab_size=32,
            semantic_dim=4,
            num_attention_heads=4,
        )

        outputs = model(**batch)
        self.assertEqual(outputs.udf.shape, (2, 6, 1))
        self.assertEqual(outputs.occ_logits.shape, (2, 6, 1))
        self.assertEqual(outputs.latent.shape, (2, 16))

        losses = octree_node_vae_loss(
            pred_udf=outputs.udf,
            pred_occ_logits=outputs.occ_logits,
            target_udf=torch.rand(2, 6, 1),
            target_occ=torch.randint(0, 2, (2, 6, 1), dtype=torch.float32),
            mu=outputs.mu,
            logvar=outputs.logvar,
        )
        losses.total_loss.backward()
        self.assertTrue(torch.isfinite(losses.total_loss))

    def test_fused_vae_predicts_rgb_on_rgb_queries(self) -> None:
        torch, _, _, _, _ = require_torch()
        batch = _make_batch(torch)
        batch["rgb_query_xyz"] = torch.randn(2, 8, 3)
        model = OctreeNodeVAE(
            num_points=8,
            hidden_dim=32,
            latent_dim=16,
            semantic_vocab_size=32,
            semantic_dim=4,
            num_attention_heads=4,
            use_rgb_fusion=True,
            predict_rgb=True,
        )

        outputs = model(**batch)
        self.assertIsNotNone(outputs.rgb)
        self.assertEqual(outputs.rgb.shape, (2, 8, 3))

    def test_octree_node_vqvae_returns_code_indices(self) -> None:
        torch, _, _, _, _ = require_torch()
        batch = _make_batch(torch)
        model = OctreeNodeVQVAE(
            num_points=8,
            hidden_dim=32,
            latent_dim=16,
            semantic_vocab_size=32,
            semantic_dim=4,
            num_attention_heads=4,
            codebook_size=16,
        )

        outputs = model(**batch)
        self.assertEqual(outputs.encoding_indices.shape, (2,))
        self.assertTrue(torch.isfinite(outputs.vq_loss))


def _make_batch(torch):
    return {
        "xyz": torch.randn(2, 8, 3),
        "rgb": torch.rand(2, 8, 3),
        "query_xyz": torch.randn(2, 6, 3),
        "node_center": torch.zeros(2, 3),
        "node_size": torch.ones(2, 3),
        "level": torch.tensor([0, 1], dtype=torch.int64),
        "split_flag": torch.tensor([7, 3], dtype=torch.int64),
        "child_index": torch.tensor([-1, 2], dtype=torch.int64),
        "semantic_id": torch.tensor([1, 14], dtype=torch.int64),
    }
