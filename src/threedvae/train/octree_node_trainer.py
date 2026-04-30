from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from threedvae.data.dataset import TreeNodePointCloudDataset
from threedvae.train.octree_node_losses import octree_node_vae_loss
from threedvae.utils.torch_compat import HAS_TORCH, DataLoader, require_torch, torch


@dataclass(slots=True)
class OctreeNodeTrainerConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cpu"
    udf_weight: float = 1.0
    occ_weight: float = 0.1
    kl_weight: float = 1e-4
    kl_warmup_ratio: float = 0.1
    rgb_weight: float = 0.0
    vq_weight: float = 1.0
    occ_target_mode: str = "soft_udf"
    occ_soft_distance: float = 0.03
    udf_loss_mode: str = "smooth_l1"
    udf_near_weight: float = 1.0
    udf_band_weight: float = 1.0
    udf_mid_weight: float = 1.0
    udf_far_weight: float = 1.0
    udf_near_threshold: float = 0.003
    udf_band_threshold: float = 0.01
    udf_mid_threshold: float = 0.03
    grad_clip_norm: float | None = 1.0
    num_workers: int = 0
    checkpoint_every: int = 1


class OctreeNodeTrainer:
    def __init__(self, model, config: OctreeNodeTrainerConfig) -> None:
        if not HAS_TORCH:
            require_torch()
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.global_step = 0
        self.total_steps = 1

    def fit(
        self,
        dataset: TreeNodePointCloudDataset,
        output_dir: str,
        *,
        val_dataset: TreeNodePointCloudDataset | None = None,
    ) -> dict[str, list[float]]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=dataset.torch_collate,
        )
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=val_dataset.torch_collate,
            )
        self.total_steps = max(1, self.config.epochs * len(train_loader))
        history = {
            "total_loss": [],
            "udf_loss": [],
            "occ_loss": [],
            "kl_loss": [],
            "rgb_loss": [],
            "vq_loss": [],
        }
        if val_loader is not None:
            history.update({f"val_{key}": [] for key in list(history.keys())})
        best_metric = float("inf")
        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(train_loader)
            for key, value in metrics.items():
                history[key].append(float(value))
            tracked_metric = float(metrics["total_loss"])
            if val_loader is not None:
                val_metrics = self.evaluate_epoch(val_loader)
                for key, value in val_metrics.items():
                    history[f"val_{key}"].append(float(value))
                tracked_metric = float(val_metrics["total_loss"])

            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(output_path / f"checkpoint_epoch_{epoch + 1}.pt", epoch + 1)
            self.save_checkpoint(output_path / "latest.pt", epoch + 1)
            if tracked_metric < best_metric:
                best_metric = tracked_metric
                self.save_checkpoint(output_path / "best.pt", epoch + 1)

        with (output_path / "history.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with (output_path / "trainer_config.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(asdict(self.config), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return history

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        total = _empty_metrics()
        step_count = 0
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            losses = self._compute_losses(batch, train=True)
            losses.total_loss.backward()
            if self.config.grad_clip_norm is not None and self.config.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            self.global_step += 1
            _accumulate(total, losses)
            step_count += 1
        if step_count == 0:
            raise ValueError("Training dataset is empty.")
        return {key: value / step_count for key, value in total.items()}

    @torch.no_grad()
    def evaluate_epoch(self, loader) -> dict[str, float]:
        self.model.eval()
        total = _empty_metrics()
        step_count = 0
        for batch in loader:
            losses = self._compute_losses(batch, train=False)
            _accumulate(total, losses)
            step_count += 1
        if step_count == 0:
            raise ValueError("Validation dataset is empty.")
        return {key: value / step_count for key, value in total.items()}

    def _compute_losses(self, batch, *, train: bool):
        del train
        batch = _move_batch(batch, self.config.device)
        outputs = self.model(
            xyz=batch["xyz"],
            rgb=batch["rgb"],
            query_xyz=batch["query_xyz"],
            rgb_query_xyz=batch.get("query_rgb_xyz"),
            node_center=batch["node_center_local"],
            node_size=batch["node_size_local"],
            level=batch["level"],
            split_flag=batch["split_flag"],
            child_index=batch["child_index"],
            semantic_id=batch["semantic_id"],
        )
        kl_weight = self._current_kl_weight()
        losses = octree_node_vae_loss(
            pred_udf=outputs.udf,
            pred_occ_logits=outputs.occ_logits,
            target_udf=batch["query_udf"],
            target_occ=batch["query_occ"],
            mu=outputs.mu,
            logvar=outputs.logvar,
            pred_rgb=outputs.rgb,
            target_rgb=batch.get("query_rgb"),
            rgb_mask=batch.get("query_rgb_mask"),
            vq_loss=getattr(outputs, "vq_loss", None),
            udf_weight=self.config.udf_weight,
            occ_weight=self.config.occ_weight,
            kl_weight=kl_weight,
            rgb_weight=self.config.rgb_weight,
            vq_weight=self.config.vq_weight,
            occ_target_mode=self.config.occ_target_mode,
            occ_soft_distance=self.config.occ_soft_distance,
            udf_loss_mode=self.config.udf_loss_mode,
            udf_near_weight=self.config.udf_near_weight,
            udf_band_weight=self.config.udf_band_weight,
            udf_mid_weight=self.config.udf_mid_weight,
            udf_far_weight=self.config.udf_far_weight,
            udf_near_threshold=self.config.udf_near_threshold,
            udf_band_threshold=self.config.udf_band_threshold,
            udf_mid_threshold=self.config.udf_mid_threshold,
        )
        if not torch.isfinite(losses.total_loss):
            raise RuntimeError("Encountered non-finite octree node training loss.")
        return losses

    def _current_kl_weight(self) -> float:
        warmup_steps = int(self.total_steps * max(0.0, self.config.kl_warmup_ratio))
        if warmup_steps <= 0:
            return float(self.config.kl_weight)
        scale = min(1.0, float(self.global_step + 1) / float(warmup_steps))
        return float(self.config.kl_weight * scale)

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": self.model.export_config() if hasattr(self.model, "export_config") else None,
                "trainer_config": asdict(self.config),
                "global_step": self.global_step,
            },
            path,
        )


def _move_batch(batch, device: str):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _empty_metrics() -> dict[str, float]:
    return {
        "total_loss": 0.0,
        "udf_loss": 0.0,
        "occ_loss": 0.0,
        "kl_loss": 0.0,
        "rgb_loss": 0.0,
        "vq_loss": 0.0,
    }


def _accumulate(total: dict[str, float], losses) -> None:
    total["total_loss"] += float(losses.total_loss.detach().cpu())
    total["udf_loss"] += float(losses.udf_loss.detach().cpu())
    total["occ_loss"] += float(losses.occ_loss.detach().cpu())
    total["kl_loss"] += float(losses.kl_loss.detach().cpu())
    total["rgb_loss"] += float(losses.rgb_loss.detach().cpu())
    total["vq_loss"] += float(losses.vq_loss.detach().cpu())
