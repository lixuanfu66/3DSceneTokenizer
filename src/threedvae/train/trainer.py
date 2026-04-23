from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from threedvae.data.dataset import InstancePointCloudDataset, TreeNodePointCloudDataset
from threedvae.models.instance_tokenizer import PointNetVQTokenizer
from threedvae.train.losses import instance_tokenizer_loss
from threedvae.utils.torch_compat import HAS_TORCH, DataLoader, require_torch, torch


@dataclass(slots=True)
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    xyz_weight: float = 1.0
    rgb_weight: float = 0.25
    vq_weight: float = 1.0
    udf_weight: float = 0.5
    num_workers: int = 0
    log_every: int = 10
    checkpoint_every: int = 1


class InstanceTokenizerTrainer:
    def __init__(self, model: PointNetVQTokenizer, config: TrainerConfig) -> None:
        if not HAS_TORCH:
            require_torch()
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def fit(
        self,
        dataset: InstancePointCloudDataset | TreeNodePointCloudDataset,
        output_dir: str,
        *,
        val_dataset: InstancePointCloudDataset | TreeNodePointCloudDataset | None = None,
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

        history = {"total_loss": [], "xyz_loss": [], "rgb_loss": [], "vq_loss": [], "udf_loss": []}
        if val_loader is not None:
            history.update(
                {
                    "val_total_loss": [],
                    "val_xyz_loss": [],
                    "val_rgb_loss": [],
                    "val_vq_loss": [],
                    "val_udf_loss": [],
                }
            )
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
        total = {"total_loss": 0.0, "xyz_loss": 0.0, "rgb_loss": 0.0, "vq_loss": 0.0, "udf_loss": 0.0}
        step_count = 0

        for batch in loader:
            points = batch["points"].to(self.config.device)
            target_xyz = batch["xyz"].to(self.config.device)
            target_rgb = batch["rgb"].to(self.config.device)
            query_xyz = batch["query_xyz"].to(self.config.device) if "query_xyz" in batch else None
            target_udf = batch["query_udf"].to(self.config.device) if "query_udf" in batch else None

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(points, query_xyz=query_xyz)
            losses = instance_tokenizer_loss(
                outputs.recon_points,
                target_xyz,
                target_rgb,
                outputs.vq_loss,
                udf_logits=outputs.udf_logits,
                target_udf=target_udf,
                xyz_weight=self.config.xyz_weight,
                rgb_weight=self.config.rgb_weight,
                vq_weight=self.config.vq_weight,
                udf_weight=self.config.udf_weight,
            )
            losses.total_loss.backward()
            self.optimizer.step()

            total["total_loss"] += float(losses.total_loss.detach().cpu())
            total["xyz_loss"] += float(losses.xyz_loss.detach().cpu())
            total["rgb_loss"] += float(losses.rgb_loss.detach().cpu())
            total["vq_loss"] += float(losses.vq_loss.detach().cpu())
            total["udf_loss"] += float(losses.udf_loss.detach().cpu())
            step_count += 1

        if step_count == 0:
            raise ValueError("Training dataset is empty.")
        return {key: value / step_count for key, value in total.items()}

    @torch.no_grad()
    def evaluate_epoch(self, loader) -> dict[str, float]:
        self.model.eval()
        total = {"total_loss": 0.0, "xyz_loss": 0.0, "rgb_loss": 0.0, "vq_loss": 0.0, "udf_loss": 0.0}
        step_count = 0

        for batch in loader:
            points = batch["points"].to(self.config.device)
            target_xyz = batch["xyz"].to(self.config.device)
            target_rgb = batch["rgb"].to(self.config.device)
            query_xyz = batch["query_xyz"].to(self.config.device) if "query_xyz" in batch else None
            target_udf = batch["query_udf"].to(self.config.device) if "query_udf" in batch else None
            outputs = self.model(points, query_xyz=query_xyz)
            losses = instance_tokenizer_loss(
                outputs.recon_points,
                target_xyz,
                target_rgb,
                outputs.vq_loss,
                udf_logits=outputs.udf_logits,
                target_udf=target_udf,
                xyz_weight=self.config.xyz_weight,
                rgb_weight=self.config.rgb_weight,
                vq_weight=self.config.vq_weight,
                udf_weight=self.config.udf_weight,
            )
            total["total_loss"] += float(losses.total_loss.detach().cpu())
            total["xyz_loss"] += float(losses.xyz_loss.detach().cpu())
            total["rgb_loss"] += float(losses.rgb_loss.detach().cpu())
            total["vq_loss"] += float(losses.vq_loss.detach().cpu())
            total["udf_loss"] += float(losses.udf_loss.detach().cpu())
            step_count += 1

        if step_count == 0:
            raise ValueError("Validation dataset is empty.")
        return {key: value / step_count for key, value in total.items()}

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": self.model.export_config() if hasattr(self.model, "export_config") else None,
                "config": asdict(self.config),
            },
            path,
        )
