from __future__ import annotations

from dataclasses import asdict, dataclass

from threedvae.utils.torch_compat import HAS_TORCH, F, nn, require_torch, torch


@dataclass(slots=True)
class OctreeNodeVAEConfig:
    num_points: int = 128
    point_feature_dim: int = 9
    rgb_feature_dim: int = 6
    hidden_dim: int = 256
    latent_dim: int = 128
    semantic_vocab_size: int = 256
    semantic_dim: int = 16
    num_attention_heads: int = 4
    use_rgb_fusion: bool = False
    predict_rgb: bool = False


@dataclass(slots=True)
class OctreeNodeVAEOutput:
    latent: object
    residual_latent: object
    mu: object
    logvar: object
    udf: object
    occ_logits: object
    decoder_features: object
    rgb: object | None = None


if HAS_TORCH:

    class MLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, layers: int = 2) -> None:
            super().__init__()
            modules: list[nn.Module] = []
            current_dim = int(input_dim)
            for _ in range(max(1, int(layers) - 1)):
                modules.extend([nn.Linear(current_dim, hidden_dim), nn.SiLU()])
                current_dim = int(hidden_dim)
            modules.append(nn.Linear(current_dim, output_dim))
            self.net = nn.Sequential(*modules)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.net(values)


    class NodeConditionEncoder(nn.Module):
        def __init__(self, semantic_vocab_size: int, semantic_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.semantic_embedding = nn.Embedding(int(semantic_vocab_size), int(semantic_dim))
            input_dim = 3 + 3 + 1 + 3 + 3 + int(semantic_dim)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )

        def forward(
            self,
            *,
            node_center: torch.Tensor,
            node_size: torch.Tensor,
            level: torch.Tensor,
            split_flag: torch.Tensor,
            child_index: torch.Tensor,
            semantic_id: torch.Tensor,
        ) -> torch.Tensor:
            split_bits = _int_bits(split_flag, bit_count=3, device=node_center.device)
            child_bits = _int_bits(torch.clamp(child_index, min=0), bit_count=3, device=node_center.device)
            level_value = level.to(dtype=node_center.dtype).unsqueeze(-1) / 8.0
            safe_size = torch.clamp(node_size, min=1e-5)
            semantic = self.semantic_embedding(
                torch.clamp(semantic_id, min=0, max=self.semantic_embedding.num_embeddings - 1)
            )
            features = torch.cat(
                [
                    node_center,
                    torch.log(safe_size),
                    level_value,
                    split_bits.to(dtype=node_center.dtype),
                    child_bits.to(dtype=node_center.dtype),
                    semantic,
                ],
                dim=-1,
            )
            return self.net(features)


    class OctreeNodeVAE(nn.Module):
        def __init__(
            self,
            *,
            num_points: int = 128,
            point_feature_dim: int = 9,
            rgb_feature_dim: int = 6,
            hidden_dim: int = 256,
            latent_dim: int = 128,
            semantic_vocab_size: int = 256,
            semantic_dim: int = 16,
            num_attention_heads: int = 4,
            use_rgb_fusion: bool = False,
            predict_rgb: bool = False,
        ) -> None:
            super().__init__()
            self.config = OctreeNodeVAEConfig(
                num_points=int(num_points),
                point_feature_dim=int(point_feature_dim),
                rgb_feature_dim=int(rgb_feature_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=int(latent_dim),
                semantic_vocab_size=int(semantic_vocab_size),
                semantic_dim=int(semantic_dim),
                num_attention_heads=int(num_attention_heads),
                use_rgb_fusion=bool(use_rgb_fusion),
                predict_rgb=bool(predict_rgb),
            )
            self.num_points = self.config.num_points
            self.condition_encoder = NodeConditionEncoder(
                semantic_vocab_size=self.config.semantic_vocab_size,
                semantic_dim=self.config.semantic_dim,
                hidden_dim=self.config.hidden_dim,
            )
            self.point_stem = MLP(
                self.config.point_feature_dim,
                self.config.hidden_dim,
                self.config.hidden_dim,
                layers=3,
            )
            self.geo_attention = nn.MultiheadAttention(
                embed_dim=self.config.hidden_dim,
                num_heads=self.config.num_attention_heads,
                batch_first=True,
            )
            if self.config.use_rgb_fusion:
                self.rgb_stem = MLP(
                    self.config.rgb_feature_dim,
                    self.config.hidden_dim,
                    self.config.hidden_dim,
                    layers=3,
                )
                self.rgb_attention = nn.MultiheadAttention(
                    embed_dim=self.config.hidden_dim,
                    num_heads=self.config.num_attention_heads,
                    batch_first=True,
                )
                self.fusion = nn.Sequential(
                    nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                    nn.SiLU(),
                )
            else:
                self.rgb_stem = None
                self.rgb_attention = None
                self.fusion = nn.Sequential(
                    nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                    nn.SiLU(),
                )
            self.mu_head = nn.Linear(self.config.hidden_dim, self.config.latent_dim)
            self.logvar_head = nn.Linear(self.config.hidden_dim, self.config.latent_dim)

            decoder_input_dim = 9 + self.config.latent_dim + self.config.hidden_dim
            self.decoder = nn.Sequential(
                nn.Linear(decoder_input_dim, self.config.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.SiLU(),
            )
            self.udf_head = nn.Linear(self.config.hidden_dim, 1)
            self.occ_head = nn.Linear(self.config.hidden_dim, 1)
            self.rgb_head = nn.Linear(self.config.hidden_dim, 3) if self.config.predict_rgb else None

        def encode(
            self,
            *,
            xyz: torch.Tensor,
            rgb: torch.Tensor | None,
            node_center: torch.Tensor,
            node_size: torch.Tensor,
            level: torch.Tensor,
            split_flag: torch.Tensor,
            child_index: torch.Tensor,
            semantic_id: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            condition = self.condition_encoder(
                node_center=node_center,
                node_size=node_size,
                level=level,
                split_flag=split_flag,
                child_index=child_index,
                semantic_id=semantic_id,
            )
            query = condition[:, None, :]
            point_features = self.point_stem(_geometry_point_features(xyz, node_center, node_size))
            geo_pooled, _ = self.geo_attention(query, point_features, point_features, need_weights=False)
            geo_pooled = geo_pooled[:, 0, :]

            if self.config.use_rgb_fusion:
                if rgb is None:
                    raise ValueError("rgb must be provided when use_rgb_fusion=True.")
                if self.rgb_stem is None or self.rgb_attention is None:
                    raise RuntimeError("RGB fusion modules are not initialized.")
                rgb_features = self.rgb_stem(_rgb_point_features(xyz, rgb, node_center, node_size))
                rgb_pooled, _ = self.rgb_attention(query, rgb_features, rgb_features, need_weights=False)
                fused = self.fusion(torch.cat([geo_pooled, rgb_pooled[:, 0, :], condition], dim=-1))
            else:
                fused = self.fusion(torch.cat([geo_pooled, condition], dim=-1))

            mu = self.mu_head(fused)
            logvar = torch.clamp(self.logvar_head(fused), min=-12.0, max=8.0)
            return mu, logvar, condition

        def decode(
            self,
            *,
            latent: torch.Tensor,
            query_xyz: torch.Tensor,
            node_center: torch.Tensor,
            node_size: torch.Tensor,
            condition: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
            query_features = _query_features(query_xyz, node_center, node_size)
            expanded_latent = latent[:, None, :].expand(-1, query_xyz.shape[1], -1)
            expanded_condition = condition[:, None, :].expand(-1, query_xyz.shape[1], -1)
            features = torch.cat([query_features, expanded_latent, expanded_condition], dim=-1)
            decoder_features = self.decoder(features)
            udf = F.softplus(self.udf_head(decoder_features))
            occ_logits = self.occ_head(decoder_features)
            rgb = torch.sigmoid(self.rgb_head(decoder_features)) if self.rgb_head is not None else None
            return udf, occ_logits, decoder_features, rgb

        def forward(
            self,
            *,
            xyz: torch.Tensor,
            query_xyz: torch.Tensor,
            node_center: torch.Tensor,
            node_size: torch.Tensor,
            level: torch.Tensor,
            split_flag: torch.Tensor,
            child_index: torch.Tensor,
            semantic_id: torch.Tensor,
            rgb: torch.Tensor | None = None,
            parent_latent: torch.Tensor | None = None,
            rgb_query_xyz: torch.Tensor | None = None,
        ) -> OctreeNodeVAEOutput:
            mu, logvar, condition = self.encode(
                xyz=xyz,
                rgb=rgb,
                node_center=node_center,
                node_size=node_size,
                level=level,
                split_flag=split_flag,
                child_index=child_index,
                semantic_id=semantic_id,
            )
            residual_latent = _reparameterize(mu, logvar) if self.training else mu
            latent = residual_latent if parent_latent is None else parent_latent + residual_latent
            udf, occ_logits, decoder_features, rgb_pred = self.decode(
                latent=latent,
                query_xyz=query_xyz,
                node_center=node_center,
                node_size=node_size,
                condition=condition,
            )
            if rgb_query_xyz is not None and self.rgb_head is not None:
                _, _, _, rgb_pred = self.decode(
                    latent=latent,
                    query_xyz=rgb_query_xyz,
                    node_center=node_center,
                    node_size=node_size,
                    condition=condition,
                )
            return OctreeNodeVAEOutput(
                latent=latent,
                residual_latent=residual_latent,
                mu=mu,
                logvar=logvar,
                udf=udf,
                occ_logits=occ_logits,
                decoder_features=decoder_features,
                rgb=rgb_pred,
            )

        def export_config(self) -> dict[str, int | bool]:
            return asdict(self.config)


    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def _geometry_point_features(
        xyz: torch.Tensor,
        node_center: torch.Tensor,
        node_size: torch.Tensor,
    ) -> torch.Tensor:
        relative = xyz - node_center[:, None, :]
        normalized = relative / torch.clamp(node_size[:, None, :], min=1e-5)
        return torch.cat([xyz, relative, normalized], dim=-1)


    def _rgb_point_features(
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        node_center: torch.Tensor,
        node_size: torch.Tensor,
    ) -> torch.Tensor:
        relative = xyz - node_center[:, None, :]
        normalized = relative / torch.clamp(node_size[:, None, :], min=1e-5)
        return torch.cat([normalized, rgb], dim=-1)


    def _query_features(
        query_xyz: torch.Tensor,
        node_center: torch.Tensor,
        node_size: torch.Tensor,
    ) -> torch.Tensor:
        relative = query_xyz - node_center[:, None, :]
        normalized = relative / torch.clamp(node_size[:, None, :], min=1e-5)
        return torch.cat([query_xyz, relative, normalized], dim=-1)


    def _int_bits(values: torch.Tensor, *, bit_count: int, device) -> torch.Tensor:
        values_cpu = values.detach().to(device="cpu", dtype=torch.int64)
        bits = [((values_cpu // (1 << bit)) % 2).to(dtype=torch.float32) for bit in range(bit_count)]
        return torch.stack(bits, dim=-1).to(device=device)

else:

    class OctreeNodeVAE:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            require_torch()
