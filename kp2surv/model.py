"""Complete KP2Surv network."""

from __future__ import annotations

from .common import *
from .data import _normalize01, segment_normalize01
from .components import *
from .metrics import *

class KP2Surv(nn.Module):
    def __init__(
        self,
        wsi_in_dim,
        gene_in_dim,
        hidden_dim=256,
        num_pathways=186,
        dropout=0.3,
        cross_attn_temperature=1.0,
        cross_num_heads=4,
        lambda_conf=0.15,
        lambda_qc=0.10,
        lambda_alpha_q=0.6,
        lambda_alpha_k=0.6,
        lambda_cross_q=0.8,
        ctr_top_frac=0.2,
        ctr_margin=0.1,
        q_topk_frac=0.1,
        stochastic_input_dropout=0.2,
        prior_table: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_pathways = int(num_pathways)
        self.q_topk_frac = float(q_topk_frac)
        self.ctr_top_frac = float(ctr_top_frac)
        self.ctr_margin = float(ctr_margin)
        self.lambda_qc = float(lambda_qc)
        self.stochastic_input_dropout = float(stochastic_input_dropout)
        self.prior_table = prior_table if prior_table is not None else load_weak_prior_table(None)

        self.gene_proj = nn.Sequential(nn.Linear(gene_in_dim, hidden_dim), nn.GELU())
        self.wsi_proj = nn.Sequential(nn.Linear(wsi_in_dim, hidden_dim), nn.GELU())

        self.wsi_backbone = HGNodeBackbone(hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.gene_backbone = HGNodeBackbone(hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

        self.wsi_edge_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Prior features: WSI hyperedges(6 continuous heuristics), Gene pathways(5)
        self.wsi_heads = ImportanceQualityHeads(hidden_dim, prior_dim=6, dropout=dropout)
        self.gene_heads = ImportanceQualityHeads(hidden_dim, prior_dim=5, dropout=dropout)

        self.cross_attn = ImportanceGuidedCrossAttention(
            hidden_dim,
            num_heads=cross_num_heads,
            dropout=dropout,
            lambda_alpha_q=lambda_alpha_q,
            lambda_alpha_k=lambda_alpha_k,
            lambda_q=lambda_cross_q,
            attn_temperature=cross_attn_temperature,
        )
        self.fusion_layer = ReliabilityDisagreementFusion(
            hidden_dim, lambda_conf=lambda_conf, lambda_qc=lambda_qc
        )

        self.wsi_branch_head = BranchSurvivalHead(hidden_dim, hidden_dim, dropout=dropout)
        self.gene_branch_head = BranchSurvivalHead(hidden_dim, hidden_dim, dropout=dropout)
        self.cross_branch_head = BranchSurvivalHead(hidden_dim, hidden_dim, dropout=dropout)
        self.fused_head = FusedSurvivalHead(hidden_dim, hidden_dim, dropout=dropout)

        self.l3 = L3Regularization(weight=0.1)

    @staticmethod
    def _num_graphs_from_batch(batch_index: Optional[torch.Tensor], fallback: int) -> int:
        if batch_index is None or batch_index.numel() == 0:
            return int(fallback)
        return int(batch_index.max().item()) + 1

    @staticmethod
    def _build_batch_index(data: GeomData, device: torch.device) -> torch.Tensor:
        if hasattr(data, "batch") and data.batch is not None:
            return data.batch.to(device=device, dtype=torch.long)
        return torch.zeros(data.num_nodes, device=device, dtype=torch.long)

    def _build_wsi_prior_features(self, wsi_batch: GeomBatch, x_proj: torch.Tensor, batch_wsi: torch.Tensor) -> torch.Tensor:
        n = x_proj.size(0)
        if n <= 0:
            return x_proj.new_zeros((0, 6))
        he = wsi_batch.hyperedge_index

        edge_score = x_proj.new_zeros((n,))
        cluster_rarity = x_proj.new_zeros((n,))
        spatial_support = x_proj.new_zeros((n,))
        feature_support = x_proj.new_zeros((n,))

        if he.numel() > 0:
            node_ids = he[0].long()
            he_ids = he[1].long()
            num_hyperedges = int(he_ids.max().item()) + 1

            one_node = torch.ones_like(node_ids, dtype=x_proj.dtype)
            one_he = torch.ones_like(he_ids, dtype=x_proj.dtype)

            edge_score = scatter_add(one_node, node_ids, dim=0, dim_size=n)
            he_size = scatter_add(one_he, he_ids, dim=0, dim_size=num_hyperedges).clamp_min(1.0)
            cluster_rarity = scatter_mean((1.0 / he_size[he_ids]), node_ids, dim=0, dim_size=n)

            use_typed_hyperedges = False
            if hasattr(wsi_batch, "hyperedge_type") and wsi_batch.hyperedge_type is not None:
                try:
                    he_type = wsi_batch.hyperedge_type
                    if not torch.is_tensor(he_type):
                        he_type = torch.as_tensor(he_type)
                    he_type = he_type.to(device=x_proj.device).view(-1)
                    if he_type.numel() == num_hyperedges:
                        he_type_inc = he_type[he_ids]
                        spatial_support = scatter_add((he_type_inc == 0).to(x_proj.dtype), node_ids, dim=0, dim_size=n)
                        feature_support = scatter_add((he_type_inc == 1).to(x_proj.dtype), node_ids, dim=0, dim_size=n)
                        use_typed_hyperedges = True
                except Exception:
                    use_typed_hyperedges = False
            if not use_typed_hyperedges:
                # Fallback when hyperedge types are unavailable.
                spatial_support = edge_score
                feature_support = edge_score

        x_mean = scatter_mean(x_proj, batch_wsi, dim=0)
        heterogeneity = (x_proj - x_mean[batch_wsi]).pow(2).mean(dim=-1).sqrt()

        feat_std = x_proj.std(dim=-1, unbiased=False)
        blur = (1.0 - segment_normalize01(feat_std, batch_wsi)).clamp(0.0, 1.0)
        stain_qc = segment_normalize01(x_proj.abs().mean(dim=-1), batch_wsi)

        p = torch.softmax(torch.clamp(x_proj, min=-6.0, max=6.0), dim=-1)
        local_entropy_raw = -(p * torch.log(torch.clamp(p, min=1e-12))).sum(dim=-1) / math.log(max(2, x_proj.size(-1)))
        local_entropy = segment_normalize01(local_entropy_raw, batch_wsi)

        edge_score = segment_normalize01(edge_score, batch_wsi)
        heterogeneity = segment_normalize01(heterogeneity, batch_wsi)
        cluster_rarity = segment_normalize01(cluster_rarity, batch_wsi)
        spatial_support = segment_normalize01(spatial_support, batch_wsi)
        feature_support = segment_normalize01(feature_support, batch_wsi)

        spatial_boundary = torch.clamp(feature_support - spatial_support, min=0.0, max=1.0)
        artifact_score = torch.clamp(0.50 * blur + 0.30 * (1.0 - stain_qc) + 0.20 * (1.0 - local_entropy), min=0.0, max=1.0)

        # External patch-level tumor probability if available; otherwise neutral prior.
        tumor_prob = x_proj.new_full((n,), 0.5)
        if hasattr(wsi_batch, "tumor_prob") and wsi_batch.tumor_prob is not None:
            try:
                t = wsi_batch.tumor_prob
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t)
                t = t.to(device=x_proj.device).float().view(-1)
                if t.numel() == n:
                    tumor_prob = t.clamp(0.0, 1.0)
            except Exception:
                pass

        # WSI prior features (continuous):
        # [edge_score, heterogeneity, cluster_rarity, spatial_boundary, artifact_score, tumor_prob]
        return torch.stack(
            [edge_score, heterogeneity, cluster_rarity, spatial_boundary, artifact_score, tumor_prob],
            dim=-1,
        )

    @staticmethod
    def _segment_normalize_feature_matrix(feat: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        if feat.numel() == 0:
            return feat
        out = feat.clone()
        for bid in torch.unique(batch_index, sorted=True):
            m = batch_index == bid
            vals = feat[m]
            if vals.numel() == 0:
                continue
            v_min = vals.min(dim=0, keepdim=True).values
            v_max = vals.max(dim=0, keepdim=True).values
            span = v_max - v_min
            norm = (vals - v_min) / (span + 1e-6)
            stable = span <= 1e-6
            if bool(stable.any()):
                norm = torch.where(stable.expand_as(norm), vals.clamp(0.0, 1.0), norm)
            out[m] = norm
        return out

    def _build_wsi_hyperedge_tokens(
        self,
        wsi_batch: GeomBatch,
        h_wsi: torch.Tensor,
        prior_wsi: torch.Tensor,
        batch_wsi: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Convert patch/node embeddings to pathology hyperedge tokens.
        These tokens are the WSI-side units for aggregation, cross-attention and
        visualization in 3.10-2.1.
        """
        he = wsi_batch.hyperedge_index
        if he is None or he.numel() == 0:
            batch_edge = torch.arange(batch_size, device=h_wsi.device, dtype=torch.long)
            h_edge = scatter_mean(h_wsi, batch_wsi, dim=0, dim_size=batch_size)
            prior_edge = scatter_mean(prior_wsi, batch_wsi, dim=0, dim_size=batch_size)
            edge_pos = None
            node_pos = getattr(wsi_batch, "pos", None)
            if node_pos is None:
                node_pos = getattr(wsi_batch, "centroid", None)
            if node_pos is not None:
                edge_pos = scatter_mean(node_pos.float(), batch_wsi, dim=0, dim_size=batch_size)
            return {
                "h": self.wsi_edge_proj(h_edge),
                "prior": prior_edge,
                "batch": batch_edge,
                "pos": edge_pos,
                "size": torch.bincount(batch_wsi, minlength=batch_size).float().unsqueeze(-1),
                "orig_hyperedge_idx": None,
            }

        node_ids = he[0].long()
        he_ids = he[1].long()
        num_edges = infer_num_hyperedges(he)

        h_edge_mean = scatter_mean(h_wsi[node_ids], he_ids, dim=0, dim_size=num_edges)
        prior_edge = scatter_mean(prior_wsi[node_ids], he_ids, dim=0, dim_size=num_edges)

        batch_edge = scatter_mean(batch_wsi[node_ids].float(), he_ids, dim=0, dim_size=num_edges)
        batch_edge = batch_edge.round().long().clamp(min=0, max=max(0, batch_size - 1))

        prior_edge = self._segment_normalize_feature_matrix(prior_edge, batch_edge)
        edge_size = scatter_add(
            torch.ones_like(he_ids, dtype=h_wsi.dtype),
            he_ids,
            dim=0,
            dim_size=num_edges,
        ).unsqueeze(-1)

        edge_pos = None
        node_pos = getattr(wsi_batch, "pos", None)
        if node_pos is None:
            node_pos = getattr(wsi_batch, "centroid", None)
        if node_pos is not None:
            edge_pos = scatter_mean(node_pos.float()[node_ids], he_ids, dim=0, dim_size=num_edges)

        orig_hyperedge_idx = getattr(wsi_batch, "orig_hyperedge_idx", None)
        if orig_hyperedge_idx is not None:
            try:
                orig_hyperedge_idx = orig_hyperedge_idx.to(device=h_wsi.device, dtype=torch.long).view(-1)
                if orig_hyperedge_idx.numel() != num_edges:
                    orig_hyperedge_idx = None
            except Exception:
                orig_hyperedge_idx = None

        return {
            "h": self.wsi_edge_proj(h_edge_mean),
            "prior": prior_edge,
            "batch": batch_edge,
            "pos": edge_pos,
            "size": edge_size,
            "orig_hyperedge_idx": orig_hyperedge_idx,
        }

    def _compute_gene_local_idx(self, batch_gene: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        node_counts = torch.bincount(batch_gene, minlength=batch_size)
        ptr = torch.zeros(batch_size + 1, device=batch_gene.device, dtype=torch.long)
        ptr[1:] = torch.cumsum(node_counts, dim=0)
        local_idx = torch.arange(batch_gene.numel(), device=batch_gene.device) - ptr[batch_gene]
        return local_idx, node_counts

    def _build_gene_prior_features(
        self,
        mask: torch.Tensor,
        expr_std_batch: torch.Tensor,
        batch_gene: torch.Tensor,
        wsi_signal: torch.Tensor,
    ) -> torch.Tensor:
        n = batch_gene.numel()
        if n <= 0:
            return expr_std_batch.new_zeros((0, 5))

        batch_size = int(expr_std_batch.size(0))
        local_idx, node_counts = self._compute_gene_local_idx(batch_gene, batch_size)
        local_idx = torch.clamp(local_idx, min=0, max=mask.size(0) - 1)

        pathway_mask = mask[local_idx]  # [N, G]
        expr_node = expr_std_batch[batch_gene]  # [N, G]
        pathway_size = pathway_mask.sum(dim=-1)
        pathway_size_n = _normalize01(pathway_size)

        coverage = ((expr_node.abs() > 1e-6).float() * pathway_mask).sum(dim=-1) / (pathway_size + 1e-6)
        train_var = (expr_node.pow(2) * pathway_mask).sum(dim=-1) / (pathway_size + 1e-6)
        train_var_n = _normalize01(train_var)
        stability = 1.0 / (1.0 + train_var)

        wsi_s = wsi_signal.squeeze(-1)[batch_gene]
        crossmodal_agreement = torch.exp(-torch.abs(train_var_n - wsi_s)).clamp(0.0, 1.0)

        return torch.stack([pathway_size_n, coverage, train_var_n, stability, crossmodal_agreement], dim=-1)

    def _resolve_wsi_prior_targets(
        self,
        wsi_batch: GeomBatch,
        prior_wsi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Build deterministic weak-prior targets from the configured prior table.
        """
        n = prior_wsi.size(0)
        if n <= 0:
            z = prior_wsi.new_zeros((0, 1))
            return {"alpha_prior": z, "q_prior": z, "alpha_conf": z, "q_conf": z}

        alpha_prior = prior_wsi.new_full((n, 1), 0.5)
        q_prior = prior_wsi.new_full((n, 1), 0.5)
        alpha_conf = prior_wsi.new_full((n, 1), 0.1)
        q_conf = prior_wsi.new_full((n, 1), 0.1)

        wsi_cfg = self.prior_table.get("wsi", {})
        default_cfg = wsi_cfg.get("default", {})
        alpha_prior.fill_(float(default_cfg.get("alpha_prior", 0.5)))
        q_prior.fill_(float(default_cfg.get("q_prior", 0.5)))
        alpha_conf.fill_(float(default_cfg.get("alpha_conf", 0.1)))
        q_conf.fill_(float(default_cfg.get("q_conf", 0.1)))

        # prior_wsi: [edge_score, heterogeneity, cluster_rarity, spatial_boundary, artifact_score, tumor_prob]
        edge_score = prior_wsi[:, 0:1]
        heterogeneity = prior_wsi[:, 1:2]
        cluster_rarity = prior_wsi[:, 2:3]
        spatial_boundary = prior_wsi[:, 3:4]
        artifact_score = prior_wsi[:, 4:5]
        tumor_prob = prior_wsi[:, 5:6]

        cont_cfg = wsi_cfg.get("continuous", {})
        alpha_w = cont_cfg.get("alpha_weights", {})
        q_w = cont_cfg.get("q_weights", {})

        wa_edge = float(alpha_w.get("edge_score", 0.25))
        wa_hetero = float(alpha_w.get("heterogeneity", 0.25))
        wa_rarity = float(alpha_w.get("cluster_rarity", 0.20))
        wa_boundary = float(alpha_w.get("spatial_boundary", 0.15))
        wa_tumor = float(alpha_w.get("tumor_prob", 0.15))
        sum_wa = max(1e-6, wa_edge + wa_hetero + wa_rarity + wa_boundary + wa_tumor)

        wq_art = float(q_w.get("artifact_score", 0.45))
        wq_blur = float(q_w.get("blur", 0.35))
        wq_inv_stain = float(q_w.get("inv_stain_qc", 0.20))
        sum_wq = max(1e-6, wq_art + wq_blur + wq_inv_stain)

        alpha_cont = (
            wa_edge * edge_score
            + wa_hetero * heterogeneity
            + wa_rarity * cluster_rarity
            + wa_boundary * spatial_boundary
            + wa_tumor * tumor_prob
        ) / sum_wa

        # blur proxy is available from artifact_score and uncertainty context.
        blur_proxy = torch.clamp(artifact_score + 0.30 * spatial_boundary, min=0.0, max=1.0)
        inv_stain_proxy = torch.clamp(artifact_score + 0.25 * (1.0 - tumor_prob), min=0.0, max=1.0)
        q_cont = (wq_art * artifact_score + wq_blur * blur_proxy + wq_inv_stain * inv_stain_proxy) / sum_wq

        alpha_mix = float(cont_cfg.get("alpha_mix", 1.0))
        q_mix = float(cont_cfg.get("q_mix", 1.0))
        alpha_prior = ((1.0 - alpha_mix) * alpha_prior + alpha_mix * alpha_cont).clamp(0.0, 1.0)
        q_prior = ((1.0 - q_mix) * q_prior + q_mix * q_cont).clamp(0.0, 1.0)

        alpha_conf_base = float(cont_cfg.get("alpha_conf_base", 0.10))
        alpha_conf_scale = float(cont_cfg.get("alpha_conf_scale", 0.60))
        q_conf_base = float(cont_cfg.get("q_conf_base", 0.15))
        q_conf_scale = float(cont_cfg.get("q_conf_scale", 0.75))
        alpha_conf_cont = (alpha_conf_base + alpha_conf_scale * (2.0 * (alpha_prior - 0.5).abs())).clamp(0.0, 1.0)
        q_conf_cont = (q_conf_base + q_conf_scale * (2.0 * (q_prior - 0.5).abs())).clamp(0.0, 1.0)
        alpha_conf = torch.maximum(alpha_conf, alpha_conf_cont)
        q_conf = torch.maximum(q_conf, q_conf_cont)

        return {
            "alpha_prior": alpha_prior.clamp(0.0, 1.0),
            "q_prior": q_prior.clamp(0.0, 1.0),
            "alpha_conf": alpha_conf.clamp(0.0, 1.0),
            "q_conf": q_conf.clamp(0.0, 1.0),
        }

    def _resolve_gene_prior_targets(self, prior_gene: torch.Tensor) -> Dict[str, torch.Tensor]:
        n = prior_gene.size(0)
        if n <= 0:
            z = prior_gene.new_zeros((0, 1))
            return {"alpha_prior": z, "q_prior": z, "alpha_conf": z, "q_conf": z}

        # prior_gene: [pathway_size, coverage, train_var, stability, agreement]
        pathway_size = prior_gene[:, 0:1]
        coverage = prior_gene[:, 1:2]
        train_var = prior_gene[:, 2:3]
        stability = prior_gene[:, 3:4]
        agreement = prior_gene[:, 4:5]

        gene_cfg = self.prior_table.get("gene", {})
        default_cfg = gene_cfg.get("default", {})
        alpha_prior = prior_gene.new_full((n, 1), float(default_cfg.get("alpha_prior", 0.5)))
        q_prior = prior_gene.new_full((n, 1), float(default_cfg.get("q_prior", 0.5)))
        alpha_conf = prior_gene.new_full((n, 1), float(default_cfg.get("alpha_conf", 0.2)))
        q_conf = prior_gene.new_full((n, 1), float(default_cfg.get("q_conf", 0.2)))

        alpha_h = (0.30 * stability + 0.30 * agreement + 0.20 * coverage + 0.20 * pathway_size).clamp(0.0, 1.0)
        q_h = (0.55 * (1.0 - coverage) + 0.35 * train_var + 0.10 * (1.0 - agreement)).clamp(0.0, 1.0)
        alpha_prior = 0.5 * alpha_prior + 0.5 * alpha_h
        q_prior = 0.5 * q_prior + 0.5 * q_h

        alpha_conf = torch.maximum(alpha_conf, (2.0 * (alpha_prior - 0.5).abs()).clamp(0.0, 1.0))
        q_conf = torch.maximum(q_conf, (2.0 * (q_prior - 0.5).abs()).clamp(0.0, 1.0))
        return {
            "alpha_prior": alpha_prior.clamp(0.0, 1.0),
            "q_prior": q_prior.clamp(0.0, 1.0),
            "alpha_conf": alpha_conf.clamp(0.0, 1.0),
            "q_conf": q_conf.clamp(0.0, 1.0),
        }

    def _aggregate_branch(
        self,
        h: torch.Tensor,
        node_out: Dict[str, torch.Tensor],
        batch_index: torch.Tensor,
        compute_losses: bool = True,
    ) -> Dict[str, torch.Tensor]:
        alpha = node_out["alpha"]
        u = node_out["u"]
        q = node_out["q"]
        s = node_out["s"]
        rho = node_out["rho"]

        agg = self._aggregate_from_components(h, alpha, u, q, s, rho, batch_index)

        if compute_losses:
            sparse_loss = segment_entropy(alpha.squeeze(-1), batch_index)
            alpha_prior_loss = weak_prior_distribution_loss(
                node_out["alpha"],
                node_out["alpha_prior"],
                node_out["alpha_prior_conf"],
                batch_index,
            )
            q_prior_loss = weak_prior_bce_loss(node_out["q"], node_out["q_prior"], node_out["q_prior_conf"])
        else:
            z = h.new_tensor(0.0)
            sparse_loss = z
            alpha_prior_loss = z
            q_prior_loss = z

        return {
            "z": agg["z"],
            "v_bio": agg["v_bio"],
            "q_case": agg["q_case"],
            "delta_r_node": agg["delta_r_node"],
            "sparse_loss": sparse_loss,
            "alpha_prior_loss": alpha_prior_loss,
            "q_prior_loss": q_prior_loss,
        }

    def _aggregate_from_components(
        self,
        h: torch.Tensor,
        alpha: torch.Tensor,
        u: torch.Tensor,
        q: torch.Tensor,
        s: torch.Tensor,
        rho: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z = scatter_add(alpha * h, batch_index, dim=0)
        delta_r_node = alpha * s
        v_bio_node = alpha.pow(2) * F.softplus(u) * rho * (1.0 - q)
        v_bio = scatter_add(v_bio_node, batch_index, dim=0)
        q_case = segment_topk_mean(q, batch_index, k_frac=self.q_topk_frac)
        return {
            "z": z,
            "delta_r_node": delta_r_node,
            "v_bio": v_bio,
            "q_case": q_case,
        }

    @staticmethod
    def _drop_and_renorm_alpha(
        alpha: torch.Tensor,
        batch_index: torch.Tensor,
        top_frac: float,
        drop_top: bool,
    ) -> torch.Tensor:
        """
        Remove top-k or bottom-k nodes per sample and renormalize remaining alpha.
        """
        if alpha.numel() == 0:
            return alpha
        out = alpha.clone()
        frac = float(np.clip(top_frac, 1e-3, 0.95))
        for bid in torch.unique(batch_index, sorted=True):
            mask = batch_index == bid
            vals = out[mask].view(-1)
            n = int(vals.numel())
            if n <= 1:
                continue
            k = max(1, int(round(n * frac)))
            k = min(k, n - 1)
            order = torch.argsort(vals, descending=True)
            drop_local = order[:k] if drop_top else order[-k:]
            vals_new = vals.clone()
            vals_new[drop_local] = 0.0
            s = vals_new.sum().clamp_min(1e-8)
            vals_new = vals_new / s
            out[mask] = vals_new.unsqueeze(-1)
        return out

    def _basic_fusion(
        self,
        z_wsi: torch.Tensor,
        mu_wsi: torch.Tensor,
        var_wsi: torch.Tensor,
        q_wsi: torch.Tensor,
        z_gene: torch.Tensor,
        mu_gene: torch.Tensor,
        var_gene: torch.Tensor,
        q_gene: torch.Tensor,
        z_cross: torch.Tensor,
        mu_cross: torch.Tensor,
        var_cross: torch.Tensor,
        q_cross: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tau_wsi = 1.0 / (var_wsi + 1e-6)
        tau_gene = 1.0 / (var_gene + 1e-6)
        tau_cross = 1.0 / (var_cross + 1e-6)
        tau_sum = tau_wsi + tau_gene + tau_cross + 1e-6
        mu_dec = (tau_wsi * mu_wsi + tau_gene * mu_gene + tau_cross * mu_cross) / tau_sum
        z_fuse = (tau_wsi * z_wsi + tau_gene * z_gene + tau_cross * z_cross) / tau_sum
        branch_disagreement = torch.cat([mu_wsi, mu_gene, mu_cross], dim=-1).var(
            dim=-1, unbiased=False, keepdim=True
        )
        var_dec = 1.0 / tau_sum + self.lambda_qc * torch.maximum(q_wsi, q_gene)
        one_w = torch.ones_like(tau_wsi)
        return {
            "mu_dec": mu_dec,
            "var_dec": var_dec.clamp_min(1e-6),
            "z_fuse": z_fuse,
            "branch_disagreement": branch_disagreement,
            "r_wsi": one_w,
            "r_gene": one_w,
            "r_cross": one_w,
            "tau_wsi": tau_wsi,
            "tau_gene": tau_gene,
            "tau_cross": tau_cross,
        }

    def forward_batch(
        self,
        wsi_batch: GeomBatch,
        gene_tpl_batch: GeomBatch,
        expr_std_batch: torch.Tensor,
        mask: torch.Tensor,
        return_explain: bool = False,
        need_attn_stats: bool = False,
        stochastic_passes: int = 1,
        enable_stochastic_variance: bool = False,
        enable_disagreement: bool = True,
        stochastic_latent: bool = True,
        compute_ctr_loss: bool = True,
        compute_aux_losses: bool = True,
    ):
        if expr_std_batch.dim() == 1:
            expr_std_batch = expr_std_batch.unsqueeze(0)
        expr_std_batch = expr_std_batch.float()
        batch_size = int(expr_std_batch.size(0))

        # 1) WSI branch
        x_wsi_proj = self.wsi_proj(wsi_batch.x)
        batch_wsi = self._build_batch_index(wsi_batch, x_wsi_proj.device)
        wsi_graphs = self._num_graphs_from_batch(batch_wsi, batch_size)
        if wsi_graphs != batch_size:
            raise RuntimeError(f"WSI batch mismatch: graphs={wsi_graphs} expr_batch={batch_size}")

        wsi_data_input = GeomData(x=x_wsi_proj, hyperedge_index=wsi_batch.hyperedge_index)
        wsi_data_input.batch = batch_wsi
        h_wsi_nodes = self.wsi_backbone(wsi_data_input)
        prior_wsi_nodes = self._build_wsi_prior_features(wsi_batch, x_wsi_proj, batch_wsi)
        wsi_edge_pack = self._build_wsi_hyperedge_tokens(
            wsi_batch,
            h_wsi_nodes,
            prior_wsi_nodes,
            batch_wsi,
            batch_size=batch_size,
        )
        h_wsi_edge = wsi_edge_pack["h"]
        prior_wsi_edge = wsi_edge_pack["prior"]
        batch_wsi_edge = wsi_edge_pack["batch"]
        wsi_node = self.wsi_heads(h_wsi_edge, prior_wsi_edge, batch_wsi_edge)
        wsi_prior_targets = self._resolve_wsi_prior_targets(wsi_batch, prior_wsi_edge)
        wsi_node["alpha_prior"] = wsi_prior_targets["alpha_prior"]
        wsi_node["q_prior"] = wsi_prior_targets["q_prior"]
        wsi_node["alpha_prior_conf"] = wsi_prior_targets["alpha_conf"]
        wsi_node["q_prior_conf"] = wsi_prior_targets["q_conf"]
        wsi_agg = self._aggregate_branch(h_wsi_edge, wsi_node, batch_wsi_edge, compute_losses=compute_aux_losses)
        wsi_signal = scatter_mean(prior_wsi_edge[:, 1:2], batch_wsi_edge, dim=0, dim_size=batch_size)

        # 2) Gene branch
        batch_gene = self._build_batch_index(gene_tpl_batch, expr_std_batch.device)
        gene_graphs = self._num_graphs_from_batch(batch_gene, batch_size)
        if gene_graphs != batch_size:
            raise RuntimeError(f"Gene batch mismatch: graphs={gene_graphs} expr_batch={batch_size}")

        num_pathways, num_genes = int(mask.size(0)), int(mask.size(1))
        if expr_std_batch.size(1) != num_genes:
            raise RuntimeError(
                f"Expression dim mismatch: expr={expr_std_batch.size(1)} mask_genes={num_genes}"
            )

        node_counts = torch.bincount(batch_gene, minlength=batch_size)

        if torch.all(node_counts == num_pathways):
            X_path = mask.unsqueeze(0) * expr_std_batch.unsqueeze(1)
            gene_input = X_path.reshape(batch_size * num_pathways, num_genes)
        else:
            rows = []
            for b in range(batch_size):
                n_nodes = int(node_counts[b].item())
                if n_nodes > num_pathways:
                    raise RuntimeError(
                        f"Gene node count exceeds mask rows: sample={b} n_nodes={n_nodes} mask_rows={num_pathways}"
                    )
                rows.append(mask[:n_nodes] * expr_std_batch[b].unsqueeze(0))
            gene_input = torch.cat(rows, dim=0) if rows else mask.new_zeros((0, num_genes))

        gene_x = self.gene_proj(gene_input)
        gene_data = GeomData(x=gene_x, hyperedge_index=gene_tpl_batch.hyperedge_index)
        gene_data.batch = batch_gene
        h_gene = self.gene_backbone(gene_data)
        prior_gene = self._build_gene_prior_features(mask, expr_std_batch, batch_gene, wsi_signal)
        gene_node = self.gene_heads(h_gene, prior_gene, batch_gene)
        gene_prior_targets = self._resolve_gene_prior_targets(prior_gene)
        gene_node["alpha_prior"] = gene_prior_targets["alpha_prior"]
        gene_node["q_prior"] = gene_prior_targets["q_prior"]
        gene_node["alpha_prior_conf"] = gene_prior_targets["alpha_conf"]
        gene_node["q_prior_conf"] = gene_prior_targets["q_conf"]
        gene_agg = self._aggregate_branch(h_gene, gene_node, batch_gene, compute_losses=compute_aux_losses)

        # 3) Cross branch (importance-guided cross-attention)
        need_attn = bool(return_explain or need_attn_stats)
        if need_attn:
            z_cross, cross_info = self.cross_attn(
                h_wsi_edge, wsi_node["alpha"], wsi_node["q"],
                h_gene, gene_node["alpha"], gene_node["q"],
                batch_wsi=batch_wsi_edge, batch_gene=batch_gene, return_attn=True
            )
        else:
            z_cross, cross_info = self.cross_attn(
                h_wsi_edge, wsi_node["alpha"], wsi_node["q"],
                h_gene, gene_node["alpha"], gene_node["q"],
                batch_wsi=batch_wsi_edge, batch_gene=batch_gene, return_attn=False
            )

        q_cross = torch.maximum(wsi_agg["q_case"], gene_agg["q_case"])
        v_bio_cross = 0.5 * (wsi_agg["v_bio"] + gene_agg["v_bio"])

        def _mu_from_alpha(alpha_wsi_cf: torch.Tensor, alpha_gene_cf: torch.Tensor) -> torch.Tensor:
            wsi_cf = self._aggregate_from_components(
                h_wsi_edge, alpha_wsi_cf, wsi_node["u"], wsi_node["q"], wsi_node["s"], wsi_node["rho"], batch_wsi_edge
            )
            gene_cf = self._aggregate_from_components(
                h_gene, alpha_gene_cf, gene_node["u"], gene_node["q"], gene_node["s"], gene_node["rho"], batch_gene
            )
            z_cross_cf, _ = self.cross_attn(
                h_wsi_edge, alpha_wsi_cf, wsi_node["q"],
                h_gene, alpha_gene_cf, gene_node["q"],
                batch_wsi=batch_wsi_edge, batch_gene=batch_gene, return_attn=False
            )

            q_cross_cf = torch.maximum(wsi_cf["q_case"], gene_cf["q_case"])
            v_bio_cross_cf = 0.5 * (wsi_cf["v_bio"] + gene_cf["v_bio"])

            pred_wsi_cf = self.wsi_branch_head(wsi_cf["z"], stochastic=False)
            pred_gene_cf = self.gene_branch_head(gene_cf["z"], stochastic=False)
            pred_cross_cf = self.cross_branch_head(z_cross_cf, stochastic=False)

            mu_wsi_cf, var_wsi_base_cf = pred_wsi_cf["mu"], pred_wsi_cf["var"]
            mu_gene_cf, var_gene_base_cf = pred_gene_cf["mu"], pred_gene_cf["var"]
            mu_cross_cf, var_cross_base_cf = pred_cross_cf["mu"], pred_cross_cf["var"]

            var_wsi_cf = var_wsi_base_cf + wsi_cf["v_bio"] + self.lambda_qc * wsi_cf["q_case"]
            var_gene_cf = var_gene_base_cf + gene_cf["v_bio"] + self.lambda_qc * gene_cf["q_case"]
            var_cross_cf = var_cross_base_cf + v_bio_cross_cf + self.lambda_qc * q_cross_cf

            if enable_disagreement:
                fusion_cf = self.fusion_layer(
                    wsi_cf["z"], mu_wsi_cf, var_wsi_cf, wsi_cf["q_case"],
                    gene_cf["z"], mu_gene_cf, var_gene_cf, gene_cf["q_case"],
                    z_cross_cf, mu_cross_cf, var_cross_cf, q_cross_cf
                )
            else:
                fusion_cf = self._basic_fusion(
                    wsi_cf["z"], mu_wsi_cf, var_wsi_cf, wsi_cf["q_case"],
                    gene_cf["z"], mu_gene_cf, var_gene_cf, gene_cf["q_case"],
                    z_cross_cf, mu_cross_cf, var_cross_cf, q_cross_cf
                )
            fused_cf = self.fused_head(fusion_cf["z_fuse"], stochastic=False)
            return 0.5 * (fusion_cf["mu_dec"] + fused_cf["mu"])

        stochastic_passes = max(1, int(stochastic_passes))
        enable_stochastic_variance = bool(enable_stochastic_variance) and stochastic_passes > 1
        enable_disagreement = bool(enable_disagreement)

        pred_wsi = self.wsi_branch_head(wsi_agg["z"], stochastic=stochastic_latent)
        pred_gene = self.gene_branch_head(gene_agg["z"], stochastic=stochastic_latent)
        pred_cross = self.cross_branch_head(z_cross, stochastic=stochastic_latent)

        mu_wsi, var_base_wsi = pred_wsi["mu"], pred_wsi["var"]
        mu_gene, var_base_gene = pred_gene["mu"], pred_gene["var"]
        mu_cross, var_base_cross = pred_cross["mu"], pred_cross["var"]

        kl_wsi = pred_wsi["kl"]
        kl_gene = pred_gene["kl"]
        kl_cross = pred_cross["kl"]

        var_wsi = var_base_wsi + wsi_agg["v_bio"] + self.lambda_qc * wsi_agg["q_case"]
        var_gene = var_base_gene + gene_agg["v_bio"] + self.lambda_qc * gene_agg["q_case"]
        var_cross = var_base_cross + v_bio_cross + self.lambda_qc * q_cross

        v_stoch_wsi = torch.zeros_like(var_wsi)
        v_stoch_gene = torch.zeros_like(var_gene)
        v_stoch_cross = torch.zeros_like(var_cross)
        v_stoch_dec = torch.zeros_like(var_cross)
        v_stoch_fh = torch.zeros_like(var_cross)
        cov_dec_fh = torch.zeros_like(var_cross)
        v_stoch_fused = torch.zeros_like(var_cross)

        if enable_stochastic_variance:
            mu_wsi_passes, mu_gene_passes, mu_cross_passes = [], [], []
            mu_dec_passes, mu_fh_passes = [], []
            for _ in range(stochastic_passes):
                mc_wsi = self.wsi_branch_head(
                    wsi_agg["z"], stochastic=True, input_dropout_p=self.stochastic_input_dropout
                )
                mc_gene = self.gene_branch_head(
                    gene_agg["z"], stochastic=True, input_dropout_p=self.stochastic_input_dropout
                )
                mc_cross = self.cross_branch_head(
                    z_cross, stochastic=True, input_dropout_p=self.stochastic_input_dropout
                )

                mu_wsi_t, var_wsi_t = mc_wsi["mu"], mc_wsi["var"] + wsi_agg["v_bio"] + self.lambda_qc * wsi_agg["q_case"]
                mu_gene_t, var_gene_t = mc_gene["mu"], mc_gene["var"] + gene_agg["v_bio"] + self.lambda_qc * gene_agg["q_case"]
                mu_cross_t, var_cross_t = mc_cross["mu"], mc_cross["var"] + v_bio_cross + self.lambda_qc * q_cross

                if enable_disagreement:
                    fusion_t = self.fusion_layer(
                        wsi_agg["z"], mu_wsi_t, var_wsi_t, wsi_agg["q_case"],
                        gene_agg["z"], mu_gene_t, var_gene_t, gene_agg["q_case"],
                        z_cross, mu_cross_t, var_cross_t, q_cross
                    )
                else:
                    fusion_t = self._basic_fusion(
                        wsi_agg["z"], mu_wsi_t, var_wsi_t, wsi_agg["q_case"],
                        gene_agg["z"], mu_gene_t, var_gene_t, gene_agg["q_case"],
                        z_cross, mu_cross_t, var_cross_t, q_cross
                    )
                fused_t = self.fused_head(
                    fusion_t["z_fuse"], stochastic=True, input_dropout_p=self.stochastic_input_dropout
                )

                mu_wsi_passes.append(mu_wsi_t)
                mu_gene_passes.append(mu_gene_t)
                mu_cross_passes.append(mu_cross_t)
                mu_dec_passes.append(fusion_t["mu_dec"])
                mu_fh_passes.append(fused_t["mu"])

            v_stoch_wsi = torch.var(torch.stack(mu_wsi_passes, dim=0), dim=0, unbiased=False)
            v_stoch_gene = torch.var(torch.stack(mu_gene_passes, dim=0), dim=0, unbiased=False)
            v_stoch_cross = torch.var(torch.stack(mu_cross_passes, dim=0), dim=0, unbiased=False)
            mu_dec_stack = torch.stack(mu_dec_passes, dim=0)
            mu_fh_stack = torch.stack(mu_fh_passes, dim=0)
            v_stoch_dec = torch.var(mu_dec_stack, dim=0, unbiased=False)
            v_stoch_fh = torch.var(mu_fh_stack, dim=0, unbiased=False)
            cov_dec_fh = torch.mean(
                (mu_dec_stack - mu_dec_stack.mean(dim=0, keepdim=True))
                * (mu_fh_stack - mu_fh_stack.mean(dim=0, keepdim=True)),
                dim=0,
            )
            # This is the stochastic part of Algorithm 1, step 7:
            # 1/4 * (Var[mu_dec] + Var[mu_fh] + 2 Cov[mu_dec, mu_fh]).
            v_stoch_fused = 0.25 * (v_stoch_dec + v_stoch_fh + 2.0 * cov_dec_fh)
            v_stoch_fused = v_stoch_fused.clamp_min(0.0)

        var_wsi = var_wsi + v_stoch_wsi
        var_gene = var_gene + v_stoch_gene
        var_cross = var_cross + v_stoch_cross

        if enable_disagreement:
            fusion = self.fusion_layer(
                wsi_agg["z"], mu_wsi, var_wsi, wsi_agg["q_case"],
                gene_agg["z"], mu_gene, var_gene, gene_agg["q_case"],
                z_cross, mu_cross, var_cross, q_cross
            )
        else:
            fusion = self._basic_fusion(
                wsi_agg["z"], mu_wsi, var_wsi, wsi_agg["q_case"],
                gene_agg["z"], mu_gene, var_gene, gene_agg["q_case"],
                z_cross, mu_cross, var_cross, q_cross
            )

        fused_pred = self.fused_head(fusion["z_fuse"], stochastic=stochastic_latent)
        mu_fh, var_fh = fused_pred["mu"], fused_pred["var"]
        kl_fused = fused_pred["kl"]

        mu = 0.5 * (fusion["mu_dec"] + mu_fh)
        # Algorithm 1, step 7 / variance-of-a-sum identity.
        var = combine_correlated_estimator_variances(
            fusion["var_dec"],
            var_fh,
            v_stoch_dec,
            v_stoch_fh,
            cov_dec_fh,
        ).clamp_min(1e-6)
        sigma = torch.sqrt(var)

        ctr_loss = mu.new_zeros(())
        if compute_ctr_loss:
            alpha_wsi_drop_top = self._drop_and_renorm_alpha(
                wsi_node["alpha"], batch_wsi_edge, top_frac=self.ctr_top_frac, drop_top=True
            )
            alpha_gene_drop_top = self._drop_and_renorm_alpha(
                gene_node["alpha"], batch_gene, top_frac=self.ctr_top_frac, drop_top=True
            )
            alpha_wsi_drop_bottom = self._drop_and_renorm_alpha(
                wsi_node["alpha"], batch_wsi_edge, top_frac=self.ctr_top_frac, drop_top=False
            )
            alpha_gene_drop_bottom = self._drop_and_renorm_alpha(
                gene_node["alpha"], batch_gene, top_frac=self.ctr_top_frac, drop_top=False
            )

            mu_full_cf = _mu_from_alpha(wsi_node["alpha"], gene_node["alpha"])
            mu_drop_top = _mu_from_alpha(alpha_wsi_drop_top, alpha_gene_drop_top)
            mu_drop_bottom = _mu_from_alpha(alpha_wsi_drop_bottom, alpha_gene_drop_bottom)
            ctr_loss = counterfactual_margin_loss(
                mu_full_cf,
                mu_drop_top,
                mu_drop_bottom,
                margin=self.ctr_margin,
            )

        if compute_aux_losses:
            align_loss = (1.0 - F.cosine_similarity(wsi_agg["z"], gene_agg["z"], dim=-1)).mean()
            sparse_loss = 0.5 * (wsi_agg["sparse_loss"] + gene_agg["sparse_loss"])
            alpha_prior_loss = 0.5 * (wsi_agg["alpha_prior_loss"] + gene_agg["alpha_prior_loss"])
            q_prior_loss = 0.5 * (wsi_agg["q_prior_loss"] + gene_agg["q_prior_loss"])

            # Optional regularizers used by the complete training objective.
            l3_loss = self.l3(h_gene, h_wsi_edge, batch_gene=batch_gene, batch_wsi=batch_wsi_edge)
            pathway_div_loss = pathway_query_diversity_loss(h_gene, batch_gene=batch_gene)
            kl_loss = (kl_wsi + kl_gene + kl_cross + kl_fused) / 4.0

            attn_entropy_loss = h_gene.new_tensor(0.0)
            if cross_info is not None:
                ent_terms = []
                if "g2w" in cross_info and "attn" in cross_info["g2w"]:
                    ent_terms.append(attention_entropy_regularization(cross_info["g2w"]["attn"]))
                if "w2g" in cross_info and "attn" in cross_info["w2g"]:
                    ent_terms.append(attention_entropy_regularization(cross_info["w2g"]["attn"]))
                if ent_terms:
                    attn_entropy_loss = torch.stack(ent_terms).mean()
        else:
            z = mu.new_zeros(())
            align_loss = z
            sparse_loss = z
            alpha_prior_loss = z
            q_prior_loss = z
            l3_loss = z
            pathway_div_loss = z
            kl_loss = z
            attn_entropy_loss = z

        out = {
            "mu": mu,
            "sigma": sigma,
            "risk_score": -mu,
            "mu_wsi": mu_wsi,
            "sigma_wsi": torch.sqrt(var_wsi.clamp_min(1e-6)),
            "mu_gene": mu_gene,
            "sigma_gene": torch.sqrt(var_gene.clamp_min(1e-6)),
            "mu_cross": mu_cross,
            "sigma_cross": torch.sqrt(var_cross.clamp_min(1e-6)),
            "l3_loss": l3_loss,
            "attn_entropy_loss": attn_entropy_loss,
            "pathway_div_loss": pathway_div_loss,
            "kl_wsi": kl_wsi,
            "kl_gene": kl_gene,
            "kl_cross": kl_cross,
            "kl_fused": kl_fused,
            "kl_loss": kl_loss,
            "sparse_loss": sparse_loss,
            "ctr_loss": ctr_loss,
            "alpha_prior_loss": alpha_prior_loss,
            "q_prior_loss": q_prior_loss,
            "align_loss": align_loss,
            "v_stoch_wsi": v_stoch_wsi,
            "v_stoch_gene": v_stoch_gene,
            "v_stoch_cross": v_stoch_cross,
            "v_stoch_dec": v_stoch_dec,
            "v_stoch_fh": v_stoch_fh,
            "cov_dec_fh": cov_dec_fh,
            "v_stoch_fused": v_stoch_fused,
            "var_dec": fusion["var_dec"],
            "var_fh": var_fh,
            "q_case_wsi": wsi_agg["q_case"],
            "q_case_gene": gene_agg["q_case"],
            "branch_disagreement": fusion["branch_disagreement"],
            "beta_wsi_edges": wsi_node["alpha"],
            "alpha_wsi_nodes": wsi_node["alpha"],
            "alpha_gene_nodes": gene_node["alpha"],
            "q_wsi_edges": wsi_node["q"],
            "q_wsi_nodes": wsi_node["q"],
            "q_gene_nodes": gene_node["q"],
            "u_wsi_edges": wsi_node["u"],
            "u_wsi_nodes": wsi_node["u"],
            "u_gene_nodes": gene_node["u"],
            "s_wsi_edges": wsi_node["s"],
            "s_wsi_nodes": wsi_node["s"],
            "s_gene_nodes": gene_node["s"],
            "delta_sigma2_wsi_edges": wsi_node["alpha"].pow(2) * F.softplus(wsi_node["u"]) * wsi_node["rho"] * (1.0 - wsi_node["q"]),
            "delta_sigma2_wsi_nodes": wsi_node["alpha"].pow(2) * F.softplus(wsi_node["u"]) * wsi_node["rho"] * (1.0 - wsi_node["q"]),
            "delta_sigma2_gene_nodes": gene_node["alpha"].pow(2) * F.softplus(gene_node["u"]) * gene_node["rho"] * (1.0 - gene_node["q"]),
        }
        if return_explain:
            out.update({
                "wsi_explain_unit": "hyperedge",
                "h_wsi": h_wsi_edge,
                "h_wsi_edges": h_wsi_edge,
                "h_wsi_nodes": h_wsi_nodes,
                "h_gene": h_gene,
                "cross_attn": cross_info,
                "batch_wsi": batch_wsi_edge,
                "batch_wsi_edges": batch_wsi_edge,
                "batch_wsi_nodes": batch_wsi,
                "batch_gene": batch_gene,
                "wsi_pos": wsi_edge_pack.get("pos", None),
                "wsi_edge_pos": wsi_edge_pack.get("pos", None),
                "wsi_node_pos": getattr(wsi_batch, "pos", None),
                "wsi_orig_node_idx": getattr(wsi_batch, "orig_node_idx", None),
                "wsi_orig_hyperedge_idx": wsi_edge_pack.get("orig_hyperedge_idx", None),
                "wsi_edge_size": wsi_edge_pack.get("size", None),
            })
        return out

    def forward_single(
        self,
        wsi_data,
        gene_tpl,
        expr_std,
        mask,
        return_explain: bool = False,
        need_attn_stats: bool = False,
        stochastic_passes: int = 1,
        enable_stochastic_variance: bool = False,
        enable_disagreement: bool = True,
        stochastic_latent: bool = True,
    ):
        wsi_batch = GeomBatch.from_data_list([wsi_data])
        gene_batch = GeomBatch.from_data_list([gene_tpl])
        if expr_std.dim() == 1:
            expr_std = expr_std.unsqueeze(0)
        return self.forward_batch(
            wsi_batch=wsi_batch.to(expr_std.device),
            gene_tpl_batch=gene_batch.to(expr_std.device),
            expr_std_batch=expr_std,
            mask=mask,
            return_explain=return_explain,
            need_attn_stats=need_attn_stats,
            stochastic_passes=stochastic_passes,
            enable_stochastic_variance=enable_stochastic_variance,
            enable_disagreement=enable_disagreement,
            stochastic_latent=stochastic_latent,
        )
