"""Neural layers, attention modules, survival heads, and fusion."""

from __future__ import annotations

from .common import *
from .metrics import kl_divergence_standard_normal

class L3Regularization(nn.Module):
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
    def forward(
        self,
        h_gene: torch.Tensor,
        h_wsi: torch.Tensor,
        batch_gene: Optional[torch.Tensor] = None,
        batch_wsi: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if h_gene.numel() == 0 or h_wsi.numel() == 0:
            return h_gene.new_tensor(0.0)

        if batch_gene is None or batch_wsi is None:
            center_g = h_gene.mean(dim=0)
            center_w = h_wsi.mean(dim=0)
            return torch.norm(center_g - center_w, p=2) * self.weight

        center_g = scatter_mean(h_gene, batch_gene, dim=0)
        center_w = scatter_mean(h_wsi, batch_wsi, dim=0)
        b = min(center_g.size(0), center_w.size(0))
        if b <= 0:
            return h_gene.new_tensor(0.0)
        d = torch.norm(center_g[:b] - center_w[:b], p=2, dim=-1)
        return d.mean() * self.weight


def segment_softmax(logits: torch.Tensor, batch_index: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    if logits.numel() == 0:
        return logits
    if batch_index.numel() != logits.numel():
        raise RuntimeError(f"segment_softmax size mismatch: logits={logits.numel()} batch={batch_index.numel()}")

    order = torch.argsort(batch_index)
    sorted_batch = batch_index[order]
    sorted_logits = logits[order]
    out_sorted = torch.empty_like(sorted_logits)

    _, counts = torch.unique_consecutive(sorted_batch, return_counts=True)
    start = 0
    for c in counts.tolist():
        end = start + int(c)
        out_sorted[start:end] = torch.softmax(sorted_logits[start:end], dim=0)
        start = end

    out = torch.empty_like(logits)
    out[order] = out_sorted
    return out.clamp_min(eps)


def segment_entropy(alpha: torch.Tensor, batch_index: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if alpha.numel() == 0:
        return alpha.new_tensor(0.0)
    if batch_index.numel() != alpha.numel():
        raise RuntimeError(f"segment_entropy size mismatch: alpha={alpha.numel()} batch={batch_index.numel()}")

    order = torch.argsort(batch_index)
    sorted_batch = batch_index[order]
    sorted_alpha = alpha[order]
    ent_terms: List[torch.Tensor] = []

    _, counts = torch.unique_consecutive(sorted_batch, return_counts=True)
    start = 0
    for c in counts.tolist():
        end = start + int(c)
        if c > 1:
            p = torch.clamp(sorted_alpha[start:end], min=eps)
            ent = -(p * torch.log(p)).sum() / math.log(float(c))
            ent_terms.append(ent)
        start = end
    if not ent_terms:
        return alpha.new_tensor(0.0)
    return torch.stack(ent_terms).mean()


def segment_topk_mean(x: torch.Tensor, batch_index: torch.Tensor, k_frac: float = 0.1) -> torch.Tensor:
    if x.dim() > 1 and x.size(-1) == 1:
        x = x.squeeze(-1)
    if x.numel() == 0:
        bsz = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        return x.new_zeros((bsz, 1))

    bsz = int(batch_index.max().item()) + 1
    out = x.new_zeros((bsz, 1))
    for bid in range(bsz):
        vals = x[batch_index == bid]
        if vals.numel() == 0:
            continue
        k = max(1, int(round(vals.numel() * k_frac)))
        topk = torch.topk(vals, k=k, largest=True).values
        out[bid, 0] = topk.mean()
    return out


def weak_prior_bce_loss(pred: torch.Tensor, prior: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_tensor(0.0)
    conf = conf.clamp(0.0, 1.0)
    pred_f = pred.float().clamp(1e-6, 1.0 - 1e-6)
    prior_f = prior.float().clamp(1e-6, 1.0 - 1e-6)
    conf_f = conf.float()
    # BCE(prob, target) is not AMP-safe under autocast; force this block to FP32.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        bce = F.binary_cross_entropy(pred_f, prior_f, reduction="none")
        denom = conf_f.sum() + 1e-6
        return (bce * conf_f).sum() / denom


def weak_prior_distribution_loss(
    alpha: torch.Tensor,
    prior: torch.Tensor,
    conf: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    """
    Match normalized alpha/beta to a normalized weak-prior distribution per patient.
    This avoids treating mutually-normalized importance weights as independent
    binary labels.
    """
    if alpha.numel() == 0:
        return alpha.new_tensor(0.0)
    alpha_1d = alpha.view(-1).float().clamp_min(1e-12)
    prior_1d = prior.view(-1).float().clamp(0.0, 1.0)
    conf_1d = conf.view(-1).float().clamp(0.0, 1.0)
    losses: List[torch.Tensor] = []
    with torch.autocast(device_type=alpha.device.type, enabled=False):
        for bid in torch.unique(batch_index, sorted=True):
            m = batch_index == bid
            if int(m.sum().item()) <= 0:
                continue
            pred = alpha_1d[m]
            pred = pred / pred.sum().clamp_min(1e-12)
            target_raw = prior_1d[m].clamp_min(1e-6)
            target = target_raw / target_raw.sum().clamp_min(1e-12)
            weight = conf_1d[m].mean().clamp_min(0.05)
            losses.append(weight * (-(target.detach() * torch.log(pred)).sum()))
    if not losses:
        return alpha.new_tensor(0.0)
    return torch.stack(losses).mean()


def counterfactual_margin_loss(
    mu_full: torch.Tensor,
    mu_drop_top: torch.Tensor,
    mu_drop_bottom: torch.Tensor,
    margin: float = 0.1
) -> torch.Tensor:
    if mu_full.numel() == 0:
        return mu_full.new_tensor(0.0)
    delta_top = (mu_full - mu_drop_top).abs()
    delta_bottom = (mu_full - mu_drop_bottom).abs()
    return F.relu(float(margin) - (delta_top - delta_bottom)).mean()

class UncertaintyAwareHypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv = HypergraphConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, hyperedge_index):
        x = self.conv(x, hyperedge_index)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)

class HGNodeBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList([
            UncertaintyAwareHypergraphConv(hidden_channels, hidden_channels, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, data: GeomData) -> torch.Tensor:
        h = self.input_proj(data.x.float())
        for conv in self.convs:
            h = conv(h, data.hyperedge_index)
        return h


class ImportanceQualityHeads(nn.Module):
    """
    Output node-level tuple:
    h, alpha, u(aleatoric), q(quality/artifact), s(direction), rho(redundancy).
    """

    def __init__(self, hidden_dim: int, prior_dim: int, dropout: float = 0.2):
        super().__init__()
        self.prior_dim = prior_dim
        self.prior_proj = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.alea_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.redundancy_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Prior prediction targets for weak prior regularization.
        self.alpha_prior_head = nn.Sequential(nn.Linear(prior_dim, 1), nn.Sigmoid())
        self.q_prior_head = nn.Sequential(nn.Linear(prior_dim, 1), nn.Sigmoid())

    def forward(self, h: torch.Tensor, prior_feat: torch.Tensor, batch_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        if prior_feat.size(-1) != self.prior_dim:
            raise RuntimeError(f"prior dim mismatch: got={prior_feat.size(-1)} expected={self.prior_dim}")

        p = self.prior_proj(prior_feat)
        hp = torch.cat([h, p], dim=-1)
        q = torch.sigmoid(self.quality_head(hp))
        u = F.softplus(self.alea_head(hp))
        rho = torch.sigmoid(self.redundancy_head(hp))
        s = torch.tanh(self.direction_head(hp))

        ctx = scatter_mean(h, batch_index, dim=0)
        ctx_node = ctx[batch_index]
        imp_in = torch.cat([h, ctx_node, p, q], dim=-1)
        alpha_logits = self.importance_head(imp_in).squeeze(-1)
        alpha = segment_softmax(alpha_logits, batch_index).unsqueeze(-1)

        alpha_prior = self.alpha_prior_head(prior_feat)
        q_prior = self.q_prior_head(prior_feat)
        conf_alpha = (2.0 * (alpha_prior - 0.5).abs()).clamp(0.0, 1.0)
        conf_q = (2.0 * (q_prior - 0.5).abs()).clamp(0.0, 1.0)

        return {
            "alpha": alpha,
            "u": u,
            "q": q,
            "s": s,
            "rho": rho,
            "alpha_prior": alpha_prior.detach(),
            "q_prior": q_prior.detach(),
            "alpha_prior_conf": conf_alpha.detach(),
            "q_prior_conf": conf_q.detach(),
        }


class ImportanceGuidedCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        lambda_alpha_q: float = 0.6,
        lambda_alpha_k: float = 0.6,
        lambda_q: float = 0.8,
        attn_temperature: float = 1.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.lambda_alpha_q = float(lambda_alpha_q)
        self.lambda_alpha_k = float(lambda_alpha_k)
        self.lambda_q = float(lambda_q)
        temp = max(float(attn_temperature), 1e-3)
        self.register_buffer("attn_temperature_tensor", torch.tensor(temp, dtype=torch.float32))

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _forward_one_dense(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        alpha_q: torch.Tensor,
        alpha_kv: torch.Tensor,
        q_q: torch.Tensor,
        q_kv: torch.Tensor,
        batch_q: torch.Tensor,
        batch_kv: torch.Tensor,
        return_attn: bool = False
    ):
        Q = self.q_proj(x_q).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(x_kv).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(x_kv).view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        temp = torch.clamp(self.attn_temperature_tensor, min=1e-3)
        attn_logits = (Q @ K.transpose(-2, -1)) * self.scale / temp
        same_mask = batch_q.view(-1, 1).eq(batch_kv.view(1, -1))
        valid_mask = same_mask.unsqueeze(0)  # [1, Nq, Nk]

        bias_q = self.lambda_alpha_q * torch.log(alpha_q.clamp_min(1e-8)).view(-1)  # [Nq]
        bias_k = self.lambda_alpha_k * torch.log(alpha_kv.clamp_min(1e-8)).view(-1)  # [Nk]
        qual_pen = self.lambda_q * (q_q.view(-1, 1) + q_kv.view(1, -1))  # [Nq, Nk]

        attn_logits = (
            attn_logits
            + bias_q.view(1, -1, 1)
            + bias_k.view(1, 1, -1)
            - qual_pen.unsqueeze(0)
        )
        attn_logits = attn_logits.masked_fill(~valid_mask, -1e9)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = attn * valid_mask.to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)

        out_nodes = (attn @ V).transpose(0, 1).reshape(-1, self.dim)
        out_nodes = self.out_proj(out_nodes)

        alpha_sum = scatter_add(alpha_q, batch_q, dim=0).clamp_min(1e-6)
        w = alpha_q / alpha_sum[batch_q]
        z = scatter_add(out_nodes * w, batch_q, dim=0)

        if return_attn:
            info = {"attn": attn, "weights": w.squeeze(-1)}
            if not self.training:
                info["attn_logits"] = attn_logits
            return z, info
        return z, None

    def _forward_one_blockwise(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        alpha_q: torch.Tensor,
        alpha_kv: torch.Tensor,
        q_q: torch.Tensor,
        q_kv: torch.Tensor,
        batch_q: torch.Tensor,
        batch_kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Fast path for training/eval when attention maps are not requested.
        Compute attention per-sample block to avoid dense cross-sample Nq x Nk matrix work.
        """
        Q = self.q_proj(x_q).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(x_kv).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(x_kv).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        temp = torch.clamp(self.attn_temperature_tensor, min=1e-3)

        out_nodes = x_q.new_zeros((x_q.size(0), self.dim), dtype=Q.dtype)
        for bid in torch.unique(batch_q, sorted=True):
            idx_q = (batch_q == bid).nonzero(as_tuple=False).view(-1)
            if idx_q.numel() == 0:
                continue
            idx_k = (batch_kv == bid).nonzero(as_tuple=False).view(-1)
            if idx_k.numel() == 0:
                continue

            Qb = Q[:, idx_q, :]  # [H, Nq_b, D]
            Kb = K[:, idx_k, :]  # [H, Nk_b, D]
            Vb = V[:, idx_k, :]  # [H, Nk_b, D]

            attn_logits = (Qb @ Kb.transpose(-2, -1)) * self.scale / temp
            bias_q = self.lambda_alpha_q * torch.log(alpha_q[idx_q].clamp_min(1e-8)).view(1, -1, 1)
            bias_k = self.lambda_alpha_k * torch.log(alpha_kv[idx_k].clamp_min(1e-8)).view(1, 1, -1)
            qual_pen = self.lambda_q * (
                q_q[idx_q].view(1, -1, 1) + q_kv[idx_k].view(1, 1, -1)
            )
            attn_logits = attn_logits + bias_q + bias_k - qual_pen

            attn = torch.softmax(attn_logits, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
            attn = self.dropout(attn)

            out_b = (attn @ Vb).transpose(0, 1).reshape(-1, self.dim)
            out_nodes[idx_q] = out_b.to(dtype=out_nodes.dtype)

        out_nodes = self.out_proj(out_nodes)
        alpha_sum = scatter_add(alpha_q, batch_q, dim=0).clamp_min(1e-6)
        w = alpha_q / alpha_sum[batch_q]
        z = scatter_add(out_nodes * w, batch_q, dim=0)
        return z, None

    def _forward_one(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        alpha_q: torch.Tensor,
        alpha_kv: torch.Tensor,
        q_q: torch.Tensor,
        q_kv: torch.Tensor,
        batch_q: torch.Tensor,
        batch_kv: torch.Tensor,
        return_attn: bool = False
    ):
        # The dense path returns attention maps for interpretation.
        if return_attn:
            return self._forward_one_dense(
                x_q, x_kv, alpha_q, alpha_kv, q_q, q_kv, batch_q, batch_kv, return_attn=True
            )
        # Fast blockwise path for training and standard validation.
        return self._forward_one_blockwise(
            x_q, x_kv, alpha_q, alpha_kv, q_q, q_kv, batch_q, batch_kv
        )

    def forward(
        self,
        h_wsi: torch.Tensor,
        alpha_wsi: torch.Tensor,
        q_wsi: torch.Tensor,
        h_gene: torch.Tensor,
        alpha_gene: torch.Tensor,
        q_gene: torch.Tensor,
        batch_wsi: torch.Tensor,
        batch_gene: torch.Tensor,
        return_attn: bool = False
    ):
        z_w2g, info_w2g = self._forward_one(
            h_wsi, h_gene, alpha_wsi, alpha_gene, q_wsi, q_gene, batch_wsi, batch_gene, return_attn=return_attn
        )
        z_g2w, info_g2w = self._forward_one(
            h_gene, h_wsi, alpha_gene, alpha_wsi, q_gene, q_wsi, batch_gene, batch_wsi, return_attn=return_attn
        )
        z_cross = 0.5 * (z_w2g + z_g2w)
        if return_attn:
            return z_cross, {"w2g": info_w2g, "g2w": info_g2w}
        return z_cross, None


class BranchSurvivalHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, dropout: float = 0.2):
        super().__init__()
        self.latent_mu = nn.Linear(in_channels, in_channels)
        self.latent_logvar = nn.Linear(in_channels, in_channels)
        self.latent_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.backbone = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_channels // 2, 1)
        self.var_head = nn.Linear(hidden_channels // 2, 1)

    def _sample_latent(self, z: torch.Tensor, stochastic: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_mu = self.latent_mu(z)
        q_logvar = torch.clamp(self.latent_logvar(z), min=LOGVAR_MIN, max=LOGVAR_MAX)
        if stochastic:
            eps = torch.randn_like(q_mu)
            z_lat = q_mu + torch.exp(0.5 * q_logvar) * eps
        else:
            z_lat = q_mu
        kl = kl_divergence_standard_normal(q_mu, q_logvar)
        return z_lat, q_mu, q_logvar, kl

    def forward(
        self,
        z: torch.Tensor,
        stochastic: bool = True,
        input_dropout_p: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        if input_dropout_p > 0.0:
            z = F.dropout(z, p=float(input_dropout_p), training=True)
        z_lat, q_mu, q_logvar, kl = self._sample_latent(z, stochastic=stochastic)
        h = self.backbone(self.latent_proj(z_lat))
        mu = self.mu_head(h)
        var = F.softplus(self.var_head(h)) + 1e-6
        return {
            "mu": mu,
            "var": var,
            "kl": kl,
            "latent_mu": q_mu,
            "latent_logvar": q_logvar,
        }


class FusedSurvivalHead(BranchSurvivalHead):
    pass


class ReliabilityDisagreementFusion(nn.Module):
    def __init__(self, dim: int, lambda_conf: float = 0.15, lambda_qc: float = 0.1):
        super().__init__()
        self.lambda_conf = float(lambda_conf)
        self.lambda_qc = float(lambda_qc)
        self.rel_mlp = nn.Sequential(
            nn.Linear(dim + 4, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def _reliability(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        var: torch.Tensor,
        q_case: torch.Tensor,
        branch_disagreement: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [z, mu, torch.log(var.clamp_min(1e-8)), q_case, branch_disagreement],
            dim=-1,
        )
        return torch.sigmoid(self.rel_mlp(x))

    def forward(
        self,
        z_wsi: torch.Tensor, mu_wsi: torch.Tensor, var_wsi: torch.Tensor, q_wsi: torch.Tensor,
        z_gene: torch.Tensor, mu_gene: torch.Tensor, var_gene: torch.Tensor, q_gene: torch.Tensor,
        z_cross: torch.Tensor, mu_cross: torch.Tensor, var_cross: torch.Tensor, q_cross: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mu_stack = torch.cat([mu_wsi, mu_gene, mu_cross], dim=-1)
        branch_disagreement = mu_stack.var(dim=-1, unbiased=False, keepdim=True)

        r_wsi = self._reliability(z_wsi, mu_wsi, var_wsi, q_wsi, branch_disagreement)
        r_gene = self._reliability(z_gene, mu_gene, var_gene, q_gene, branch_disagreement)
        r_cross = self._reliability(z_cross, mu_cross, var_cross, q_cross, branch_disagreement)

        tau_wsi = r_wsi / (var_wsi + 1e-6)
        tau_gene = r_gene / (var_gene + 1e-6)
        tau_cross = r_cross / (var_cross + 1e-6)
        tau_sum = tau_wsi + tau_gene + tau_cross + 1e-6

        mu_dec = (tau_wsi * mu_wsi + tau_gene * mu_gene + tau_cross * mu_cross) / tau_sum
        var_dec = (
            1.0 / tau_sum
            + self.lambda_conf * branch_disagreement
            + self.lambda_qc * torch.maximum(q_wsi, q_gene)
        )
        z_fuse = (tau_wsi * z_wsi + tau_gene * z_gene + tau_cross * z_cross) / tau_sum

        return {
            "mu_dec": mu_dec,
            "var_dec": var_dec.clamp_min(1e-6),
            "z_fuse": z_fuse,
            "branch_disagreement": branch_disagreement,
            "r_wsi": r_wsi,
            "r_gene": r_gene,
            "r_cross": r_cross,
            "tau_wsi": tau_wsi,
            "tau_gene": tau_gene,
            "tau_cross": tau_cross,
        }
