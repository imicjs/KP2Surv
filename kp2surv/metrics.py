"""Censoring-aware losses, calibration metrics, and regularizers."""

from __future__ import annotations

from .common import *

def _raise_if_non_finite(name: str, tensor: torch.Tensor, context: str = "") -> None:
    if torch.isfinite(tensor).all():
        return
    finite_mask = torch.isfinite(tensor)
    if finite_mask.any():
        t_min = float(tensor[finite_mask].min().detach().cpu().item())
        t_max = float(tensor[finite_mask].max().detach().cpu().item())
    else:
        t_min = float("nan")
        t_max = float("nan")
    raise FloatingPointError(
        f"Non-finite values in {name}. context={context} shape={tuple(tensor.shape)} "
        f"finite_min={t_min:.6g} finite_max={t_max:.6g}"
    )


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)

    if logvar.shape != mu.shape:
        if logvar.dim() == mu.dim() and logvar.size(-1) == 1:
            logvar = logvar.expand_as(mu)
        else:
            raise ValueError(f"Incompatible KL shapes: mu={tuple(mu.shape)}, logvar={tuple(logvar.shape)}")

    kl = 0.5 * (mu.pow(2) + torch.exp(logvar) - 1.0 - logvar)
    if kl.dim() == 1:
        return kl.mean()
    return kl.reshape(kl.size(0), -1).sum(dim=1).mean()


def combine_correlated_estimator_variances(
    var_dec: torch.Tensor,
    var_fh: torch.Tensor,
    var_stoch_dec: torch.Tensor,
    var_stoch_fh: torch.Tensor,
    cov_dec_fh: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 1 variance for the equally weighted decision/fused estimators."""
    return 0.25 * (
        var_dec
        + var_stoch_dec
        + var_fh
        + var_stoch_fh
        + 2.0 * cov_dec_fh
    )


def lognormal_nll_loss(mu, sigma, time, event):
    mu = mu.float()
    sigma = torch.clamp(sigma.float(), min=1e-6, max=1e3)
    time = torch.as_tensor(time, dtype=mu.dtype, device=mu.device)
    event = torch.as_tensor(event, dtype=mu.dtype, device=mu.device)

    while time.dim() < mu.dim():
        time = time.unsqueeze(-1)
    while event.dim() < mu.dim():
        event = event.unsqueeze(-1)

    time = torch.clamp(time, min=1e-8)
    event = torch.clamp(event, min=0.0, max=1.0)
    log_t = torch.log(time)
    z = (log_t - mu) / sigma

    log_pdf = -torch.log(sigma) - log_t - 0.5 * z.pow(2) - 0.5 * math.log(2.0 * math.pi)
    survival_prob = 0.5 * torch.special.erfc(z / math.sqrt(2.0))
    survival_prob = torch.clamp(survival_prob, min=1e-12, max=1.0)
    log_surv = torch.log(survival_prob)

    loss = -(event * log_pdf + (1.0 - event) * log_surv)
    _raise_if_non_finite("lognormal_nll_loss", loss)
    return loss.mean()


def _lognormal_nll_numpy(mu_arr: np.ndarray, sigma_arr: np.ndarray, time_arr: np.ndarray, event_arr: np.ndarray) -> float:
    mu = torch.tensor(mu_arr, dtype=torch.float32)
    sigma = torch.tensor(sigma_arr, dtype=torch.float32)
    time = torch.tensor(time_arr, dtype=torch.float32)
    event = torch.tensor(event_arr, dtype=torch.float32)
    return float(lognormal_nll_loss(mu.unsqueeze(-1), sigma.unsqueeze(-1), time, event).item())


def expected_calibration_error(conf: np.ndarray, obs: np.ndarray, weights: np.ndarray, n_bins: int = 10) -> float:
    if conf.size == 0 or obs.size == 0:
        return float("nan")
    n_bins = max(2, int(n_bins))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    w_sum_all = float(np.sum(weights)) + 1e-12
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        w = weights[mask]
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            continue
        conf_i = float(np.sum(conf[mask] * w) / w_sum)
        obs_i = float(np.sum(obs[mask] * w) / w_sum)
        ece += abs(conf_i - obs_i) * (w_sum / w_sum_all)
    return float(ece)


def _fit_reverse_km(time_arr: np.ndarray, event_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate censoring survival G(t)=P(C>=t) using reverse Kaplan-Meier."""
    n = int(time_arr.size)
    if n <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    order = np.argsort(time_arr)
    t = np.asarray(time_arr[order], dtype=np.float64)
    c_event = (1.0 - np.asarray(event_arr[order], dtype=np.float64) > 0.5).astype(np.int32)
    uniq = np.unique(t)

    g_vals = []
    surv = 1.0
    at_risk = float(n)
    for ut in uniq:
        m = t == ut
        d = float(np.sum(c_event[m]))
        if at_risk > 0.0 and d > 0.0:
            surv *= max(0.0, 1.0 - d / at_risk)
        g_vals.append(surv)
        at_risk -= float(np.sum(m))
    return uniq.astype(np.float64), np.asarray(g_vals, dtype=np.float64)


def _km_lookup(uniq_t: np.ndarray, surv_vals: np.ndarray, t: float, left_limit: bool = False) -> float:
    if uniq_t.size == 0:
        return 1.0
    side = "left" if left_limit else "right"
    idx = int(np.searchsorted(uniq_t, float(t), side=side)) - 1
    if idx < 0:
        return 1.0
    return float(surv_vals[min(idx, surv_vals.size - 1)])


def _lognormal_survival_np(mu_arr: np.ndarray, sigma_arr: np.ndarray, horizon_arr: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu_arr, dtype=np.float64).reshape(-1, 1)
    sigma = np.clip(np.asarray(sigma_arr, dtype=np.float64).reshape(-1, 1), 1e-6, 1e3)
    h = np.clip(np.asarray(horizon_arr, dtype=np.float64).reshape(1, -1), 1e-8, None)
    z = (np.log(h) - mu) / sigma
    # 0.5 * erfc(z/sqrt(2)) using torch for numerical stability without scipy dependency.
    z_t = torch.from_numpy(z.astype(np.float32))
    s = 0.5 * torch.special.erfc(z_t / math.sqrt(2.0))
    return np.clip(s.numpy().astype(np.float64), 1e-12, 1.0)


def _build_time_grid(time_arr: np.ndarray, event_arr: np.ndarray, num_grid: int) -> np.ndarray:
    ev_t = np.asarray(time_arr[event_arr > 0.5], dtype=np.float64)
    if ev_t.size < 5:
        ev_t = np.asarray(time_arr, dtype=np.float64)
    if ev_t.size == 0:
        return np.asarray([], dtype=np.float64)
    lo = float(np.quantile(ev_t, 0.1))
    hi = float(np.quantile(ev_t, 0.9))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(ev_t))
        hi = float(np.max(ev_t))
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(max(1e-8, lo), max(lo + 1e-8, hi), num=max(5, int(num_grid)))


def _ipcw_terms_at_horizon(
    time_arr: np.ndarray,
    event_arr: np.ndarray,
    surv_pred_h: np.ndarray,
    horizon: float,
    g_t: np.ndarray,
    g_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (weighted_sqerr, event_obs, event_weight) for horizon.
    weighted_sqerr is IPCW Brier contribution per sample (0 if not valid).
    event_obs/event_weight are for calibration of event probability at horizon.
    """
    n = int(time_arr.size)
    sqerr = np.zeros(n, dtype=np.float64)
    event_obs = np.zeros(n, dtype=np.float64)
    event_w = np.zeros(n, dtype=np.float64)

    g_h = max(_km_lookup(g_t, g_s, horizon, left_limit=False), 1e-6)
    y_surv = (time_arr > horizon).astype(np.float64)

    for i in range(n):
        ti = float(time_arr[i])
        di = float(event_arr[i])
        s = float(np.clip(surv_pred_h[i], 1e-12, 1.0))
        if ti <= horizon and di > 0.5:
            g = max(_km_lookup(g_t, g_s, ti, left_limit=True), 1e-6)
            w = 1.0 / g
            sqerr[i] = w * ((0.0 - s) ** 2)
            event_obs[i] = 1.0
            event_w[i] = w
        elif ti > horizon:
            w = 1.0 / g_h
            sqerr[i] = w * ((1.0 - s) ** 2)
            event_obs[i] = 0.0
            event_w[i] = w
        else:
            # censored before horizon: zero IPCW contribution
            sqerr[i] = 0.0
            event_obs[i] = 0.0
            event_w[i] = 0.0
    return sqerr, event_obs, event_w


def uncertainty_calibration_metrics(
    mu_arr: np.ndarray,
    sigma_arr: np.ndarray,
    time_arr: np.ndarray,
    event_arr: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    n = int(time_arr.size)
    if n < 10:
        return {"ece": float("nan"), "unc_err_corr": float("nan"), "uncertainty_error_gap": float("nan")}

    g_t, g_s = _fit_reverse_km(time_arr, event_arr)
    horizons = _build_time_grid(time_arr, event_arr, num_grid=6)
    if horizons.size == 0:
        return {"ece": float("nan"), "unc_err_corr": float("nan"), "uncertainty_error_gap": float("nan")}

    surv_mat = _lognormal_survival_np(mu_arr, sigma_arr, horizons)

    ece_terms = []
    per_sample_err = np.zeros(n, dtype=np.float64)
    for j, h in enumerate(horizons):
        sqerr, event_obs, event_w = _ipcw_terms_at_horizon(
            time_arr, event_arr, surv_mat[:, j], float(h), g_t, g_s
        )
        per_sample_err += sqerr

        pred_event = 1.0 - surv_mat[:, j]
        valid = event_w > 0.0
        if np.any(valid):
            ece_h = expected_calibration_error(
                pred_event[valid],
                event_obs[valid],
                event_w[valid],
                n_bins=n_bins,
            )
            if np.isfinite(ece_h):
                ece_terms.append(float(ece_h))

    per_sample_err = per_sample_err / max(1, horizons.size)
    sigma = np.clip(np.asarray(sigma_arr, dtype=np.float64), 1e-6, 1e3)
    ece = float(np.mean(ece_terms)) if ece_terms else float("nan")

    if np.std(sigma) < 1e-12 or np.std(per_sample_err) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(sigma, per_sample_err)[0, 1])

    n_top = max(1, int(round(0.2 * n)))
    order = np.argsort(-sigma)
    top_err = float(np.mean(per_sample_err[order[:n_top]]))
    bot_err = float(np.mean(per_sample_err[order[-n_top:]]))
    gap = top_err - bot_err
    return {"ece": ece, "unc_err_corr": corr, "uncertainty_error_gap": float(gap)}


def ause_from_uncertainty(
    mu_arr: np.ndarray,
    sigma_arr: np.ndarray,
    time_arr: np.ndarray,
    event_arr: np.ndarray,
    steps: int = 20,
) -> float:
    n = int(time_arr.size)
    if n < 15:
        return float("nan")

    g_t, g_s = _fit_reverse_km(time_arr, event_arr)
    horizons = _build_time_grid(time_arr, event_arr, num_grid=12)
    if horizons.size == 0:
        return float("nan")

    surv_mat = _lognormal_survival_np(mu_arr, sigma_arr, horizons)
    per_sample_err = np.zeros(n, dtype=np.float64)
    for j, h in enumerate(horizons):
        sqerr, _, _ = _ipcw_terms_at_horizon(time_arr, event_arr, surv_mat[:, j], float(h), g_t, g_s)
        per_sample_err += sqerr
    per_sample_err = per_sample_err / max(1, horizons.size)

    sigma = np.clip(np.asarray(sigma_arr, dtype=np.float64), 1e-6, 1e3)
    idx_unc = np.argsort(-sigma)
    idx_oracle = np.argsort(-per_sample_err)

    steps = max(5, int(steps))
    fracs = np.linspace(0.0, 0.9, steps)
    gaps = []
    all_idx = np.arange(n)
    for f in fracs:
        k = min(n - 1, int(round(f * n)))
        keep_unc = np.ones(n, dtype=bool)
        keep_oracle = np.ones(n, dtype=bool)
        if k > 0:
            keep_unc[idx_unc[:k]] = False
            keep_oracle[idx_oracle[:k]] = False
        err_unc = float(np.mean(per_sample_err[all_idx[keep_unc]]))
        err_oracle = float(np.mean(per_sample_err[all_idx[keep_oracle]]))
        gaps.append(max(0.0, err_unc - err_oracle))
    denom = float(fracs[-1] - fracs[0]) + 1e-8
    return float(np.trapz(np.asarray(gaps, dtype=np.float64), fracs) / denom)


def integrated_brier_score_ipcw(
    mu_arr: np.ndarray,
    sigma_arr: np.ndarray,
    time_arr: np.ndarray,
    event_arr: np.ndarray,
    num_grid: int = 20,
) -> float:
    if time_arr.size < 10:
        return float("nan")
    horizons = _build_time_grid(time_arr, event_arr, num_grid=max(8, int(num_grid)))
    if horizons.size == 0:
        return float("nan")

    g_t, g_s = _fit_reverse_km(time_arr, event_arr)
    surv_mat = _lognormal_survival_np(mu_arr, sigma_arr, horizons)

    bs_curve = []
    n = float(time_arr.size)
    for j, h in enumerate(horizons):
        sqerr, _, _ = _ipcw_terms_at_horizon(time_arr, event_arr, surv_mat[:, j], float(h), g_t, g_s)
        bs_curve.append(float(np.sum(sqerr) / n))
    if len(bs_curve) <= 1:
        return float(bs_curve[0]) if bs_curve else float("nan")
    hspan = float(horizons[-1] - horizons[0]) + 1e-8
    return float(np.trapz(np.asarray(bs_curve, dtype=np.float64), horizons) / hspan)


def fit_sigma_temperature(
    mu_arr: np.ndarray,
    sigma_arr: np.ndarray,
    time_arr: np.ndarray,
    event_arr: np.ndarray,
    grid_size: int = 31,
) -> Tuple[float, float]:
    if mu_arr.size == 0:
        return 1.0, float("nan")
    scales = np.exp(np.linspace(math.log(0.5), math.log(2.0), max(5, int(grid_size))))
    best_scale = 1.0
    best_nll = float("inf")
    for s in scales:
        cur_nll = _lognormal_nll_numpy(mu_arr, sigma_arr * float(s), time_arr, event_arr)
        if np.isfinite(cur_nll) and cur_nll < best_nll:
            best_nll = float(cur_nll)
            best_scale = float(s)
    return best_scale, best_nll


def update_wsi_strata_stats(
    stats: Dict[str, Dict[str, float]],
    patch_type: Optional[torch.Tensor],
    alpha_nodes: torch.Tensor,
    u_nodes: torch.Tensor,
    q_nodes: torch.Tensor,
    delta_sigma2_nodes: Optional[torch.Tensor] = None,
) -> None:
    if patch_type is None:
        return
    try:
        pt = torch.as_tensor(patch_type).view(-1).detach().cpu().numpy()
    except Exception:
        return
    alpha = alpha_nodes.detach().view(-1).cpu().numpy()
    u = u_nodes.detach().view(-1).cpu().numpy()
    q = q_nodes.detach().view(-1).cpu().numpy()
    if delta_sigma2_nodes is None:
        delta_sigma2 = np.zeros_like(alpha, dtype=np.float64)
    else:
        delta_sigma2 = delta_sigma2_nodes.detach().view(-1).cpu().numpy()
    if not (pt.size == alpha.size == u.size == q.size == delta_sigma2.size):
        return

    uniq = np.unique(pt)
    for t in uniq:
        mask = pt == t
        n = int(np.sum(mask))
        if n <= 0:
            continue
        try:
            key = str(int(t))
        except Exception:
            key = str(float(t))
        if key not in stats:
            stats[key] = {
                "count": 0,
                "alpha_sum": 0.0,
                "u_sum": 0.0,
                "q_sum": 0.0,
                "delta_sigma2_sum": 0.0,
            }
        stats[key]["count"] += n
        stats[key]["alpha_sum"] += float(np.sum(alpha[mask]))
        stats[key]["u_sum"] += float(np.sum(u[mask]))
        stats[key]["q_sum"] += float(np.sum(q[mask]))
        stats[key]["delta_sigma2_sum"] += float(np.sum(delta_sigma2[mask]))


def finalize_wsi_strata_stats(stats: Dict[str, Dict[str, float]], min_count: int = 20) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k, v in stats.items():
        c = int(v.get("count", 0))
        if c < int(min_count):
            continue
        out[k] = {
            "count": c,
            "alpha_mean": float(v["alpha_sum"] / max(1, c)),
            "u_mean": float(v["u_sum"] / max(1, c)),
            "q_mean": float(v["q_sum"] / max(1, c)),
            "delta_sigma2_mean": float(v["delta_sigma2_sum"] / max(1, c)),
        }
    return out


def attention_entropy_regularization(attn: torch.Tensor) -> torch.Tensor:
    """
    Penalize high-entropy (overly uniform) attention distributions.
    attn: [H, N_query, N_key], already softmax-normalized.
    """
    if attn.numel() == 0 or attn.size(-1) <= 1:
        return attn.new_tensor(0.0)
    p = torch.clamp(attn, min=1e-12)
    ent = -(p * torch.log(p)).sum(dim=-1)
    ent_norm = ent / math.log(attn.size(-1))
    return ent_norm.mean()


def pathway_query_diversity_loss(h_gene: torch.Tensor, batch_gene: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Reduce pathway embedding collapse by penalizing large off-diagonal cosine similarity.
    h_gene: [N_pathway, D]
    """
    if h_gene.numel() == 0 or h_gene.size(0) <= 1:
        return h_gene.new_tensor(0.0)

    if batch_gene is None:
        h = F.normalize(h_gene, p=2, dim=-1, eps=1e-6)
        sim = h @ h.t()
        n = sim.size(0)
        offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        offdiag = sim[offdiag_mask]
        if offdiag.numel() == 0:
            return sim.new_tensor(0.0)
        return (offdiag ** 2).mean()

    losses = []
    for bid in torch.unique(batch_gene, sorted=True):
        idx = (batch_gene == bid).nonzero(as_tuple=False).view(-1)
        if idx.numel() <= 1:
            continue
        h = F.normalize(h_gene[idx], p=2, dim=-1, eps=1e-6)
        sim = h @ h.t()
        n = sim.size(0)
        offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        offdiag = sim[offdiag_mask]
        if offdiag.numel() > 0:
            losses.append((offdiag ** 2).mean())
    if not losses:
        return h_gene.new_tensor(0.0)
    return torch.stack(losses).mean()
