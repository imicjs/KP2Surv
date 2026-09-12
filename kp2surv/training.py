"""Training, evaluation, calibration, checkpoint-selection, and export helpers."""

from __future__ import annotations

from .common import *
from .metrics import *
from .metrics import _raise_if_non_finite
def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    mu_g,
    sd_g,
    mask,
    device,
    l3_weight: float = 0.0,
    attn_entropy_weight: float = 0.0,
    pathway_div_weight: float = 0.0,
    kl_weight: float = 0.0,
    lambda_uni: float = 0.3,
    lambda_ctr: float = 0.1,
    lambda_sparse: float = 0.01,
    lambda_alpha_prior: float = 0.01,
    lambda_q_prior: float = 0.05,
    lambda_align: float = 0.05,
    stochastic_passes: int = 1,
    enable_stochastic_variance: bool = False,
    enable_disagreement: bool = True,
    stochastic_latent: bool = True,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
):
    model.train()
    total_loss = 0.0
    total_nll_fuse = 0.0
    total_nll_branch = 0.0
    total_l3 = 0.0
    total_attn = 0.0
    total_div = 0.0
    total_kl = 0.0
    total_ctr = 0.0
    total_sparse = 0.0
    total_alpha_prior = 0.0
    total_q_prior = 0.0
    total_align = 0.0
    total_samples = 0
    amp_on = bool(use_amp and device.type == "cuda")
    
    for batch in tqdm(loader, desc="Train", leave=False):
        wsis = batch['wsi'].to(device, non_blocking=True)
        gene_tpls = batch['gene_tpl'].to(device, non_blocking=True)
        exprs = batch['expr'].to(device, non_blocking=True)
        times = batch['time'].to(device, non_blocking=True)
        events = batch['event'].to(device, non_blocking=True)
        case_ids = batch['case_id']
        
        optimizer.zero_grad(set_to_none=True)
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on else nullcontext()
        with amp_ctx:
            expr_std = (exprs - mu_g.unsqueeze(0)) / sd_g.unsqueeze(0)

            out = model.forward_batch(
                wsis,
                gene_tpls,
                expr_std,
                mask,
                need_attn_stats=(attn_entropy_weight > 0.0),
                stochastic_passes=stochastic_passes,
                enable_stochastic_variance=enable_stochastic_variance,
                enable_disagreement=enable_disagreement,
                stochastic_latent=stochastic_latent,
                compute_ctr_loss=(lambda_ctr > 0.0),
                compute_aux_losses=True,
            )

            batch_ctx = f"train cases={case_ids[0]}..{case_ids[-1]}" if len(case_ids) > 1 else f"train case={case_ids[0]}"
            _raise_if_non_finite("mu", out['mu'], context=batch_ctx)
            _raise_if_non_finite("sigma", out['sigma'], context=batch_ctx)
            nll = lognormal_nll_loss(out['mu'], out['sigma'], times, events)
            nll_wsi = lognormal_nll_loss(out['mu_wsi'], out['sigma_wsi'], times, events)
            nll_gene = lognormal_nll_loss(out['mu_gene'], out['sigma_gene'], times, events)
            nll_cross = lognormal_nll_loss(out['mu_cross'], out['sigma_cross'], times, events)
            branch_nll = (nll_wsi + nll_gene + nll_cross) / 3.0

            l3_reg = out['l3_loss']
            attn_reg = out.get("attn_entropy_loss", nll.new_tensor(0.0))
            div_reg = out.get("pathway_div_loss", nll.new_tensor(0.0))
            kl_reg = out.get("kl_loss", nll.new_tensor(0.0))
            ctr_reg = out.get("ctr_loss", nll.new_tensor(0.0))
            sparse_reg = out.get("sparse_loss", nll.new_tensor(0.0))
            alpha_prior_reg = out.get("alpha_prior_loss", nll.new_tensor(0.0))
            q_prior_reg = out.get("q_prior_loss", nll.new_tensor(0.0))
            align_reg = out.get("align_loss", nll.new_tensor(0.0))

            loss_batch = (
                nll
                + lambda_uni * branch_nll
                + lambda_ctr * ctr_reg
                + lambda_sparse * sparse_reg
                + lambda_alpha_prior * alpha_prior_reg
                + lambda_q_prior * q_prior_reg
                + lambda_align * align_reg
                + l3_weight * l3_reg
                + attn_entropy_weight * attn_reg
                + pathway_div_weight * div_reg
                + kl_weight * kl_reg
            )

        _raise_if_non_finite("loss_batch", loss_batch, context="train batch")
        if amp_on and scaler is not None:
            scaler.scale(loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_batch.backward()
            optimizer.step()
        bs = int(times.size(0))
        total_loss += float(loss_batch.detach().item()) * bs
        total_nll_fuse += float(nll.detach().item()) * bs
        total_nll_branch += float(branch_nll.detach().item()) * bs
        total_l3 += float(l3_reg.detach().item()) * bs
        total_attn += float(attn_reg.detach().item()) * bs
        total_div += float(div_reg.detach().item()) * bs
        total_kl += float(kl_reg.detach().item()) * bs
        total_ctr += float(ctr_reg.detach().item()) * bs
        total_sparse += float(sparse_reg.detach().item()) * bs
        total_alpha_prior += float(alpha_prior_reg.detach().item()) * bs
        total_q_prior += float(q_prior_reg.detach().item()) * bs
        total_align += float(align_reg.detach().item()) * bs
        total_samples += bs
        
    if total_samples <= 0:
        return {
            "loss": 0.0,
            "nll_fuse": 0.0,
            "nll_branch": 0.0,
            "l3": 0.0,
            "attn_entropy": 0.0,
            "pathway_div": 0.0,
            "kl": 0.0,
            "ctr": 0.0,
            "sparse": 0.0,
            "alpha_prior": 0.0,
            "q_prior": 0.0,
            "align": 0.0,
        }
    return {
        "loss": total_loss / total_samples,
        "nll_fuse": total_nll_fuse / total_samples,
        "nll_branch": total_nll_branch / total_samples,
        "l3": total_l3 / total_samples,
        "attn_entropy": total_attn / total_samples,
        "pathway_div": total_div / total_samples,
        "kl": total_kl / total_samples,
        "ctr": total_ctr / total_samples,
        "sparse": total_sparse / total_samples,
        "alpha_prior": total_alpha_prior / total_samples,
        "q_prior": total_q_prior / total_samples,
        "align": total_align / total_samples,
    }

@torch.no_grad()
def evaluate(
    model,
    loader,
    mu_g,
    sd_g,
    mask,
    device,
    stochastic_passes: int = 1,
    enable_stochastic_variance: bool = False,
    enable_disagreement: bool = True,
    ece_bins: int = 10,
    ause_steps: int = 20,
    sigma_temperature: float = 1.0,
    return_raw: bool = False,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
):
    model.eval()
    risks, times, events = [], [], []
    mus, sigmas = [], []
    case_ids_all: List[str] = []
    wsi_strata_stats: Dict[str, Dict[str, float]] = {}
    val_loss = 0.0
    count = 0
    amp_on = bool(use_amp and device.type == "cuda")
    
    for batch in tqdm(loader, desc="Eval", leave=False):
        wsis = batch['wsi'].to(device, non_blocking=True)
        gene_tpls = batch['gene_tpl'].to(device, non_blocking=True)
        exprs = batch['expr'].to(device, non_blocking=True)
        t = batch['time'].to(device, non_blocking=True)
        e = batch['event'].to(device, non_blocking=True)
        case_ids = batch['case_id']

        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on else nullcontext()
        with amp_ctx:
            expr_std = (exprs - mu_g.unsqueeze(0)) / sd_g.unsqueeze(0)
            out = model.forward_batch(
                wsis,
                gene_tpls,
                expr_std,
                mask,
                stochastic_passes=stochastic_passes,
                enable_stochastic_variance=enable_stochastic_variance,
                enable_disagreement=enable_disagreement,
                stochastic_latent=False,
                compute_ctr_loss=False,
                compute_aux_losses=False,
            )

        batch_ctx = f"eval cases={case_ids[0]}..{case_ids[-1]}" if len(case_ids) > 1 else f"eval case={case_ids[0]}"
        _raise_if_non_finite("mu", out['mu'], context=batch_ctx)
        _raise_if_non_finite("sigma", out['sigma'], context=batch_ctx)
        sigma_scaled = torch.clamp(out['sigma'] * float(sigma_temperature), min=1e-6, max=1e3)
        nll = lognormal_nll_loss(out['mu'], sigma_scaled, t, e)
        _raise_if_non_finite("nll", nll, context=batch_ctx)
        _raise_if_non_finite("risk_score", out['risk_score'], context=batch_ctx)

        bs = int(t.size(0))
        val_loss += float(nll.item()) * bs
        count += bs

        risks.extend(out['risk_score'].detach().view(-1).cpu().tolist())
        times.extend(t.detach().view(-1).cpu().tolist())
        events.extend(e.detach().view(-1).cpu().tolist())
        mus.extend(out['mu'].detach().view(-1).cpu().tolist())
        sigmas.extend(sigma_scaled.detach().view(-1).cpu().tolist())
        case_ids_all.extend(list(case_ids))
        update_wsi_strata_stats(
            wsi_strata_stats,
            getattr(wsis, "patch_classify_type", None),
            out["alpha_wsi_nodes"],
            out["u_wsi_nodes"],
            out["q_wsi_nodes"],
            out.get("delta_sigma2_wsi_nodes", None),
        )
             
    avg_loss = val_loss / max(1, count)
    time_arr = np.asarray(times, dtype=np.float64)
    event_arr = np.asarray(events, dtype=np.float64)
    risk_arr = np.asarray(risks, dtype=np.float64)
    mu_arr = np.asarray(mus, dtype=np.float64)
    sigma_arr = np.asarray(sigmas, dtype=np.float64)

    c_index = concordance_index_fallback(time_arr, risk_arr, event_arr)
    ibs = integrated_brier_score_ipcw(mu_arr, sigma_arr, time_arr, event_arr, num_grid=20)
    ause = ause_from_uncertainty(mu_arr, sigma_arr, time_arr, event_arr, steps=ause_steps)
    cal_metrics = uncertainty_calibration_metrics(
        mu_arr, sigma_arr, time_arr, event_arr, n_bins=ece_bins
    )

    strata_summary = finalize_wsi_strata_stats(wsi_strata_stats, min_count=20)

    out_metrics = {
        "nll": float(avg_loss),
        "sigma_temperature": float(sigma_temperature),
        "c_index": float(c_index),
        "ibs": float(ibs),
        "ause": float(ause),
        "ece": float(cal_metrics["ece"]),
        "unc_err_corr": float(cal_metrics["unc_err_corr"]),
        "uncertainty_error_gap": float(cal_metrics["uncertainty_error_gap"]),
        "wsi_strata_summary": strata_summary,
    }
    if return_raw:
        out_metrics.update({
            "raw_mu": mu_arr,
            "raw_sigma": sigma_arr,
            "raw_time": time_arr,
            "raw_event": event_arr,
            "raw_risk": risk_arr,
            "raw_case_id": case_ids_all,
        })
    return out_metrics


@torch.no_grad()
def calibrate_sigma_temperature_from_loader(
    model,
    loader,
    mu_g,
    sd_g,
    mask,
    device,
    stochastic_passes: int = 1,
    enable_stochastic_variance: bool = False,
    enable_disagreement: bool = True,
) -> float:
    if loader is None:
        return 1.0
    metrics = evaluate(
        model=model,
        loader=loader,
        mu_g=mu_g,
        sd_g=sd_g,
        mask=mask,
        device=device,
        stochastic_passes=stochastic_passes,
        enable_stochastic_variance=enable_stochastic_variance,
        enable_disagreement=enable_disagreement,
        sigma_temperature=1.0,
        return_raw=True,
    )
    mu_arr = metrics.get("raw_mu", np.asarray([], dtype=np.float64))
    sigma_arr = metrics.get("raw_sigma", np.asarray([], dtype=np.float64))
    time_arr = metrics.get("raw_time", np.asarray([], dtype=np.float64))
    event_arr = metrics.get("raw_event", np.asarray([], dtype=np.float64))
    if int(np.asarray(mu_arr).size) < 10:
        return 1.0
    sigma_temp, _ = fit_sigma_temperature(mu_arr, sigma_arr, time_arr, event_arr)
    if not np.isfinite(sigma_temp):
        return 1.0
    return float(np.clip(sigma_temp, 0.5, 2.0))


def current_kl_weight(base_weight: float, warmup_epochs: int, epoch_idx: int) -> float:
    if base_weight <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return base_weight
    progress = min(1.0, float(epoch_idx + 1) / float(warmup_epochs))
    return base_weight * progress


def _safe_cindex(c_idx: float) -> float:
    return float(c_idx) if np.isfinite(c_idx) else float("-inf")


def dual_selection_better(
    c_idx: float,
    val_nll: float,
    best_c: float,
    best_nll: float,
    min_delta_c: float = 0.0,
    min_delta_nll: float = 0.0
) -> bool:
    c_cur = _safe_cindex(c_idx)
    c_best = _safe_cindex(best_c)

    if c_cur > (c_best + min_delta_c):
        return True
    if abs(c_cur - c_best) <= min_delta_c and val_nll < (best_nll - min_delta_nll):
        return True
    return False


def plateau_score(c_idx: float, val_nll: float, nll_weight: float) -> float:
    c_safe = float(c_idx) if np.isfinite(c_idx) else 0.0
    return c_safe - nll_weight * float(val_nll)


def stage_runtime_flags(stage_id: int, args) -> Tuple[bool, bool, int, float]:
    if int(stage_id) <= 1:
        return False, False, 1, 1.0
    if int(stage_id) == 2:
        return True, True, max(1, int(args.stochastic_passes_stage2)), 1.0
    return True, True, max(1, int(args.stochastic_passes_stage3)), 1.0


@torch.no_grad()
def export_predictions_for_split(
    model,
    loader,
    split_name: str,
    fold: int,
    seed: int,
    epoch: int,
    stage_id: int,
    sigma_temperature: float,
    export_dir: str,
    mu_g,
    sd_g,
    mask,
    device,
    args,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    enable_stochastic_variance, enable_disagreement, stochastic_passes, _ = stage_runtime_flags(
        stage_id, args
    )
    raw_metrics = evaluate(
        model=model,
        loader=loader,
        mu_g=mu_g,
        sd_g=sd_g,
        mask=mask,
        device=device,
        stochastic_passes=stochastic_passes,
        enable_stochastic_variance=enable_stochastic_variance,
        enable_disagreement=enable_disagreement,
        ece_bins=args.ece_bins,
        ause_steps=args.ause_steps,
        sigma_temperature=float(sigma_temperature),
        return_raw=True,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    pred_df = prediction_frame_from_raw(
        raw_metrics=raw_metrics,
        fold=fold,
        seed=seed,
        split=split_name,
        sigma_temperature=float(sigma_temperature),
        stage=stage_id,
        epoch=epoch,
    )
    pred_path = os.path.join(export_dir, f"fold{fold}_{split_name}_pred.csv")
    save_csv(pred_df, pred_path)

    km_df, km_thr = km_frame_from_predictions(
        pred_df,
        threshold_rule=args.km_threshold_rule,
        quantile=args.km_quantile,
    )
    km_path = os.path.join(export_dir, f"fold{fold}_{split_name}_km.csv")
    save_csv(km_df, km_path)

    out = {
        "fold": int(fold),
        "split": split_name,
        "num_samples": int(pred_df.shape[0]),
        "c_index": float(raw_metrics.get("c_index", float("nan"))),
        "nll": float(raw_metrics.get("nll", float("nan"))),
        "ibs": float(raw_metrics.get("ibs", float("nan"))),
        "ece": float(raw_metrics.get("ece", float("nan"))),
        "ause": float(raw_metrics.get("ause", float("nan"))),
        "uncertainty_error_gap": float(raw_metrics.get("uncertainty_error_gap", float("nan"))),
        "sigma_temperature": float(sigma_temperature),
        "km_threshold": float(km_thr) if np.isfinite(km_thr) else float("nan"),
        "pred_path": pred_path,
        "km_path": km_path,
    }
    return out


def resolve_training_stage(epoch_idx: int, stage1_epochs: int, stage2_epochs: int) -> int:
    s1 = max(0, int(stage1_epochs))
    s2 = max(0, int(stage2_epochs))
    if epoch_idx < s1:
        return 1
    if epoch_idx < (s1 + s2):
        return 2
    return 3
