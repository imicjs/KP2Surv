"""Command-line entry point for cross-validated KP2Surv training."""

from .common import *
from .data import *
from .model import KP2Surv
from .training import *

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clinical", required=True, help="Clinical table containing patient ID, survival time and vital status.")
    p.add_argument("--wsi_dir", required=True, help="Directory containing WSI hypergraph .pt files.")
    p.add_argument("--gene_dir", required=True, help="Directory containing patient-matched pathway structure .pt files.")
    p.add_argument("--expr", required=True, help="Case-by-gene expression CSV from scripts/make_expression_and_mask.py.")
    p.add_argument("--mask", required=True, help="Pathway-gene mask .npz from scripts/make_expression_and_mask.py.")
    p.add_argument("--out", required=True, help="Output directory for checkpoints and exported results.")
    p.add_argument("--prior_table", default="", help="Optional JSON override for the built-in weak-prior table.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--survival_time_strata_bins", type=int, default=4, help="Maximum within-event observed-time quantile bins for survival-stratified CV.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    p.add_argument("--batch", type=int, default=4) 
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--kl_weight", type=float, default=1e-3, help="Max KL weight for variational regularization.")
    p.add_argument("--kl_warmup_epochs", type=int, default=5, help="Warmup epochs to ramp KL weight.")
    p.add_argument("--cross_attn_temperature", type=float, default=0.90, help="Cross-attention softmax temperature (<1 sharper).")
    p.add_argument("--cross_num_heads", type=int, default=4, help="Number of heads for importance-guided cross-attention.")
    p.add_argument("--lambda_alpha_q", type=float, default=0.6, help="Query-side importance bias in cross-attention logits.")
    p.add_argument("--lambda_alpha_k", type=float, default=0.6, help="Key-side importance bias in cross-attention logits.")
    p.add_argument("--lambda_cross_q", type=float, default=0.8, help="QC penalty term in cross-attention logits.")
    p.add_argument("--lambda_conf", type=float, default=0.15, help="Conflict penalty weight in decision fusion.")
    p.add_argument("--lambda_qc", type=float, default=0.10, help="QC penalty weight for branch/fused variance.")
    p.add_argument("--ctr_top_frac", type=float, default=0.20, help="Top fraction used by counterfactual margin loss.")
    p.add_argument("--ctr_margin", type=float, default=0.10, help="Counterfactual margin value.")
    p.add_argument("--q_topk_frac", type=float, default=0.10, help="Top fraction used for case-level QC aggregation.")

    p.add_argument("--use_mixed_sampling", type=int, default=1, help="Enable mixed WSI sampling (1/0).")
    p.add_argument("--use_hyperedge_sampling", type=int, default=1, help="Use hyperedge-centric WSI sampling (1/0).")
    p.add_argument("--wsi_cache_dir", type=str, default="", help="Disk cache directory for sampled WSI sub-hypergraphs. Empty means <out>/wsi_subgraph_cache.")
    p.add_argument("--stable_wsi_cache_seed", type=int, default=1, help="Use stable per-WSI sampling seeds so subgraph caches are reusable across folds/workers.")
    p.add_argument("--rebuild_wsi_cache", type=int, default=0, help="Ignore existing WSI subgraph cache and rebuild it.")
    p.add_argument("--wsi_cache_variants", type=int, default=6, help="Number of cached WSI subgraph variants per slide. Training can sample among them; eval/visualization should use a fixed variant.")
    p.add_argument("--train_wsi_cache_variant_mode", type=str, default="random", choices=["fixed", "random"], help="How training chooses cached WSI subgraph variants.")
    p.add_argument("--eval_wsi_cache_variant_index", type=int, default=0, help="Fixed WSI cache variant used for validation/calibration/export.")
    p.add_argument("--coverage_ratio", type=float, default=0.50, help="Coverage component of mixed sampling.")
    p.add_argument("--prior_hot_ratio", type=float, default=0.25, help="Prior-guided component of mixed sampling.")
    p.add_argument("--hard_replay_ratio", type=float, default=0.25, help="Hard-example component of mixed sampling.")
    p.add_argument("--artifact_cap_ratio", type=float, default=0.15, help="Maximum artifact-heavy fraction in a sampled subset.")

    p.add_argument("--lambda_uni", type=float, default=0.30, help="Weight of branch survival auxiliary loss.")
    p.add_argument("--lambda_ctr", type=float, default=0.10, help="Weight of counterfactual contribution loss.")
    p.add_argument("--lambda_sparse", type=float, default=0.01, help="Weight of importance sparsity loss.")
    p.add_argument("--lambda_alpha_prior", type=float, default=0.01, help="Weight of alpha weak-prior loss.")
    p.add_argument("--lambda_q_prior", type=float, default=0.05, help="Weight of QC weak-prior loss (should be > alpha prior).")
    p.add_argument("--lambda_alpha_prior_stage1", type=float, default=0.002, help="Stage-1 alpha prior weight.")
    p.add_argument("--lambda_q_prior_stage1", type=float, default=0.05, help="Stage-1 QC prior weight.")
    p.add_argument("--lambda_align", type=float, default=0.05, help="Weight of cross-modal alignment loss.")
    p.add_argument("--lambda_align_stage1", type=float, default=0.0, help="Stage-1 alignment weight.")
    p.add_argument("--stage1_epochs", type=int, default=5, help="Stage-1 epochs without repeated stochastic passes or branch-disagreement fusion.")
    p.add_argument("--stage2_epochs", type=int, default=12, help="Stage-2 epochs with four stochastic passes and branch-disagreement fusion.")
    p.add_argument("--stochastic_passes_stage2", type=int, default=4, help="Repeated stochastic passes in stage 2.")
    p.add_argument("--stochastic_passes_stage3", type=int, default=8, help="Repeated stochastic passes in stage 3.")
    p.add_argument("--stochastic_input_dropout", type=float, default=0.20, help="Input dropout used during repeated stochastic passes.")
    p.add_argument("--ece_bins", type=int, default=10, help="Bin count for expected calibration error.")
    p.add_argument("--ause_steps", type=int, default=20, help="Steps for AUSE sparsification curve.")
    p.add_argument("--calib_ratio", type=float, default=0.15, help="Fraction of train fold reserved for stage-3 post-hoc calibration.")
    p.add_argument("--stage3_calibrate", action="store_true", help="Use train-calib split for post-hoc sigma temperature in stage-3 eval only.")

    p.add_argument("--l3_weight", type=float, default=0.0, help="L3 regularization weight; zero disables it.")
    p.add_argument("--attn_entropy_weight", type=float, default=0.015, help="Cross-attention entropy regularization weight.")
    p.add_argument("--pathway_div_weight", type=float, default=0.0075, help="Pathway diversity regularization weight.")
    p.add_argument("--early_stop_patience", type=int, default=10, help="Stop if no dual-metric improvement for this many epochs (<=0 disables).")
    p.add_argument("--early_stop_min_delta_c", type=float, default=1e-4, help="Minimum C-index improvement to count as better.")
    p.add_argument("--early_stop_min_delta_nll", type=float, default=1e-4, help="Minimum Val NLL decrease to break C-index ties.")
    p.add_argument("--plateau_patience", type=int, default=3, help="ReduceLROnPlateau patience.")
    p.add_argument("--plateau_factor", type=float, default=0.5, help="LR multiply factor when plateau is reached.")
    p.add_argument("--plateau_threshold", type=float, default=1e-4, help="Absolute threshold for plateau scheduler.")
    p.add_argument("--plateau_nll_weight", type=float, default=0.01, help="Plateau score = C-index - weight * Val NLL.")
    p.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for plateau scheduler.")
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader worker count.")
    p.add_argument("--pin_memory", type=int, default=1 if PIN_MEMORY else 0, help="Use pinned host memory for DataLoader.")
    p.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor (workers>0 only).")
    p.add_argument("--amp", type=int, default=1, help="Use CUDA AMP mixed precision when available (1/0).")
    p.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"], help="AMP dtype.")
    p.add_argument("--deterministic", type=int, default=0, help="Deterministic kernels (slower, 1/0).")
    p.add_argument("--tf32", type=int, default=1, help="Enable TF32 matmul/cudnn on Ampere+ GPUs (1/0).")
    p.add_argument("--compile", type=int, default=0, help="Enable torch.compile for model forward (1/0).")
    p.add_argument("--compile_mode", type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_last", action="store_true", help="Also save last-epoch checkpoint for each fold.")
    p.add_argument("--export_outputs", type=int, default=1, help="Export split tables, patient-level predictions and metrics CSV (1/0).")
    p.add_argument("--export_dir", type=str, default="exports", help="Relative folder under --out for exported CSV and JSON files.")
    p.add_argument("--export_splits", type=str, default="val", help="Comma-separated splits to export: train,calib,val or all.")
    p.add_argument("--km_threshold_rule", type=str, default="median", choices=["median", "quantile"], help="Risk threshold rule for KM table export.")
    p.add_argument("--km_quantile", type=float, default=0.5, help="Quantile used when km_threshold_rule=quantile.")
    p.add_argument("--export_only", type=int, default=0, help="Skip training and export from saved fold checkpoints only (1/0).")
    p.add_argument("--checkpoint_dir", type=str, default="", help="Directory containing fold{fold}_best.pt for export_only mode. Defaults to --out.")
    return p.parse_args()

def main():
    args = parse_args()
    args.use_mixed_sampling = bool(args.use_mixed_sampling)
    args.use_hyperedge_sampling = bool(args.use_hyperedge_sampling)
    args.stable_wsi_cache_seed = bool(args.stable_wsi_cache_seed)
    args.rebuild_wsi_cache = bool(args.rebuild_wsi_cache)
    args.wsi_cache_variants = max(1, int(args.wsi_cache_variants))
    if args.wsi_cache_variants <= 1:
        args.train_wsi_cache_variant_mode = "fixed"
    args.eval_wsi_cache_variant_index = max(
        0,
        min(int(args.eval_wsi_cache_variant_index), args.wsi_cache_variants - 1),
    )
    args.pin_memory = bool(args.pin_memory)
    args.amp = bool(args.amp)
    args.deterministic = bool(args.deterministic)
    args.tf32 = bool(args.tf32)
    args.compile = bool(args.compile)
    args.export_outputs = bool(args.export_outputs)
    args.export_only = bool(args.export_only)
    export_splits = parse_export_splits(args.export_splits)
    if args.lambda_q_prior <= args.lambda_alpha_prior:
        raise ValueError(
            f"Require lambda_q_prior > lambda_alpha_prior, got "
            f"{args.lambda_q_prior} <= {args.lambda_alpha_prior}"
        )
    if args.lambda_q_prior_stage1 <= args.lambda_alpha_prior_stage1:
        raise ValueError(
            f"Require lambda_q_prior_stage1 > lambda_alpha_prior_stage1, got "
            f"{args.lambda_q_prior_stage1} <= {args.lambda_alpha_prior_stage1}"
        )
    prior_table_path = args.prior_table if args.prior_table else None
    if prior_table_path and not os.path.isfile(prior_table_path):
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(prior_table_path))
        if os.path.isfile(alt_path):
            prior_table_path = alt_path
    prior_table_cfg = load_weak_prior_table(prior_table_path)
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    configure_runtime(device=device, deterministic=args.deterministic, tf32=args.tf32)

    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=True) if (use_amp and amp_dtype == torch.float16) else None
    ensure_dir(args.out)
    if not args.wsi_cache_dir:
        args.wsi_cache_dir = os.path.join(args.out, "wsi_subgraph_cache")
    if args.wsi_cache_dir:
        ensure_dir(args.wsi_cache_dir)
    export_dir = os.path.join(args.out, args.export_dir)
    if args.export_outputs:
        ensure_dir(export_dir)
        save_json(
            os.path.join(export_dir, "run_manifest.json"),
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "export_only" if args.export_only else "train_and_export",
                "seed": int(args.seed),
                "folds": int(args.folds),
                "export_splits": export_splits,
                "args": vars(args),
            },
        )
    
    clinical = ClinicalDataLoader.load(args.clinical)
    expr_df, expr_genes = load_expression_table(args.expr)
    mask_obj = load_pathway_gene_mask(args.mask, device)
    expr_df, gene_order = align_expression_to_mask(expr_df, mask_obj)
    
    full_ds = MultiModalSurvivalDataset(
        args.wsi_dir,
        args.gene_dir,
        clinical,
        expr_df,
        gene_order,
        seed=args.seed,
        use_mixed_sampling=args.use_mixed_sampling,
        use_hyperedge_sampling=args.use_hyperedge_sampling,
        wsi_cache_dir=args.wsi_cache_dir,
        stable_wsi_cache_seed=args.stable_wsi_cache_seed,
        rebuild_wsi_cache=args.rebuild_wsi_cache,
        wsi_cache_variants=args.wsi_cache_variants,
        wsi_cache_variant_mode="fixed",
        wsi_cache_variant_index=args.eval_wsi_cache_variant_index,
        coverage_ratio=args.coverage_ratio,
        prior_hot_ratio=args.prior_hot_ratio,
        hard_replay_ratio=args.hard_replay_ratio,
        artifact_cap_ratio=args.artifact_cap_ratio,
    )
    
    from sklearn.model_selection import StratifiedKFold
    survival_strata, strata_description = build_survival_strata(
        full_ds.samples,
        n_splits=args.folds,
        max_time_bins=args.survival_time_strata_bins,
    )
    print(f"  CV stratification: {strata_description}")
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    results = []
    fold_metric_records: List[Dict[str, float]] = []
    export_metric_records: List[Dict[str, Any]] = []
    
    split_features = np.zeros((len(full_ds.samples), 1), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(kf.split(split_features, survival_strata)):
        print(f"\n=== Fold {fold+1}/{args.folds} ===")
        train_ids_all, train_ids, calib_ids, val_ids = split_ids_for_fold(
            full_ds=full_ds,
            train_idx=train_idx,
            val_idx=val_idx,
            seed=args.seed,
            fold=fold,
            stage3_calibrate=bool(args.stage3_calibrate),
            calib_ratio=float(args.calib_ratio),
        )
        print(f"  Split sizes: train={len(train_ids)} calib={len(calib_ids)} val={len(val_ids)}")
        if args.export_outputs:
            export_fold_splits(export_dir, fold, train_ids, calib_ids, val_ids)
        
        train_ds = MultiModalSurvivalDataset(
            args.wsi_dir, args.gene_dir, clinical, expr_df, gene_order,
            ids=set(train_ids), seed=args.seed,
            use_mixed_sampling=args.use_mixed_sampling,
            use_hyperedge_sampling=args.use_hyperedge_sampling,
            wsi_cache_dir=args.wsi_cache_dir,
            stable_wsi_cache_seed=args.stable_wsi_cache_seed,
            rebuild_wsi_cache=args.rebuild_wsi_cache,
            wsi_cache_variants=args.wsi_cache_variants,
            wsi_cache_variant_mode=args.train_wsi_cache_variant_mode,
            wsi_cache_variant_index=args.eval_wsi_cache_variant_index,
            coverage_ratio=args.coverage_ratio,
            prior_hot_ratio=args.prior_hot_ratio,
            hard_replay_ratio=args.hard_replay_ratio,
            artifact_cap_ratio=args.artifact_cap_ratio,
        )
        val_ds = MultiModalSurvivalDataset(
            args.wsi_dir, args.gene_dir, clinical, expr_df, gene_order,
            ids=set(val_ids), seed=args.seed,
            use_mixed_sampling=args.use_mixed_sampling,
            use_hyperedge_sampling=args.use_hyperedge_sampling,
            wsi_cache_dir=args.wsi_cache_dir,
            stable_wsi_cache_seed=args.stable_wsi_cache_seed,
            rebuild_wsi_cache=args.rebuild_wsi_cache,
            wsi_cache_variants=args.wsi_cache_variants,
            wsi_cache_variant_mode="fixed",
            wsi_cache_variant_index=args.eval_wsi_cache_variant_index,
            coverage_ratio=args.coverage_ratio,
            prior_hot_ratio=args.prior_hot_ratio,
            hard_replay_ratio=args.hard_replay_ratio,
            artifact_cap_ratio=args.artifact_cap_ratio,
        )
        calib_ds = None
        if len(calib_ids) > 0:
            calib_ds = MultiModalSurvivalDataset(
                args.wsi_dir, args.gene_dir, clinical, expr_df, gene_order,
                ids=set(calib_ids), seed=args.seed,
                use_mixed_sampling=args.use_mixed_sampling,
                use_hyperedge_sampling=args.use_hyperedge_sampling,
                wsi_cache_dir=args.wsi_cache_dir,
                stable_wsi_cache_seed=args.stable_wsi_cache_seed,
                rebuild_wsi_cache=args.rebuild_wsi_cache,
                wsi_cache_variants=args.wsi_cache_variants,
                wsi_cache_variant_mode="fixed",
                wsi_cache_variant_index=args.eval_wsi_cache_variant_index,
                coverage_ratio=args.coverage_ratio,
                prior_hot_ratio=args.prior_hot_ratio,
                hard_replay_ratio=args.hard_replay_ratio,
                artifact_cap_ratio=args.artifact_cap_ratio,
            )
        
        persistent = bool(args.num_workers > 0)
        loader_common = {
            "collate_fn": survival_collate,
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": persistent,
        }
        if int(args.num_workers) > 0:
            loader_common["prefetch_factor"] = max(2, int(args.prefetch_factor))

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True,
            **loader_common,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            **loader_common,
        )
        calib_loader = None
        if calib_ds is not None and len(calib_ds) > 0:
            calib_loader = DataLoader(
                calib_ds,
                batch_size=args.batch,
                shuffle=False,
                **loader_common,
            )

        stats_ids = train_ids if len(train_ids) > 0 else train_ids_all
        mu_np, sd_np = compute_train_gene_stats(expr_df, stats_ids, gene_order)
        mu_g = torch.from_numpy(mu_np).to(device)
        sd_g = torch.from_numpy(sd_np).to(device)
        
        if len(train_ds) > 0:
            s0 = train_ds[0]
            wsi_dim = s0['wsi'].x.shape[1]
        else:
            wsi_dim = 1024
            
        gene_in_dim = mask_obj.G 
        
        model = KP2Surv(
            wsi_dim,
            gene_in_dim,
            hidden_dim=args.hidden_dim,
            num_pathways=mask_obj.P,
            dropout=args.dropout,
            cross_attn_temperature=args.cross_attn_temperature,
            cross_num_heads=args.cross_num_heads,
            lambda_conf=args.lambda_conf,
            lambda_qc=args.lambda_qc,
            lambda_alpha_q=args.lambda_alpha_q,
            lambda_alpha_k=args.lambda_alpha_k,
            lambda_cross_q=args.lambda_cross_q,
            ctr_top_frac=args.ctr_top_frac,
            ctr_margin=args.ctr_margin,
            q_topk_frac=args.q_topk_frac,
            stochastic_input_dropout=args.stochastic_input_dropout,
            prior_table=prior_table_cfg,
        ).to(device)
        if args.compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode=args.compile_mode, fullgraph=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=args.plateau_threshold,
            threshold_mode="abs",
            min_lr=args.min_lr,
        )
        
        best_c = -float("inf")
        best_val_nll = float("inf")
        best_ckpt_path = os.path.join(args.out, f"fold{fold}_best.pt")
        last_ckpt_path = os.path.join(args.out, f"fold{fold}_last.pt")

        best_epoch = -1
        no_improve_epochs = 0
        last_metrics = {"c_index": float("nan"), "nll": float("nan")}
        best_metrics: Optional[Dict[str, float]] = None
        last_epoch_idx = -1
        epoch_metric_records: List[Dict[str, Any]] = []

        if args.export_only:
            ckpt_root = args.checkpoint_dir if args.checkpoint_dir else args.out
            best_ckpt_path = os.path.join(ckpt_root, f"fold{fold}_best.pt")
            if not os.path.isfile(best_ckpt_path):
                print(f"  [WARN] export_only: checkpoint missing, skip fold {fold}: {best_ckpt_path}")
                continue

            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

            ckpt_mu = ckpt.get("train_gene_mean", None)
            ckpt_sd = ckpt.get("train_gene_std", None)
            if ckpt_mu is not None and ckpt_sd is not None:
                mu_eval = torch.from_numpy(np.asarray(ckpt_mu, dtype=np.float32)).to(device)
                sd_eval = torch.from_numpy(np.asarray(ckpt_sd, dtype=np.float32)).to(device)
            else:
                mu_eval = mu_g
                sd_eval = sd_g

            best_epoch = int(ckpt.get("epoch", 0))
            stage_id = int(ckpt.get("stage", resolve_training_stage(max(best_epoch - 1, 0), args.stage1_epochs, args.stage2_epochs)))
            sigma_temperature = float(
                ckpt.get(
                    "sigma_temperature",
                    ckpt.get("val_metrics", {}).get("sigma_temperature_fit", 1.0)
                    if isinstance(ckpt.get("val_metrics", {}), dict)
                    else 1.0,
                )
            )
            sigma_temperature = float(sigma_temperature if np.isfinite(sigma_temperature) else 1.0)

            split_loader_map = {"train": train_loader, "val": val_loader}
            if calib_loader is not None:
                split_loader_map["calib"] = calib_loader

            for split_name in export_splits:
                loader = split_loader_map.get(split_name)
                if loader is None:
                    continue
                rec = export_predictions_for_split(
                    model=model,
                    loader=loader,
                    split_name=split_name,
                    fold=fold,
                    seed=args.seed,
                    epoch=best_epoch,
                    stage_id=stage_id,
                    sigma_temperature=sigma_temperature,
                    export_dir=export_dir,
                    mu_g=mu_eval,
                    sd_g=sd_eval,
                    mask=mask_obj.mask,
                    device=device,
                    args=args,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                export_metric_records.append(rec)
                if split_name == "val":
                    results.append(rec["c_index"] if np.isfinite(rec["c_index"]) else 0.5)
                    fold_metric_records.append(
                        {
                            "fold": int(fold),
                            "best_epoch": int(best_epoch),
                            "stage": int(stage_id),
                            "c_index": float(rec["c_index"]),
                            "nll": float(rec["nll"]),
                            "ibs": float(rec["ibs"]),
                            "ece": float(rec["ece"]),
                            "ause": float(rec["ause"]),
                            "uncertainty_error_gap": float(rec["uncertainty_error_gap"]),
                        }
                    )
            print(f"  export_only done for fold {fold}: checkpoint={best_ckpt_path}")
            continue

        for ep in range(args.epochs):
            last_epoch_idx = ep
            stage_id = resolve_training_stage(ep, args.stage1_epochs, args.stage2_epochs)
            if stage_id == 1:
                ep_enable_stochastic_variance = False
                ep_enable_disagreement = False
                ep_stochastic_passes = 1
                ep_lambda_align = args.lambda_align_stage1
                ep_lambda_alpha_prior = args.lambda_alpha_prior_stage1
                ep_lambda_q_prior = args.lambda_q_prior_stage1
            elif stage_id == 2:
                ep_enable_stochastic_variance = True
                ep_enable_disagreement = True
                ep_stochastic_passes = max(1, int(args.stochastic_passes_stage2))
                ep_lambda_align = args.lambda_align
                ep_lambda_alpha_prior = args.lambda_alpha_prior
                ep_lambda_q_prior = args.lambda_q_prior
            else:
                ep_enable_stochastic_variance = True
                ep_enable_disagreement = True
                ep_stochastic_passes = max(1, int(args.stochastic_passes_stage3))
                ep_lambda_align = args.lambda_align
                ep_lambda_alpha_prior = args.lambda_alpha_prior
                ep_lambda_q_prior = args.lambda_q_prior

            kl_beta = current_kl_weight(args.kl_weight, args.kl_warmup_epochs, ep)
            train_stats = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                mu_g,
                sd_g,
                mask_obj.mask,
                device,
                l3_weight=args.l3_weight,
                attn_entropy_weight=args.attn_entropy_weight,
                pathway_div_weight=args.pathway_div_weight,
                kl_weight=kl_beta,
                lambda_uni=args.lambda_uni,
                lambda_ctr=args.lambda_ctr,
                lambda_sparse=args.lambda_sparse,
                lambda_alpha_prior=ep_lambda_alpha_prior,
                lambda_q_prior=ep_lambda_q_prior,
                lambda_align=ep_lambda_align,
                stochastic_passes=ep_stochastic_passes,
                enable_stochastic_variance=ep_enable_stochastic_variance,
                enable_disagreement=ep_enable_disagreement,
                stochastic_latent=True,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            sigma_temp = 1.0
            if stage_id >= 3 and args.stage3_calibrate and calib_loader is not None:
                sigma_temp = calibrate_sigma_temperature_from_loader(
                    model=model,
                    loader=calib_loader,
                    mu_g=mu_g,
                    sd_g=sd_g,
                    mask=mask_obj.mask,
                    device=device,
                    stochastic_passes=ep_stochastic_passes,
                    enable_stochastic_variance=ep_enable_stochastic_variance,
                    enable_disagreement=ep_enable_disagreement,
                )

            val_metrics_raw = evaluate(
                model,
                val_loader,
                mu_g,
                sd_g,
                mask_obj.mask,
                device,
                stochastic_passes=ep_stochastic_passes,
                enable_stochastic_variance=ep_enable_stochastic_variance,
                enable_disagreement=ep_enable_disagreement,
                ece_bins=args.ece_bins,
                ause_steps=args.ause_steps,
                sigma_temperature=1.0,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            if stage_id >= 3 and args.stage3_calibrate and sigma_temp != 1.0:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    mu_g,
                    sd_g,
                    mask_obj.mask,
                    device,
                    stochastic_passes=ep_stochastic_passes,
                    enable_stochastic_variance=ep_enable_stochastic_variance,
                    enable_disagreement=ep_enable_disagreement,
                    ece_bins=args.ece_bins,
                    ause_steps=args.ause_steps,
                    sigma_temperature=sigma_temp,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                val_metrics["nll_calibrated"] = float(val_metrics["nll"])
            else:
                val_metrics = dict(val_metrics_raw)
                val_metrics["nll_calibrated"] = float(val_metrics_raw["nll"])

            val_metrics["nll_raw"] = float(val_metrics_raw["nll"])
            val_metrics["sigma_temperature_fit"] = float(sigma_temp)

            val_nll = float(val_metrics_raw["nll"])
            c_idx = float(val_metrics_raw["c_index"])
            val_nll_for_select = val_nll
            if not np.isfinite(val_nll_for_select):
                val_nll_for_select = 1e6
            last_metrics = val_metrics

            sched_metric = plateau_score(c_idx, val_nll_for_select, args.plateau_nll_weight)
            scheduler.step(sched_metric)
            cur_lr = float(optimizer.param_groups[0]["lr"])

            improved = dual_selection_better(
                c_idx=c_idx,
                val_nll=val_nll_for_select,
                best_c=best_c,
                best_nll=best_val_nll,
                min_delta_c=args.early_stop_min_delta_c,
                min_delta_nll=args.early_stop_min_delta_nll,
            )

            if improved:
                best_c = c_idx if np.isfinite(c_idx) else best_c
                best_val_nll = float(val_nll_for_select)
                best_epoch = ep + 1
                no_improve_epochs = 0
                best_metrics = val_metrics
                torch.save({
                    "checkpoint_schema_version": 1,
                    "fold": fold,
                    "epoch": ep + 1,
                    "best_c_index": float(best_c),
                    "best_val_nll": float(best_val_nll),
                    "stage": int(stage_id),
                    "sigma_temperature": float(val_metrics["sigma_temperature_fit"]),
                    "val_metrics": val_metrics,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_gene_mean": mu_np,
                    "train_gene_std": sd_np,
                    "gene_order": gene_order,
                    "pathway_names": mask_obj.pathway_names if mask_obj.pathway_names else None,
                    "prior_table": prior_table_cfg,
                    "args": vars(args),
                }, best_ckpt_path)
            else:
                no_improve_epochs += 1

            epoch_metric_records.append(
                {
                    "fold": int(fold),
                    "epoch": int(ep + 1),
                    "stage": int(stage_id),
                    "seed": int(args.seed),
                    "lr": float(cur_lr),
                    "loss": float(train_stats["loss"]),
                    "nll_fuse": float(train_stats["nll_fuse"]),
                    "nll_branch": float(train_stats["nll_branch"]),
                    "ctr": float(train_stats["ctr"]),
                    "sparse": float(train_stats["sparse"]),
                    "alpha_prior": float(train_stats["alpha_prior"]),
                    "q_prior": float(train_stats["q_prior"]),
                    "align": float(train_stats["align"]),
                    "l3": float(train_stats["l3"]),
                    "attn_entropy": float(train_stats["attn_entropy"]),
                    "pathway_div": float(train_stats["pathway_div"]),
                    "kl": float(train_stats["kl"]),
                    "kl_weight": float(kl_beta),
                    "val_nll_raw": float(val_metrics["nll_raw"]),
                    "val_nll_calibrated": float(val_metrics["nll_calibrated"]),
                    "sigma_temperature_fit": float(val_metrics["sigma_temperature_fit"]),
                    "val_c_index": float(c_idx),
                    "val_ibs": float(val_metrics["ibs"]),
                    "val_ece": float(val_metrics["ece"]),
                    "val_ause": float(val_metrics["ause"]),
                    "val_uncertainty_error_gap": float(val_metrics["uncertainty_error_gap"]),
                    "best_c_index_so_far": float(best_c),
                    "best_val_nll_so_far": float(best_val_nll),
                    "no_improve_epochs": int(no_improve_epochs),
                }
            )
             
            print(
                f"  Ep {ep+1}: "
                f"[S{stage_id}] "
                f"Loss={train_stats['loss']:.4f} "
                f"(NLL_f={train_stats['nll_fuse']:.4f}, NLL_b={train_stats['nll_branch']:.4f}, "
                f"CTR={train_stats['ctr']:.4f}, Sparse={train_stats['sparse']:.4f}, "
                f"APrior={train_stats['alpha_prior']:.4f}, QPrior={train_stats['q_prior']:.4f}, Align={train_stats['align']:.4f}, "
                f"L3={train_stats['l3']:.4f}, "
                f"AttnEnt={train_stats['attn_entropy']:.4f}, PathDiv={train_stats['pathway_div']:.4f}, "
                f"KL={train_stats['kl']:.4f}, KLw={kl_beta:.6f}) | "
                f"Val NLL(raw/cal)={val_metrics['nll_raw']:.4f}/{val_metrics['nll_calibrated']:.4f} "
                f"T={val_metrics['sigma_temperature_fit']:.3f} C-Idx={c_idx:.4f} "
                f"IBS={val_metrics['ibs']:.4f} ECE={val_metrics['ece']:.4f} "
                f"AUSE={val_metrics['ause']:.4f} UGap={val_metrics['uncertainty_error_gap']:.4f} "
                f"Strata={len(val_metrics['wsi_strata_summary'])} "
                f"(Best: C={best_c:.4f}, NLL={best_val_nll:.4f}) | "
                f"LR={cur_lr:.2e} ES={no_improve_epochs}/{args.early_stop_patience}"
            )

            if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                print(
                    f"  Early stopping at epoch {ep+1}: no dual-metric improvement for "
                    f"{args.early_stop_patience} epochs."
                )
                break

        # If all c_idx are non-finite, still save a usable model file.
        if best_epoch < 0:
            best_c = float("nan")
            best_epoch = last_epoch_idx + 1 if args.epochs > 0 else 0
            best_metrics = last_metrics
            fallback_stage = resolve_training_stage(
                max(last_epoch_idx, 0), args.stage1_epochs, args.stage2_epochs
            )
            torch.save({
                "checkpoint_schema_version": 1,
                "fold": fold,
                "epoch": best_epoch,
                "best_c_index": float("nan"),
                "best_val_nll": float(best_val_nll) if np.isfinite(best_val_nll) else float("nan"),
                "stage": int(fallback_stage),
                "sigma_temperature": float(last_metrics.get("sigma_temperature_fit", 1.0)),
                "val_metrics": last_metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_gene_mean": mu_np,
                "train_gene_std": sd_np,
                "gene_order": gene_order,
                "pathway_names": mask_obj.pathway_names if mask_obj.pathway_names else None,
                "prior_table": prior_table_cfg,
                "args": vars(args),
            }, best_ckpt_path)

        if args.save_last:
            last_stage = resolve_training_stage(
                max(last_epoch_idx, 0), args.stage1_epochs, args.stage2_epochs
            )
            torch.save({
                "checkpoint_schema_version": 1,
                "fold": fold,
                "epoch": last_epoch_idx + 1 if args.epochs > 0 else 0,
                "last_c_index": float(last_metrics["c_index"]) if np.isfinite(last_metrics["c_index"]) else float("nan"),
                "last_val_nll": float(last_metrics.get("nll_raw", last_metrics.get("nll", float("nan")))),
                "stage": int(last_stage),
                "sigma_temperature": float(last_metrics.get("sigma_temperature_fit", 1.0)),
                "val_metrics": last_metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_gene_mean": mu_np,
                "train_gene_std": sd_np,
                "gene_order": gene_order,
                "pathway_names": mask_obj.pathway_names if mask_obj.pathway_names else None,
                "prior_table": prior_table_cfg,
                "args": vars(args),
            }, last_ckpt_path)

        print(f"  Saved best checkpoint: {best_ckpt_path} (epoch={best_epoch}, c-index={best_c})")
        if args.save_last:
            print(f"  Saved last checkpoint: {last_ckpt_path}")

        if args.export_outputs and epoch_metric_records:
            save_csv(
                pd.DataFrame(epoch_metric_records),
                os.path.join(export_dir, f"fold{fold}_metrics_epoch.csv"),
            )

        if args.export_outputs and os.path.isfile(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
            sigma_temperature = float(
                best_ckpt.get(
                    "sigma_temperature",
                    best_ckpt.get("val_metrics", {}).get("sigma_temperature_fit", 1.0)
                    if isinstance(best_ckpt.get("val_metrics", {}), dict)
                    else 1.0,
                )
            )
            sigma_temperature = float(sigma_temperature if np.isfinite(sigma_temperature) else 1.0)

            ckpt_mu = best_ckpt.get("train_gene_mean", None)
            ckpt_sd = best_ckpt.get("train_gene_std", None)
            if ckpt_mu is not None and ckpt_sd is not None:
                mu_eval = torch.from_numpy(np.asarray(ckpt_mu, dtype=np.float32)).to(device)
                sd_eval = torch.from_numpy(np.asarray(ckpt_sd, dtype=np.float32)).to(device)
            else:
                mu_eval = mu_g
                sd_eval = sd_g

            stage_for_export = int(
                best_ckpt.get(
                    "stage",
                    resolve_training_stage(max(int(best_epoch) - 1, 0), args.stage1_epochs, args.stage2_epochs),
                )
            )
            split_loader_map = {"train": train_loader, "val": val_loader}
            if calib_loader is not None:
                split_loader_map["calib"] = calib_loader

            for split_name in export_splits:
                loader = split_loader_map.get(split_name)
                if loader is None:
                    continue
                rec = export_predictions_for_split(
                    model=model,
                    loader=loader,
                    split_name=split_name,
                    fold=fold,
                    seed=args.seed,
                    epoch=int(best_epoch),
                    stage_id=stage_for_export,
                    sigma_temperature=sigma_temperature,
                    export_dir=export_dir,
                    mu_g=mu_eval,
                    sd_g=sd_eval,
                    mask=mask_obj.mask,
                    device=device,
                    args=args,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                export_metric_records.append(rec)

        results.append(best_c if np.isfinite(best_c) else 0.5)
        if best_metrics is not None:
            fold_metric_records.append({
                "fold": int(fold),
                "best_epoch": int(best_epoch),
                "stage": int(resolve_training_stage(max(int(best_epoch) - 1, 0), args.stage1_epochs, args.stage2_epochs)),
                "c_index": float(best_metrics.get("c_index", float("nan"))),
                "nll": float(best_metrics.get("nll_raw", best_metrics.get("nll", float("nan")))),
                "ibs": float(best_metrics.get("ibs", float("nan"))),
                "ece": float(best_metrics.get("ece", float("nan"))),
                "ause": float(best_metrics.get("ause", float("nan"))),
                "uncertainty_error_gap": float(best_metrics.get("uncertainty_error_gap", float("nan"))),
            })
         
    mean_c, std_c = sample_mean_and_std(results)
    print(f"\nAverage C-Index: {mean_c:.4f} +/- {std_c:.4f}")
    if args.export_outputs and fold_metric_records:
        save_csv(pd.DataFrame(fold_metric_records), os.path.join(export_dir, "metrics_fold.csv"))
    if args.export_outputs and export_metric_records:
        save_csv(pd.DataFrame(export_metric_records), os.path.join(export_dir, "metrics_export.csv"))

    if fold_metric_records:
        def _mean_std(name: str) -> Tuple[float, float]:
            arr = np.asarray([r[name] for r in fold_metric_records], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return float("nan"), float("nan")
            return sample_mean_and_std(arr)

        nll_m, nll_s = _mean_std("nll")
        ibs_m, ibs_s = _mean_std("ibs")
        ece_m, ece_s = _mean_std("ece")
        ause_m, ause_s = _mean_std("ause")
        gap_m, gap_s = _mean_std("uncertainty_error_gap")
        print(
            "Avg Uncertainty Metrics: "
            f"NLL={nll_m:.4f}+/-{nll_s:.4f}, "
            f"IBS={ibs_m:.4f}+/-{ibs_s:.4f}, "
            f"ECE={ece_m:.4f}+/-{ece_s:.4f}, "
            f"AUSE={ause_m:.4f}+/-{ause_s:.4f}, "
            f"UGap={gap_m:.4f}+/-{gap_s:.4f}"
        )
        if args.export_outputs:
            save_json(
                os.path.join(export_dir, "metrics_cv_summary.json"),
                {
                    "c_index_mean": mean_c,
                    "c_index_sample_std": std_c,
                    "nll_mean": nll_m,
                    "nll_sample_std": nll_s,
                    "ibs_mean": ibs_m,
                    "ibs_sample_std": ibs_s,
                    "ece_mean": ece_m,
                    "ece_sample_std": ece_s,
                    "ause_mean": ause_m,
                    "ause_sample_std": ause_s,
                    "uncertainty_error_gap_mean": gap_m,
                    "uncertainty_error_gap_sample_std": gap_s,
                    "standard_deviation_ddof": 1,
                    "num_folds": int(len(fold_metric_records)),
                },
            )

if __name__ == "__main__":
    main()
