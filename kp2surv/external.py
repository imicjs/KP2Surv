"""External-cohort inference with fold-specific TCGA checkpoints."""

from __future__ import annotations

import argparse
import os
import re
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import (
    HypergraphData,
    configure_runtime,
    ensure_dir,
    extract_case_id,
    infer_num_hyperedges,
    prediction_frame_from_raw,
    sample_mean_and_std,
    save_csv,
    save_json,
    set_global_seed,
    load_weak_prior_table,
)
from .data import (
    ClinicalDataLoader,
    MultiModalSurvivalDataset,
    PathwayGeneMask,
    load_pathway_gene_mask,
    survival_collate,
)
from .model import KP2Surv
from .training import evaluate, resolve_training_stage


def load_external_expression(
    expression_path: str,
    gene_mapping_path: str = "",
) -> pd.DataFrame:
    """Load a case-by-gene expression table and apply an optional gene map."""
    expression = pd.read_csv(expression_path, index_col=0)
    expression.index = expression.index.map(extract_case_id)
    expression.columns = expression.columns.astype(str)
    expression = expression.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if expression.columns.duplicated().any():
        expression = expression.T.groupby(level=0, sort=True).mean().T
    if expression.index.duplicated().any():
        expression = expression.groupby(level=0, sort=True).mean()

    if gene_mapping_path:
        mapping = pd.read_csv(gene_mapping_path)
        required = {"source_gene", "target_gene"}
        if not required.issubset(mapping.columns):
            raise ValueError(
                "Gene mapping CSV must contain source_gene and target_gene columns"
            )
        rename_map = {
            str(source): str(target)
            for source, target in mapping[["source_gene", "target_gene"]].itertuples(
                index=False, name=None
            )
            if pd.notna(source) and pd.notna(target)
        }
        expression = expression.rename(columns=rename_map)
        if expression.columns.duplicated().any():
            expression = expression.T.groupby(level=0, sort=True).mean().T

    return expression.sort_index()


def align_external_expression(
    expression: pd.DataFrame,
    gene_order: Sequence[str],
    max_missing_fraction: float = 0.10,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Align external expression to a checkpoint gene space."""
    genes = [str(g) for g in gene_order]
    if not genes:
        raise ValueError("Checkpoint gene_order is empty")
    if len(set(genes)) != len(genes):
        raise ValueError("Checkpoint gene_order contains duplicates")

    missing = [gene for gene in genes if gene not in expression.columns]
    missing_fraction = len(missing) / len(genes)
    threshold = float(max_missing_fraction)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("max_missing_fraction must be between 0 and 1")
    if missing_fraction > threshold:
        raise ValueError(
            f"Missing-gene fraction {missing_fraction:.4f} exceeds threshold "
            f"{threshold:.4f} ({len(missing)}/{len(genes)})"
        )

    aligned = expression.reindex(columns=genes, fill_value=0.0)
    aligned = aligned.astype(np.float32, copy=False)
    report = {
        "checkpoint_gene_count": len(genes),
        "missing_gene_count": len(missing),
        "missing_gene_fraction": float(missing_fraction),
        "max_missing_gene_fraction": threshold,
        "missing_genes": missing,
    }
    return aligned, report


def align_mask_to_checkpoint(
    mask_obj: PathwayGeneMask,
    gene_order: Sequence[str],
    checkpoint_pathways: Sequence[str] | None,
    device: torch.device,
) -> torch.Tensor:
    """Align mask rows and columns to checkpoint pathway and gene order."""
    if not mask_obj.gene_names:
        raise ValueError("External evaluation requires gene_names in the pathway mask")

    source_genes = [str(g) for g in mask_obj.gene_names]
    source_gene_index = {gene: idx for idx, gene in enumerate(source_genes)}

    if checkpoint_pathways:
        if not mask_obj.pathway_names:
            raise ValueError(
                "Checkpoint contains pathway_names but the supplied mask does not"
            )
        source_pathway_index = {
            str(pathway): idx for idx, pathway in enumerate(mask_obj.pathway_names)
        }
        missing_pathways = [
            str(pathway)
            for pathway in checkpoint_pathways
            if str(pathway) not in source_pathway_index
        ]
        if missing_pathways:
            raise ValueError(
                f"Mask is missing {len(missing_pathways)} checkpoint pathways"
            )
        row_indices = [source_pathway_index[str(p)] for p in checkpoint_pathways]
    else:
        row_indices = list(range(mask_obj.mask.size(0)))

    aligned = torch.zeros(
        (len(row_indices), len(gene_order)),
        dtype=mask_obj.mask.dtype,
        device=device,
    )
    source_mask = mask_obj.mask.to(device)
    for target_col, gene in enumerate(gene_order):
        source_col = source_gene_index.get(str(gene))
        if source_col is not None:
            aligned[:, target_col] = source_mask[row_indices, source_col]
    return aligned


def build_external_pathway_structures(
    pathway_template: str,
    wsi_dir: str,
    output_dir: str,
    eligible_case_ids: Sequence[str],
) -> str:
    """Create structure-only pathway files from the TCGA pathway template."""
    template = torch.load(pathway_template, map_location="cpu", weights_only=False)
    if not hasattr(template, "hyperedge_index"):
        raise ValueError("Pathway template does not contain hyperedge_index")
    num_nodes = int(template.num_nodes)
    num_hyperedges = int(
        getattr(template, "num_hyperedges", infer_num_hyperedges(template.hyperedge_index))
    )
    eligible = set(eligible_case_ids)
    ensure_dir(output_dir)

    created = 0
    for filename in sorted(os.listdir(wsi_dir)):
        if not filename.endswith(".pt"):
            continue
        case_id = extract_case_id(os.path.splitext(filename)[0])
        if case_id not in eligible:
            continue
        structure = HypergraphData(
            hyperedge_index=template.hyperedge_index.clone(),
            num_nodes=num_nodes,
            num_hyperedges=num_hyperedges,
        )
        structure.pathway_names = list(getattr(template, "pathway_names", []))
        structure.features_built_online = True
        torch.save(structure, os.path.join(output_dir, filename))
        created += 1

    if created == 0:
        raise RuntimeError("No external pathway structures were created")
    return output_dir


def discover_checkpoints(
    checkpoint_dir: str,
    pattern: str = "fold*_best.pt",
    expected_count: int = 5,
) -> List[str]:
    """Find and numerically order fold checkpoints."""
    candidates = [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if _matches_fold_pattern(name, pattern)
    ]

    def fold_number(path: str) -> int:
        match = re.search(r"fold(\d+)", os.path.basename(path), flags=re.IGNORECASE)
        return int(match.group(1)) if match else 10**9

    checkpoints = sorted(candidates, key=lambda path: (fold_number(path), path))
    if expected_count > 0 and len(checkpoints) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} checkpoints matching {pattern!r}, "
            f"found {len(checkpoints)}"
        )
    if not checkpoints:
        raise RuntimeError(f"No checkpoints matching {pattern!r} in {checkpoint_dir}")
    return checkpoints


def _matches_fold_pattern(filename: str, pattern: str) -> bool:
    escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.fullmatch(escaped, filename) is not None


def _checkpoint_arg(
    checkpoint_args: Dict[str, object],
    name: str,
    default,
    old_name: str = "",
):
    if name in checkpoint_args:
        return checkpoint_args[name]
    if old_name and old_name in checkpoint_args:
        return checkpoint_args[old_name]
    return default


def normalized_checkpoint_args(checkpoint: Dict[str, object]) -> SimpleNamespace:
    """Normalize inference settings from current and earlier checkpoints."""
    saved_args = checkpoint.get("args", {}) or {}
    if isinstance(saved_args, dict):
        source = dict(saved_args)
    elif hasattr(saved_args, "__dict__"):
        source = dict(vars(saved_args))
    else:
        raise TypeError("Checkpoint args must be a mapping or argparse namespace")
    normalized = dict(source)
    normalized["stochastic_passes_stage2"] = int(
        _checkpoint_arg(source, "stochastic_passes_stage2", 4, "mc_dropout_samples_stage2")
    )
    normalized["stochastic_passes_stage3"] = int(
        _checkpoint_arg(source, "stochastic_passes_stage3", 8, "mc_dropout_samples_stage3")
    )
    normalized["stochastic_input_dropout"] = float(
        _checkpoint_arg(source, "stochastic_input_dropout", 0.20, "mc_input_dropout")
    )
    normalized["use_mixed_sampling"] = bool(
        _checkpoint_arg(source, "use_mixed_sampling", True, "use_v2_sampling")
    )
    defaults = {
        "stage1_epochs": 5,
        "stage2_epochs": 12,
        "hidden_dim": 256,
        "dropout": 0.35,
        "cross_attn_temperature": 0.90,
        "cross_num_heads": 4,
        "lambda_conf": 0.15,
        "lambda_qc": 0.10,
        "lambda_alpha_q": 0.6,
        "lambda_alpha_k": 0.6,
        "lambda_cross_q": 0.8,
        "ctr_top_frac": 0.20,
        "ctr_margin": 0.10,
        "q_topk_frac": 0.10,
        "use_hyperedge_sampling": True,
        "wsi_cache_variants": 6,
        "eval_wsi_cache_variant_index": 0,
        "coverage_ratio": 0.50,
        "prior_hot_ratio": 0.25,
        "hard_replay_ratio": 0.25,
        "artifact_cap_ratio": 0.15,
        "wsi_N": 500,
        "wsi_K": 50,
        "batch": 4,
        "seed": 42,
        "amp": True,
        "amp_dtype": "auto",
        "deterministic": False,
        "tf32": True,
        "ece_bins": 10,
        "ause_steps": 20,
    }
    for name, value in defaults.items():
        normalized.setdefault(name, value)
    return SimpleNamespace(**normalized)


def checkpoint_stage(checkpoint: Dict[str, object], args: SimpleNamespace) -> int:
    if "stage" in checkpoint:
        return int(checkpoint["stage"])
    epoch = max(int(checkpoint.get("epoch", 1)) - 1, 0)
    return resolve_training_stage(epoch, args.stage1_epochs, args.stage2_epochs)


def checkpoint_sigma_temperature(checkpoint: Dict[str, object]) -> float:
    if "sigma_temperature" in checkpoint:
        value = checkpoint["sigma_temperature"]
    else:
        metrics = checkpoint.get("val_metrics", {})
        value = metrics.get("sigma_temperature_fit", 1.0) if isinstance(metrics, dict) else 1.0
    value = float(value)
    return value if np.isfinite(value) and value > 0 else 1.0


def checkpoint_state_dict(checkpoint: Dict[str, object]) -> Dict[str, torch.Tensor]:
    state = dict(checkpoint["model_state_dict"])
    if state and all(key.startswith("_orig_mod.") for key in state):
        state = {key[len("_orig_mod."):]: value for key, value in state.items()}
    return state


def build_model_from_checkpoint(
    checkpoint: Dict[str, object],
    args: SimpleNamespace,
    wsi_dim: int,
    gene_dim: int,
    num_pathways: int,
    device: torch.device,
) -> KP2Surv:
    prior_table = checkpoint.get("prior_table")
    if not isinstance(prior_table, dict):
        prior_table = load_weak_prior_table(None)
    model = KP2Surv(
        wsi_in_dim=wsi_dim,
        gene_in_dim=gene_dim,
        hidden_dim=int(args.hidden_dim),
        num_pathways=num_pathways,
        dropout=float(args.dropout),
        cross_attn_temperature=float(args.cross_attn_temperature),
        cross_num_heads=int(args.cross_num_heads),
        lambda_conf=float(args.lambda_conf),
        lambda_qc=float(args.lambda_qc),
        lambda_alpha_q=float(args.lambda_alpha_q),
        lambda_alpha_k=float(args.lambda_alpha_k),
        lambda_cross_q=float(args.lambda_cross_q),
        ctr_top_frac=float(args.ctr_top_frac),
        ctr_margin=float(args.ctr_margin),
        q_topk_frac=float(args.q_topk_frac),
        stochastic_input_dropout=float(args.stochastic_input_dropout),
        prior_table=prior_table,
    ).to(device)
    incompatible = model.load_state_dict(checkpoint_state_dict(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate five TCGA fold checkpoints on an external cohort."
    )
    parser.add_argument("--cohort", required=True, help="External cohort label.")
    parser.add_argument("--clinical", required=True, help="External clinical CSV, TSV, or spreadsheet.")
    parser.add_argument("--wsi_dir", required=True, help="External WSI hypergraph directory.")
    parser.add_argument("--expr", required=True, help="External case-by-gene expression CSV.")
    parser.add_argument("--mask", required=True, help="TCGA pathway-gene mask used for training.")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing TCGA fold checkpoints.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--gene_dir", default="", help="Prebuilt external pathway-structure directory.")
    parser.add_argument("--pathway_template", default="", help="TCGA pathway template used when gene_dir is omitted.")
    parser.add_argument("--gene_mapping", default="", help="Optional source_gene,target_gene CSV.")
    parser.add_argument("--checkpoint_pattern", default="fold*_best.pt")
    parser.add_argument("--expected_checkpoints", type=int, default=5)
    parser.add_argument("--max_missing_gene_fraction", type=float, default=0.10)
    parser.add_argument("--batch", type=int, default=0, help="Inference batch size; 0 restores checkpoint value.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if not cli.gene_dir and not cli.pathway_template:
        raise ValueError("Provide either --gene_dir or --pathway_template")
    ensure_dir(cli.out)

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    expression_source = load_external_expression(cli.expr, cli.gene_mapping)
    clinical = ClinicalDataLoader.load(cli.clinical)
    eligible_ids = sorted(set(expression_source.index) & set(clinical))
    if not eligible_ids:
        raise RuntimeError("No patients overlap between external expression and clinical data")

    gene_dir = cli.gene_dir
    if not gene_dir:
        gene_dir = build_external_pathway_structures(
            pathway_template=cli.pathway_template,
            wsi_dir=cli.wsi_dir,
            output_dir=os.path.join(cli.out, "pathway_structures"),
            eligible_case_ids=eligible_ids,
        )

    checkpoints = discover_checkpoints(
        cli.checkpoint_dir,
        pattern=cli.checkpoint_pattern,
        expected_count=cli.expected_checkpoints,
    )
    mask_source = load_pathway_gene_mask(cli.mask, torch.device("cpu"))
    fold_records = []

    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = normalized_checkpoint_args(checkpoint)
        set_global_seed(int(args.seed))
        configure_runtime(
            device=device,
            deterministic=bool(args.deterministic),
            tf32=bool(args.tf32),
        )

        gene_order = [str(g) for g in checkpoint.get("gene_order", [])]
        if not gene_order:
            raise ValueError(f"Checkpoint has no gene_order: {checkpoint_path}")
        expression, alignment = align_external_expression(
            expression_source,
            gene_order,
            max_missing_fraction=cli.max_missing_gene_fraction,
        )
        checkpoint_pathways = checkpoint.get("pathway_names")
        mask = align_mask_to_checkpoint(
            mask_source,
            gene_order,
            checkpoint_pathways,
            device,
        )

        train_mean = np.asarray(checkpoint.get("train_gene_mean"), dtype=np.float32)
        train_std = np.asarray(checkpoint.get("train_gene_std"), dtype=np.float32)
        if train_mean.shape != (len(gene_order),) or train_std.shape != (len(gene_order),):
            raise ValueError(
                f"Checkpoint normalization shape does not match gene_order: {checkpoint_path}"
            )
        train_std = train_std.copy()
        train_std[train_std < 1e-6] = 1.0
        mean_tensor = torch.from_numpy(train_mean).to(device)
        std_tensor = torch.from_numpy(train_std).to(device)

        fold = int(checkpoint.get("fold", len(fold_records)))
        dataset = MultiModalSurvivalDataset(
            cli.wsi_dir,
            gene_dir,
            clinical,
            expression,
            gene_order,
            ids=set(eligible_ids),
            wsi_N=int(args.wsi_N),
            wsi_K=int(args.wsi_K),
            seed=int(args.seed),
            use_mixed_sampling=bool(args.use_mixed_sampling),
            use_hyperedge_sampling=bool(args.use_hyperedge_sampling),
            wsi_cache_dir=os.path.join(cli.out, "wsi_subgraph_cache"),
            stable_wsi_cache_seed=True,
            rebuild_wsi_cache=False,
            wsi_cache_variants=max(1, int(args.wsi_cache_variants)),
            wsi_cache_variant_mode="fixed",
            wsi_cache_variant_index=int(args.eval_wsi_cache_variant_index),
            coverage_ratio=float(args.coverage_ratio),
            prior_hot_ratio=float(args.prior_hot_ratio),
            hard_replay_ratio=float(args.hard_replay_ratio),
            artifact_cap_ratio=float(args.artifact_cap_ratio),
        )
        if len(dataset) == 0:
            raise RuntimeError(f"No matched external patients for checkpoint fold {fold}")

        batch_size = int(cli.batch) if cli.batch > 0 else int(args.batch)
        loader = DataLoader(
            dataset,
            batch_size=max(1, batch_size),
            shuffle=False,
            num_workers=max(0, int(cli.num_workers)),
            collate_fn=survival_collate,
        )
        first = dataset[0]
        model = build_model_from_checkpoint(
            checkpoint,
            args,
            wsi_dim=int(first["wsi"].x.shape[1]),
            gene_dim=len(gene_order),
            num_pathways=int(mask.size(0)),
            device=device,
        )

        stage = checkpoint_stage(checkpoint, args)
        if stage <= 1:
            enable_stochastic_variance, enable_disagreement, stochastic_passes = False, False, 1
        elif stage == 2:
            enable_stochastic_variance, enable_disagreement = True, True
            stochastic_passes = max(1, int(args.stochastic_passes_stage2))
        else:
            enable_stochastic_variance, enable_disagreement = True, True
            stochastic_passes = max(1, int(args.stochastic_passes_stage3))
        sigma_temperature = checkpoint_sigma_temperature(checkpoint)

        amp_dtype_name = str(args.amp_dtype)
        if amp_dtype_name == "bf16":
            amp_dtype = torch.bfloat16
        elif amp_dtype_name == "fp16":
            amp_dtype = torch.float16
        else:
            amp_dtype = (
                torch.bfloat16
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
            )
        metrics = evaluate(
            model=model,
            loader=loader,
            mu_g=mean_tensor,
            sd_g=std_tensor,
            mask=mask,
            device=device,
            stochastic_passes=stochastic_passes,
            enable_stochastic_variance=enable_stochastic_variance,
            enable_disagreement=enable_disagreement,
            ece_bins=int(args.ece_bins),
            ause_steps=int(args.ause_steps),
            sigma_temperature=sigma_temperature,
            return_raw=True,
            use_amp=bool(args.amp),
            amp_dtype=amp_dtype,
        )

        predictions = prediction_frame_from_raw(
            metrics,
            fold=fold,
            seed=int(args.seed),
            split=str(cli.cohort),
            sigma_temperature=sigma_temperature,
            stage=stage,
            epoch=int(checkpoint.get("epoch", 0)),
        )
        prediction_path = os.path.join(cli.out, f"fold{fold}_{cli.cohort}_predictions.csv")
        save_csv(predictions, prediction_path)

        record = {
            "cohort": str(cli.cohort),
            "fold": fold,
            "checkpoint": os.path.basename(checkpoint_path),
            "patients": int(len(dataset)),
            "c_index": float(metrics["c_index"]),
            "nll": float(metrics["nll"]),
            "ibs": float(metrics["ibs"]),
            "ece": float(metrics["ece"]),
            "ause": float(metrics["ause"]),
            "uncertainty_error_gap": float(metrics["uncertainty_error_gap"]),
            "stage": stage,
            "stochastic_passes": stochastic_passes,
            "stochastic_input_dropout": float(args.stochastic_input_dropout),
            "sigma_temperature": sigma_temperature,
            "missing_gene_count": int(alignment["missing_gene_count"]),
            "missing_gene_fraction": float(alignment["missing_gene_fraction"]),
            "prediction_file": os.path.basename(prediction_path),
        }
        fold_records.append(record)
        save_json(
            os.path.join(cli.out, f"fold{fold}_{cli.cohort}_alignment.json"),
            alignment,
        )

    fold_frame = pd.DataFrame(fold_records).sort_values("fold")
    save_csv(fold_frame, os.path.join(cli.out, "external_metrics_by_checkpoint.csv"))
    metric_summary = {}
    for metric_name in [
        "c_index",
        "nll",
        "ibs",
        "ece",
        "ause",
        "uncertainty_error_gap",
    ]:
        metric_mean, metric_std = sample_mean_and_std(
            fold_frame[metric_name].to_numpy()
        )
        metric_summary[f"{metric_name}_mean"] = metric_mean
        metric_summary[f"{metric_name}_sample_std"] = metric_std
    summary = {
        "cohort": str(cli.cohort),
        "checkpoint_count": int(len(fold_frame)),
        **metric_summary,
        "standard_deviation_ddof": 1,
        "retraining": False,
        "external_fine_tuning": False,
        "outcomes_used_for_model_selection": False,
    }
    save_json(os.path.join(cli.out, "external_summary.json"), summary)
    print(
        f"{cli.cohort}: C-index={summary['c_index_mean']:.4f} +/- "
        f"{summary['c_index_sample_std']:.4f} "
        f"across {len(fold_frame)} TCGA checkpoints (sample SD)"
    )


if __name__ == "__main__":
    main()
