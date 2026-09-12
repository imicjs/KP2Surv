#!/usr/bin/env python3
"""Shared runtime imports, reproducibility helpers, splits, and exports for KP2Surv."""

from __future__ import annotations

import os
import math
import random
import warnings
import argparse
import json
import hashlib
from datetime import datetime
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import HypergraphConv
from torch_geometric.data import Data as GeomData, Batch as GeomBatch
from torch_scatter import scatter_add, scatter_mean

try:
    from sklearn.cluster import MiniBatchKMeans
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

cudnn.benchmark = False

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
PIN_MEMORY = os.environ.get("PIN_MEMORY", "1").lower() not in {"0", "false", "no"}
FIXED_WSI_SUBGRAPH = True     # Cache sub-hypergraphs for speed
LOGVAR_MIN = -8.0
LOGVAR_MAX = 8.0


class HypergraphData(GeomData):
    """PyG Data with correct batching increments for hyperedge incidence matrices."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "hyperedge_index":
            num_nodes = int(self.num_nodes) if self.num_nodes is not None else 0
            if hasattr(self, "num_hyperedges") and self.num_hyperedges is not None:
                num_hyperedges = int(self.num_hyperedges)
            elif value.numel() > 0:
                num_hyperedges = int(value[1].max().item()) + 1
            else:
                num_hyperedges = 0
            # NOTE: must be [2, 1] so it can broadcast with hyperedge_index shape [2, E]
            return torch.tensor([[num_nodes], [num_hyperedges]], device=value.device)
        if key in {"orig_node_idx", "orig_hyperedge_idx"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def infer_num_hyperedges(hyperedge_index: torch.Tensor) -> int:
    if hyperedge_index is None or hyperedge_index.numel() == 0:
        return 0
    return int(hyperedge_index[1].max().item()) + 1


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def configure_runtime(
    device: torch.device,
    deterministic: bool = False,
    tf32: bool = True,
) -> None:
    deterministic = bool(deterministic)
    cudnn.deterministic = deterministic
    cudnn.benchmark = (device.type == "cuda") and (not deterministic)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        # "high" enables TensorFloat-32 fast path for matmul on supported GPUs.
        torch.set_float32_matmul_precision("high" if tf32 else "highest")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def stable_hash_int(text: str, modulo: int = 2_000_000_000) -> int:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % int(modulo)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(df: pd.DataFrame, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    df.to_csv(path, index=False, encoding="utf-8")


def sample_mean_and_std(values) -> Tuple[float, float]:
    """Return the arithmetic mean and sample standard deviation."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    return mean, std


def parse_export_splits(spec: str) -> List[str]:
    allowed = {"train", "calib", "val"}
    raw = [s.strip().lower() for s in str(spec).split(",") if s.strip()]
    if not raw:
        return ["val"]
    if "all" in raw:
        return ["train", "calib", "val"]
    out: List[str] = []
    for name in raw:
        if name in allowed and name not in out:
            out.append(name)
    return out if out else ["val"]


def split_ids_for_fold(
    full_ds,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    fold: int,
    stage3_calibrate: bool,
    calib_ratio: float,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    train_ids_all = [full_ds.samples[i]["case_id"] for i in train_idx]
    val_ids = [full_ds.samples[i]["case_id"] for i in val_idx]

    rng_split = np.random.RandomState(int(seed) + 97 * (int(fold) + 1))
    shuffled_train_ids = list(train_ids_all)
    rng_split.shuffle(shuffled_train_ids)

    calib_ids: List[str] = []
    train_ids = list(shuffled_train_ids)
    if stage3_calibrate:
        clipped = float(np.clip(calib_ratio, 0.0, 0.5))
        n_calib = int(round(len(shuffled_train_ids) * clipped))
        if n_calib >= 1 and (len(shuffled_train_ids) - n_calib) >= 2:
            calib_ids = shuffled_train_ids[:n_calib]
            train_ids = shuffled_train_ids[n_calib:]

    return train_ids_all, train_ids, calib_ids, val_ids


def build_survival_strata(
    samples: List[Dict[str, Any]],
    n_splits: int,
    max_time_bins: int = 4,
) -> Tuple[np.ndarray, str]:
    """Build joint event/time strata for manuscript-aligned cross-validation."""
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >=2, got {n_splits}")
    if len(samples) < n_splits:
        raise ValueError(f"Need at least {n_splits} patients, got {len(samples)}")

    times = np.asarray([float(s["time"]) for s in samples], dtype=np.float64)
    events = np.asarray([int(s["event"]) for s in samples], dtype=np.int64)
    if not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("All survival times must be finite and positive before splitting")

    # Quantile bins are formed within each event group, then crossed with event
    # status. Decrease the bin count until every joint stratum can populate all
    # folds. This balances both censoring status and the observed-time ranks.
    for num_bins in range(max(2, int(max_time_bins)), 1, -1):
        labels = np.full(len(samples), -1, dtype=np.int64)
        next_label = 0
        valid = True
        for event_value in sorted(np.unique(events).tolist()):
            idx = np.flatnonzero(events == event_value)
            try:
                local_bins = pd.qcut(
                    times[idx],
                    q=num_bins,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                valid = False
                break
            local_bins = np.asarray(local_bins)
            if pd.isna(local_bins).any():
                valid = False
                break
            local_bins = local_bins.astype(np.int64, copy=False)
            if np.unique(local_bins).size < 2:
                valid = False
                break
            for local_bin in sorted(np.unique(local_bins).tolist()):
                members = idx[local_bins == local_bin]
                labels[members] = next_label
                next_label += 1

        if valid and np.all(labels >= 0):
            counts = np.bincount(labels)
            if counts.size > 1 and int(counts.min()) >= n_splits:
                return labels, f"event_status x within-event {num_bins}-quantile observed-time bins"

    event_counts = np.bincount(events)
    event_counts = event_counts[event_counts > 0]
    if event_counts.size > 1 and int(event_counts.min()) >= n_splits:
        print("  [WARN] Joint event/time strata are infeasible; using event-status stratification.")
        return events, "event status only (joint event/time stratification infeasible)"

    raise ValueError(
        "Cannot construct survival-stratified folds: at least one outcome group "
        f"has fewer than {n_splits} patients."
    )


def export_fold_splits(
    export_dir: str,
    fold: int,
    train_ids: List[str],
    calib_ids: List[str],
    val_ids: List[str],
) -> None:
    split_map = {
        "train": train_ids,
        "calib": calib_ids,
        "val": val_ids,
    }
    for split_name, ids in split_map.items():
        rows = [{"fold": int(fold), "split": split_name, "case_id": cid} for cid in ids]
        save_csv(pd.DataFrame(rows), os.path.join(export_dir, f"fold{fold}_{split_name}_split.csv"))


def prediction_frame_from_raw(
    raw_metrics: Dict[str, Any],
    fold: int,
    seed: int,
    split: str,
    sigma_temperature: float,
    stage: int,
    epoch: int,
) -> pd.DataFrame:
    mu = np.asarray(raw_metrics.get("raw_mu", []), dtype=np.float64)
    sigma = np.asarray(raw_metrics.get("raw_sigma", []), dtype=np.float64)
    time_arr = np.asarray(raw_metrics.get("raw_time", []), dtype=np.float64)
    event_arr = np.asarray(raw_metrics.get("raw_event", []), dtype=np.float64)
    risk = np.asarray(raw_metrics.get("raw_risk", []), dtype=np.float64)
    case_ids = list(raw_metrics.get("raw_case_id", []))

    n = int(mu.shape[0])
    if risk.shape[0] != n:
        risk = np.full((n,), np.nan, dtype=np.float64)
    if sigma.shape[0] != n:
        sigma = np.full((n,), np.nan, dtype=np.float64)
    if time_arr.shape[0] != n:
        time_arr = np.full((n,), np.nan, dtype=np.float64)
    if event_arr.shape[0] != n:
        event_arr = np.full((n,), np.nan, dtype=np.float64)
    if len(case_ids) != n:
        case_ids = [f"unknown_{i}" for i in range(n)]

    df = pd.DataFrame(
        {
            "case_id": case_ids,
            "fold": int(fold),
            "seed": int(seed),
            "split": str(split),
            "epoch": int(epoch),
            "stage": int(stage),
            "sigma_temperature": float(sigma_temperature),
            "time": time_arr,
            "event": event_arr,
            "risk": risk,
            "mu": mu,
            "sigma": sigma,
        }
    )
    return df


def km_frame_from_predictions(
    pred_df: pd.DataFrame,
    threshold_rule: str = "median",
    quantile: float = 0.5,
) -> Tuple[pd.DataFrame, float]:
    df = pred_df.copy()
    if df.empty:
        df["risk_group"] = []
        return df, float("nan")
    valid_risk = pd.to_numeric(df["risk"], errors="coerce")
    if threshold_rule == "quantile":
        q = float(np.clip(quantile, 0.0, 1.0))
        threshold = float(np.nanquantile(valid_risk.to_numpy(dtype=np.float64), q))
    else:
        threshold = float(np.nanmedian(valid_risk.to_numpy(dtype=np.float64)))
    df["threshold_rule"] = threshold_rule
    df["threshold_value"] = threshold
    df["risk_group"] = np.where(valid_risk >= threshold, "high", "low")
    return df, threshold


def load_weak_prior_table(prior_table_path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load external prior table config.
    For WSI, continuous priors are preferred over discrete type maps.
    """
    fallback = {
        "wsi": {
            "default": {"alpha_prior": 0.5, "q_prior": 0.5, "alpha_conf": 0.1, "q_conf": 0.1},
            "continuous": {
                "alpha_weights": {
                    "edge_score": 0.25,
                    "heterogeneity": 0.25,
                    "cluster_rarity": 0.20,
                    "spatial_boundary": 0.15,
                    "tumor_prob": 0.15,
                },
                "q_weights": {
                    "artifact_score": 0.45,
                    "blur": 0.35,
                    "inv_stain_qc": 0.20,
                },
                "alpha_mix": 1.0,
                "q_mix": 1.0,
                "alpha_conf_base": 0.10,
                "alpha_conf_scale": 0.60,
                "q_conf_base": 0.15,
                "q_conf_scale": 0.75,
            },
        },
        "gene": {
            "default": {"alpha_prior": 0.5, "q_prior": 0.5, "alpha_conf": 0.1, "q_conf": 0.1},
        },
    }
    if not prior_table_path:
        return fallback
    if not os.path.isfile(prior_table_path):
        raise FileNotFoundError(f"Weak-prior table not found: {prior_table_path}")
    with open(prior_table_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid weak-prior table in {prior_table_path}: expected a JSON object.")

    def _deep_update(dst: Dict, src: Dict) -> Dict:
        for key, val in src.items():
            if key in dst and isinstance(dst[key], dict) and isinstance(val, dict):
                _deep_update(dst[key], val)
            else:
                dst[key] = val
        return dst

    # Merge with fallback to guarantee keys exist.
    out = fallback
    _deep_update(out, obj)
    return out

def extract_case_id(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    parts = base.split('-')
    return '-'.join(parts[:3]) if len(parts) >= 3 else base

def concordance_index_fallback(time_arr: np.ndarray, risk_arr: np.ndarray, event_arr: np.ndarray) -> float:
    """ Harrell's C-Index. """
    n = len(time_arr)
    concordant = 0.0
    permissible = 0.0
    ties = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if time_arr[i] == time_arr[j]:
                continue
            if time_arr[i] < time_arr[j]:
                if event_arr[i] != 1: continue
                permissible += 1
                if risk_arr[i] > risk_arr[j]: concordant += 1
                elif risk_arr[i] == risk_arr[j]: ties += 1
            else:
                if event_arr[j] != 1: continue
                permissible += 1
                if risk_arr[j] > risk_arr[i]: concordant += 1
                elif risk_arr[i] == risk_arr[j]: ties += 1
    if permissible == 0: return 0.5
    return float((concordant + 0.5 * ties) / permissible)
