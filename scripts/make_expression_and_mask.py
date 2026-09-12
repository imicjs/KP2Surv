#!/usr/bin/env python3
"""
Create the expression matrix and pathway-gene mask used by KP2Surv.

Outputs
-------
1) gene_expression.csv  : rows = case_id (TCGA-XX-YYYY), cols = genes (same order as gene_names in mask)
2) pathway_gene_mask.npz: fields:
   - mask (P x G) uint8
   - pathway_names (P,) object
   - gene_names (G,) object

The template must be the same file used to build the patient pathway
hypergraphs so that pathway ordering remains consistent.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch

def extract_case_id(sample_id: str) -> str:
    parts = sample_id.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else sample_id

def tcga_sample_type_code(sample_id: str):
    parts = sample_id.split("-")
    if len(parts) >= 4 and len(parts[3]) >= 2 and parts[3][:2].isdigit():
        return int(parts[3][:2])
    return None

def choose_one_sample_per_case(sample_ids):
    primary = [s for s in sample_ids if "-01A-" in s]
    if primary:
        return sorted(primary)[0]
    tumor = []
    for s in sample_ids:
        code = tcga_sample_type_code(s)
        if code is not None and code < 10:
            tumor.append(s)
    if tumor:
        return sorted(tumor)[0]
    return sorted(sample_ids)[0]

def load_mrna_csv(path: str, gene_col: str = "Gene_Symbol", drop_meta_cols: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path)
    if gene_col not in df.columns:
        raise ValueError(f"Cannot find gene_col='{gene_col}' in {path}. Available: {list(df.columns)[:10]}...")
    df = df.set_index(gene_col)
    if drop_meta_cols > 0:
        df = df.iloc[:, drop_meta_cols:]
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()
    if df.values.max() > 20:
        df = np.log1p(df)
    return df

def build_expr_table(mrna_df: pd.DataFrame) -> pd.DataFrame:
    sxg = mrna_df.T  # samples x genes
    case_to_samples = {}
    for sid in sxg.index:
        cid = extract_case_id(sid)
        case_to_samples.setdefault(cid, []).append(sid)
    picked = {cid: choose_one_sample_per_case(sids) for cid, sids in case_to_samples.items()}
    rows = []
    for cid, sid in picked.items():
        row = sxg.loc[sid].copy()
        row.name = cid
        rows.append(row)
    cxg = pd.DataFrame(rows).sort_index()
    cxg.index.name = "case_id"
    return cxg

def build_mask_from_template(template_path: str, gene_names):
    template = torch.load(template_path, map_location="cpu", weights_only=False)
    pathway_names = list(template.pathway_names)
    pathway_genes_list = list(template.pathway_genes)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    P, G = len(pathway_names), len(gene_names)
    mask = np.zeros((P, G), dtype=np.uint8)
    for p, genes in enumerate(pathway_genes_list):
        for g in genes:
            j = gene_to_idx.get(str(g))
            if j is not None:
                mask[p, j] = 1
    return mask, pathway_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrna_csv", required=True, help="TCGA expression CSV (genes x samples)")
    ap.add_argument("--template_pt", required=True, help="Pathway-node template .pt saved when building gene hypergraph")
    ap.add_argument("--out_expr", required=True, help="Output gene_expression.csv (cases x genes)")
    ap.add_argument("--out_mask", required=True, help="Output pathway_gene_mask.npz")
    ap.add_argument("--gene_col", default="Gene_Symbol")
    ap.add_argument("--drop_meta_cols", type=int, default=1)
    ap.add_argument("--expected_pathways", type=int, default=186, help="Expected mask rows; 0 disables the check.")
    args = ap.parse_args()

    mrna_df = load_mrna_csv(args.mrna_csv, gene_col=args.gene_col, drop_meta_cols=args.drop_meta_cols)
    expr = build_expr_table(mrna_df)

    gene_names = list(expr.columns.astype(str))
    mask, pathway_names = build_mask_from_template(args.template_pt, gene_names)
    if args.expected_pathways > 0 and mask.shape[0] != args.expected_pathways:
        raise RuntimeError(
            f"Expected {args.expected_pathways} pathways for the manuscript configuration, "
            f"but the template contains {mask.shape[0]}."
        )

    os.makedirs(os.path.dirname(args.out_expr) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_mask) or ".", exist_ok=True)

    expr.to_csv(args.out_expr, index=True)
    np.savez_compressed(
        args.out_mask,
        mask=mask,
        pathway_names=np.array(pathway_names, dtype=object),
        gene_names=np.array(gene_names, dtype=object),
    )

    print("Saved:")
    print(f"  EXPR: {args.out_expr}  shape={expr.shape}  (cases x genes)")
    print(f"  MASK: {args.out_mask}  mask_shape={mask.shape}  (pathways x genes)")
    print("Sanity:")
    print(f"  mask density: {mask.mean():.6f}")

if __name__ == "__main__":
    main()
