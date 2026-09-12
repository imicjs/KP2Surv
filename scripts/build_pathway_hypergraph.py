#!/usr/bin/env python3
"""
Build pathway-node hypergraphs for KP2Surv.

Each gene induces a hyperedge connecting all pathways that contain it. The
offline files store only the shared structure and sample mapping. Expression
features are standardized with training-fold statistics and assembled online.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data as GeomData
from collections import defaultdict
from tqdm import tqdm


def load_kegg_wide_format(kegg_csv_path):
    """Read a wide-format KEGG pathway file."""
    if not os.path.isfile(kegg_csv_path):
        raise FileNotFoundError(f"KEGG file does not exist: {kegg_csv_path}")
    
    df = pd.read_csv(kegg_csv_path)
    print(f"Reading KEGG pathway file: {kegg_csv_path}")
    print(f"  Pathways in source: {len(df)}")
    
    pathway_dict = {}
    gene_count_stats = []
    
    for _, row in df.iterrows():
        pathway_name = str(row.iloc[0]).strip()
        genes = []
        seen_genes = set()
        for val in row.iloc[1:]:
            if pd.notna(val):
                gene = str(val).strip()
                if gene and gene.lower() not in ['nan', 'na', ''] and gene not in seen_genes:
                    genes.append(gene)
                    seen_genes.add(gene)
        if genes:
            pathway_dict[pathway_name] = genes
            gene_count_stats.append(len(genes))
    
    print(f"  Parsed pathways: {len(pathway_dict)}")
    if not gene_count_stats:
        raise ValueError(f"No valid pathways were found in {kegg_csv_path}")
    print(
        f"  Genes per pathway: min={min(gene_count_stats)}, "
        f"max={max(gene_count_stats)}, "
        f"mean={np.mean(gene_count_stats):.1f}"
    )
    
    return pathway_dict


def load_expr_matrix_from_csv(csv_path, gene_col="Gene_Symbol", drop_meta_cols=1):
    """
    Read a gene-by-sample expression matrix.

    A log1p transform is applied when needed. Standardization is fitted within
    each training fold by the training program.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Expression CSV does not exist: {csv_path}")
    
    print(f"\nReading expression matrix: {csv_path}")
    df = pd.read_csv(csv_path)

    if gene_col not in df.columns:
        raise ValueError(f"Column '{gene_col}' was not found in the CSV file")

    df = df.set_index(gene_col)
    if drop_meta_cols > 0:
        df = df.iloc[:, drop_meta_cols:]

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.index.duplicated().any():
        dup_genes = df.index[df.index.duplicated()].nunique()
        print(f"  Duplicate gene symbols: {dup_genes}; aggregating by mean")
        df = df.groupby(level=0).mean()

    df = df.fillna(0.0)
    print(f"  Input range: min={df.values.min():.2f}, max={df.values.max():.2f}")

    # The elementwise transform does not mix information across samples.
    if df.values.max() > 20:
        print("  Applying log1p transform")
        df = np.log1p(df)
        print(f"  Transformed range: min={df.values.min():.2f}, max={df.values.max():.2f}")

    print("  Global z-score normalization is not applied")
    print(f"  Matrix shape: {df.shape[0]} genes x {df.shape[1]} samples")
    
    return df


def filter_pathways_by_coverage(
    pathway_dict,
    expr_gene_set,
    min_genes=10,
    max_genes=400,
    min_coverage=0.5
):
    """Filter pathways by observed gene count and expression coverage."""
    filtered = {}
    stats = {'total': len(pathway_dict), 'too_few': 0, 'too_many': 0, 'low_coverage': 0}
    
    for pathway_name, genes in pathway_dict.items():
        genes_in_expr = [g for g in genes if g in expr_gene_set]
        n_total = len(genes)
        n_in_expr = len(genes_in_expr)
        
        if n_total == 0:
            continue
        
        coverage = n_in_expr / n_total
        
        if n_in_expr < min_genes:
            stats['too_few'] += 1
            continue
        if n_in_expr > max_genes:
            stats['too_many'] += 1
            continue
        if coverage < min_coverage:
            stats['low_coverage'] += 1
            continue
        
        filtered[pathway_name] = genes_in_expr
    
    print("\nPathway filtering:")
    print(f"  Source pathways: {stats['total']}")
    print(f"  Below {min_genes} observed genes: {stats['too_few']}")
    print(f"  Above {max_genes} observed genes: {stats['too_many']}")
    print(f"  Coverage below {min_coverage}: {stats['low_coverage']}")
    print(f"  Retained pathways: {len(filtered)}")
    
    return filtered


def build_gene_induced_hyperedges(
    pathway_dict,
    pathway_names,
    min_pathways_per_gene=2,
    max_pathways_per_gene=50
):
    """Create one hyperedge per gene over the pathways containing that gene."""
    num_pathways = len(pathway_names)
    pathway_to_idx = {name: i for i, name in enumerate(pathway_names)}
    
    print("\nBuilding gene-induced hyperedges")
    print(f"  Pathways: {num_pathways}")
    print(f"  Gene degree range: [{min_pathways_per_gene}, {max_pathways_per_gene}]")
    
    # Map each gene to the pathways that contain it.
    gene_to_pathway_indices = defaultdict(set)
    for pathway_name in pathway_names:
        if pathway_name not in pathway_dict:
            continue
        pathway_idx = pathway_to_idx[pathway_name]
        for gene in pathway_dict[pathway_name]:
            gene_to_pathway_indices[gene].add(pathway_idx)
    
    print(f"  Genes represented: {len(gene_to_pathway_indices)}")
    
    raw_hyperedges = []
    hyperedge_genes = []
    degree_stats = []
    
    for gene, pathway_indices in gene_to_pathway_indices.items():
        degree = len(pathway_indices)
        degree_stats.append(degree)
        
        if degree < min_pathways_per_gene:
            continue
        if degree > max_pathways_per_gene:
            continue
        
        raw_hyperedges.append(sorted(pathway_indices))
        hyperedge_genes.append(gene)
    
    if degree_stats:
        print(
            "  Gene degree distribution: "
            f"min={min(degree_stats)}, max={max(degree_stats)}, "
            f"mean={np.mean(degree_stats):.1f}, "
            f"median={np.median(degree_stats):.0f}"
        )
    else:
        print("  Gene degree distribution: no genes represented")
    print(f"  Hyperedges before deduplication: {len(raw_hyperedges)}")
    
    # Merge genes that induce the same pathway set.
    unique_hyperedges = {}
    for he, gene in zip(raw_hyperedges, hyperedge_genes):
        key = tuple(he)
        if key not in unique_hyperedges:
            unique_hyperedges[key] = {'nodes': list(he), 'genes': [gene]}
        else:
            unique_hyperedges[key]['genes'].append(gene)
    
    final_hyperedges = [v['nodes'] for v in unique_hyperedges.values()]
    final_genes = [v['genes'] for v in unique_hyperedges.values()]
    
    print(f"  Unique hyperedges: {len(final_hyperedges)}")
    
    covered_nodes = set()
    for he in final_hyperedges:
        covered_nodes.update(he)
    
    uncovered_nodes = set(range(num_pathways)) - covered_nodes
    print(
        f"  Node coverage: {len(covered_nodes)}/{num_pathways} "
        f"({100 * len(covered_nodes) / num_pathways:.1f}%)"
    )
    
    # Add self-hyperedges for otherwise isolated pathways.
    num_self_hyperedges = 0
    for node_idx in uncovered_nodes:
        final_hyperedges.append([node_idx])
        final_genes.append([f"__SELF_{pathway_names[node_idx]}__"])
        num_self_hyperedges += 1
    
    if num_self_hyperedges > 0:
        print(f"  Self-hyperedges added: {num_self_hyperedges}")
    
    he_sizes = [len(he) for he in final_hyperedges]
    print(f"  Final hyperedges: {len(final_hyperedges)}")
    print(
        f"  Hyperedge size: min={min(he_sizes)}, "
        f"max={max(he_sizes)}, mean={np.mean(he_sizes):.1f}"
    )
    
    size_counts = defaultdict(int)
    for s in he_sizes:
        if s == 1:
            size_counts['1 (self)'] += 1
        elif s == 2:
            size_counts['2'] += 1
        elif s <= 5:
            size_counts['3-5'] += 1
        elif s <= 10:
            size_counts['6-10'] += 1
        else:
            size_counts['>10'] += 1
    
    print(f"  Hyperedge size groups: {dict(size_counts)}")
    
    return final_hyperedges, final_genes


def to_hyperedge_index(hyperedges, num_nodes):
    """Convert hyperedges to PyTorch Geometric incidence format."""
    if not hyperedges:
        return torch.empty((2, 0), dtype=torch.long)

    node_indices = []
    he_indices = []
    for e_id, he in enumerate(hyperedges):
        for v in he:
            node_indices.append(v)
            he_indices.append(e_id)

    return torch.tensor([node_indices, he_indices], dtype=torch.long)


def build_pathway_node_hypergraph_template(
    kegg_csv_path,
    mrna_csv_path,
    min_genes=10,
    max_genes=400,
    min_coverage=0.5,
    min_pathways_per_gene=2,
    max_pathways_per_gene=50,
    gene_col="Gene_Symbol",
    drop_meta_cols=1,
    expected_pathways=0,
    save_path=None,
):
    """Build the shared pathway-node hypergraph structure."""
    pathway_dict = load_kegg_wide_format(kegg_csv_path)

    mrna_df = load_expr_matrix_from_csv(
        mrna_csv_path,
        gene_col=gene_col,
        drop_meta_cols=drop_meta_cols,
    )
    expr_gene_set = set(mrna_df.index)
    
    filtered_pathway_dict = filter_pathways_by_coverage(
        pathway_dict,
        expr_gene_set,
        min_genes=min_genes,
        max_genes=max_genes,
        min_coverage=min_coverage,
    )
    
    if not filtered_pathway_dict:
        raise RuntimeError("No pathways remain after filtering")

    pathway_names = sorted(filtered_pathway_dict.keys())
    num_pathways = len(pathway_names)
    if int(expected_pathways) > 0 and num_pathways != int(expected_pathways):
        raise RuntimeError(
            f"Expected {expected_pathways} retained pathways for the manuscript configuration, "
            f"but built {num_pathways}. Check the KEGG version and expression gene identifiers."
        )
    
    print(f"\nPathway nodes: {num_pathways}")

    hyperedges, hyperedge_genes = build_gene_induced_hyperedges(
        filtered_pathway_dict,
        pathway_names,
        min_pathways_per_gene=min_pathways_per_gene,
        max_pathways_per_gene=max_pathways_per_gene,
    )
    
    hyperedge_index = to_hyperedge_index(hyperedges, num_pathways)
    
    template = GeomData(
        num_nodes=num_pathways,
        hyperedge_index=hyperedge_index,
        num_hyperedges=len(hyperedges),
    )
    
    template.pathway_names = pathway_names
    template.pathway_genes = [filtered_pathway_dict[name] for name in pathway_names]
    template.hyperedge_inducing_genes = hyperedge_genes
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(template, save_path)
        print(f"\nTemplate saved: {save_path}")
    
    return template, mrna_df, filtered_pathway_dict


def get_sample_type_from_tcga_id(sample_id):
    try:
        parts = sample_id.split('-')
        if len(parts) >= 4:
            code = int(parts[3][:2])
            if code < 10:
                return "Tumor"
            if code < 20:
                return "Normal"
    except Exception:
        pass
    return "Unknown"


def extract_case_id(name):
    parts = name.split('-')
    return '-'.join(parts[:3]) if len(parts) >= 3 else name


def map_wsi_fname_to_sample_id(basename, expr_sample_ids, case_to_samples):
    wsi_type = get_sample_type_from_tcga_id(basename)
    
    if basename in expr_sample_ids:
        return basename, extract_case_id(basename), "direct"

    parts = basename.split('-')
    if len(parts) >= 3:
        case_id = '-'.join(parts[:3])
        if case_id in case_to_samples:
            candidates = case_to_samples[case_id]
            
            if wsi_type == "Normal":
                valid = [s for s in candidates if get_sample_type_from_tcga_id(s) == "Normal"]
            else:
                valid = [s for s in candidates if get_sample_type_from_tcga_id(s) == "Tumor"]
            
            if not valid and wsi_type == "Unknown":
                valid = candidates
            
            if not valid:
                return None, case_id, "type_mismatch"

            primary = [s for s in valid if "-01A-" in s]
            if primary:
                return sorted(primary)[0], case_id, "case_prefix_primary"
            if valid:
                return sorted(valid)[0], case_id, "case_prefix_first"

    return None, "", "unmatched"


def create_patient_pathway_hypergraphs(
    template,
    mrna_df,
    wsi_hg_dir,
    save_dir,
    mapping_save_path=None,
):
    """Create a patient-matched pathway structure for each WSI hypergraph."""
    pathway_names = template.pathway_names
    num_pathways = template.num_nodes

    # Gene identifiers are stored as metadata; expression is built online.
    all_genes = list(mrna_df.index)
    num_genes = len(all_genes)
    
    print("\nCreating patient pathway hypergraphs")
    print(f"  Pathway nodes: {num_pathways}")
    print(f"  Genes in the online feature space: {num_genes}")
    print("  Expression features will be standardized and assembled during training")
    
    sample_ids = list(mrna_df.columns)
    sample_to_col = {sid: j for j, sid in enumerate(sample_ids)}
    expr_sample_ids = set(sample_ids)
    
    case_to_samples = defaultdict(list)
    for sid in sample_ids:
        case_id = extract_case_id(sid)
        case_to_samples[case_id].append(sid)
    
    if not os.path.isdir(wsi_hg_dir):
        raise NotADirectoryError(f"WSI hypergraph directory does not exist: {wsi_hg_dir}")
    
    os.makedirs(save_dir, exist_ok=True)
    wsi_files = sorted(f for f in os.listdir(wsi_hg_dir) if f.endswith(".pt"))
    print(f"  WSI hypergraph files: {len(wsi_files)}")
    
    num_built = 0
    num_skipped = 0
    mapping_records = []
    
    for fname in tqdm(wsi_files, desc="Building pathway hypergraphs"):
        basename = os.path.splitext(fname)[0]
        
        sample_id, case_id, match_type = map_wsi_fname_to_sample_id(
            basename, expr_sample_ids, case_to_samples
        )
        
        if sample_id is None or sample_id not in sample_to_col:
            num_skipped += 1
            continue
        
        # Empty features prevent accidental use of values fitted outside a fold.
        gene_hg = GeomData(
            x=torch.empty((num_pathways, 0), dtype=torch.float32),
            num_nodes=num_pathways,
            hyperedge_index=template.hyperedge_index.clone(),
            num_hyperedges=template.num_hyperedges,
        )
        
        gene_hg.pathway_names = pathway_names
        gene_hg.sample_id = sample_id
        gene_hg.all_genes = all_genes
        gene_hg.features_built_online = True
        
        out_path = os.path.join(save_dir, fname)
        torch.save(gene_hg, out_path)
        num_built += 1
        
        mapping_records.append({
            "wsi_file": fname,
            "basename": basename,
            "case_prefix": case_id,
            "sample_id": sample_id,
            "match_type": match_type,
        })
    
    print("\nPathway hypergraph generation complete:")
    print(f"  Created: {num_built}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Output directory: {save_dir}")
    
    if mapping_save_path:
        df_map = pd.DataFrame(mapping_records)
        os.makedirs(os.path.dirname(mapping_save_path) or ".", exist_ok=True)
        df_map.to_csv(mapping_save_path, index=False)
        print(f"  Mapping table: {mapping_save_path}")
    
    if num_built > 0:
        sample_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')][:3]
        print("\nOutput inspection:")
        for sf in sample_files:
            sample_path = os.path.join(save_dir, sf)
            sample_data = torch.load(sample_path)
            print(f"  {sf}:")
            print(f"    x.shape: {sample_data.x.shape}  # expression is built online")
            print(f"    hyperedge_index: {sample_data.hyperedge_index.shape}")


def main():
    print("=" * 70)
    print(" KP2Surv pathway-node hypergraph construction")
    print(" - Each gene induces one pathway hyperedge")
    print(" - Offline files contain structure only")
    print(" - Fold-specific expression features are assembled during training")
    print("=" * 70)
    
    ap = argparse.ArgumentParser(description="Build the manuscript-aligned pathway hypergraph structure.")
    ap.add_argument("--kegg_csv", required=True, help="Wide-format KEGG pathway CSV.")
    ap.add_argument("--mrna_csv", required=True, help="Gene-by-sample expression CSV.")
    ap.add_argument("--wsi_hg_dir", required=True, help="Directory containing patient WSI hypergraphs.")
    ap.add_argument("--template_out", required=True, help="Output pathway hypergraph template .pt file.")
    ap.add_argument("--gene_hg_dir", required=True, help="Output directory for patient-matched pathway structures.")
    ap.add_argument("--mapping_csv", required=True, help="Output WSI/expression mapping CSV.")
    ap.add_argument("--gene_col", default="Gene_Symbol")
    ap.add_argument("--drop_meta_cols", type=int, default=1)
    ap.add_argument("--min_genes", type=int, default=10)
    ap.add_argument("--max_genes", type=int, default=400)
    ap.add_argument("--min_coverage", type=float, default=0.5)
    ap.add_argument("--min_pathways_per_gene", type=int, default=2)
    ap.add_argument("--max_pathways_per_gene", type=int, default=50)
    ap.add_argument("--expected_pathways", type=int, default=186, help="Expected retained pathway count; 0 disables the check.")
    args = ap.parse_args()
    
    print("\n[Step 1] Build the pathway-node hypergraph template")
    template, mrna_df, pathway_dict = build_pathway_node_hypergraph_template(
        kegg_csv_path=args.kegg_csv,
        mrna_csv_path=args.mrna_csv,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        min_coverage=args.min_coverage,
        min_pathways_per_gene=args.min_pathways_per_gene,
        max_pathways_per_gene=args.max_pathways_per_gene,
        gene_col=args.gene_col,
        drop_meta_cols=args.drop_meta_cols,
        expected_pathways=args.expected_pathways,
        save_path=args.template_out,
    )
    
    print("\n" + "-" * 50)
    print("Template summary:")
    print(f"  Pathway nodes: {template.num_nodes}")
    print(f"  Hyperedges: {template.num_hyperedges}")
    print(f"  hyperedge_index: {template.hyperedge_index.shape}")
    print("-" * 50)
    
    print("\n[Step 2] Create patient-matched pathway structures")
    create_patient_pathway_hypergraphs(
        template=template,
        mrna_df=mrna_df,
        wsi_hg_dir=args.wsi_hg_dir,
        save_dir=args.gene_hg_dir,
        mapping_save_path=args.mapping_csv,
    )
    
    print("\n" + "=" * 70)
    print(" Complete")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  Template: {args.template_out}")
    print(f"  Hypergraph directory: {args.gene_hg_dir}")
    print(f"  Mapping table: {args.mapping_csv}")


if __name__ == "__main__":
    main()
