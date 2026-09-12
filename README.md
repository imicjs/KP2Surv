# KP2Surv

KP2Surv is a multimodal survival model that combines pathology and pathway
hypergraphs with uncertainty-aware fusion.

## Overview

- Continuous-time survival modeling with a censoring-aware log-normal likelihood.
- Fold-specific expression normalization and survival-stratified cross-validation.
- Deterministic WSI and expression-sample selection.
- Five-fold results reported as mean and sample standard deviation (`ddof=1`).

## Installation

```bash
conda env create -f environment.yml
conda activate kp2surv
```

Alternatively, install the pinned dependencies from `requirements.txt` and then run
`pip install -e .`. PyTorch Geometric wheels must match the installed PyTorch and CUDA
versions.

## Data preparation

```bash
python scripts/build_wsi_hypergraphs.py \
  --h5_dir /path/to/h5_features \
  --out_dir /path/to/wsi_hypergraphs

python scripts/build_pathway_hypergraph.py \
  --kegg_csv /path/to/kegg_wide_format.csv \
  --mrna_csv /path/to/tcga_mrna.csv \
  --wsi_hg_dir /path/to/wsi_hypergraphs \
  --template_out /path/to/pathway_node_template.pt \
  --gene_hg_dir /path/to/pathway_structures \
  --mapping_csv /path/to/pathway_node_mapping.csv

python scripts/make_expression_and_mask.py \
  --mrna_csv /path/to/tcga_mrna.csv \
  --template_pt /path/to/pathway_node_template.pt \
  --out_expr /path/to/gene_expression.csv \
  --out_mask /path/to/pathway_gene_mask.npz
```

## Training

```bash
python train.py \
  --clinical /path/to/clinical.xlsx \
  --wsi_dir /path/to/wsi_hypergraphs \
  --gene_dir /path/to/pathway_structures \
  --expr /path/to/gene_expression.csv \
  --mask /path/to/pathway_gene_mask.npz \
  --out /path/to/output
```

Use `python train.py --help` for the complete option list.

## CPTAC external validation

CPTAC evaluation directly applies the five checkpoints trained for the corresponding
TCGA cancer type without retraining or fine-tuning.

```bash
kp2surv-external \
  --cohort CPTAC \
  --clinical /path/to/cptac_clinical.csv \
  --wsi_dir /path/to/cptac_wsi_hypergraphs \
  --expr /path/to/cptac_expression.csv \
  --mask /path/to/tcga_pathway_gene_mask.npz \
  --pathway_template /path/to/tcga_pathway_node_template.pt \
  --checkpoint_dir /path/to/tcga_checkpoints \
  --out /path/to/cptac_results
```

See `docs/external_validation.md` for gene alignment, missing-gene handling, checkpoint
requirements, and output files.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Patient data and generated model artifacts are not included in this repository.
