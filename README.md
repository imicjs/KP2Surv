# KP2Surv

This directory contains the data-processing and training implementation aligned with the
revised `KP2Surv (4)` manuscript.

## Implemented formulation

- Pathology and pathway inputs are represented as hypergraphs.
- The pathology builder uses spatial neighborhoods with `k = 1, 2, 3`, a bin width of
  eight estimated grid steps, at most 300 spatial centers per scale, and the adaptive
  retry and retention limits described in the manuscript.
- Pathways are nodes. Genes shared by 2--50 retained pathways induce hyperedges. The
  manuscript defaults retain pathways with 10--400 measured genes and coverage at least
  0.5; the command-line pipeline checks for the reported 186 pathways by default.
- Expression is not standardized in the offline pathway-graph builder. Mean and standard
  deviation are fitted only on each training fold, after which pathway features are built
  as `pathway_gene_mask * standardized_expression`.
- Survival is modeled in continuous time with a censoring-aware log-normal likelihood.
  There are no discrete survival-time or hazard bins. Time grids in the source are used
  only for evaluation metrics.
- Cross-validation jointly stratifies event status and within-event observed-time
  quantiles whenever cohort size permits.
- When several files map to one case, pathology files use the lexicographically first
  filename. Expression samples prefer `-01A-`, then tumor samples, with a
  lexicographic tie-break at every selection step.
- Final distributional fusion follows Algorithm 1:

  `mu = 0.5 * (mu_dec + mu_fh)`

  `var = 0.25 * (var_dec_total + var_fh_total + 2 * cov_dec_fh)`

  The population covariance is estimated from paired stochastic passes. When stochastic
  prediction is disabled, its value is zero.

The generic training script reports the overall validation C-index. It intentionally does
not contain a GBM/LGG-specific macro-averaging branch; cohort-specific post-processing can
be performed from the exported patient-level predictions.

## Files

- `train.py`: repository-level training entry point.
- `kp2surv/common.py`: shared configuration, reproducibility, export, and split helpers.
- `kp2surv/data.py`: clinical loading, expression alignment, sampling, and datasets.
- `kp2surv/components.py`: hypergraph layers, attention, prediction heads, and fusion.
- `kp2surv/metrics.py`: censoring-aware losses, calibration metrics, and regularizers.
- `kp2surv/model.py`: complete KP2Surv forward model.
- `kp2surv/training.py`: training, evaluation, calibration, and prediction export.
- `kp2surv/cli.py`: command-line configuration and cross-validation orchestration.
- `scripts/`: WSI and pathway hypergraph preprocessing programs.

## Installation

Create a Python environment appropriate for the installed CUDA version, install PyTorch
first, and then install the project:

```bash
pip install -e .
```

For exact package pins, use `environment.yml` or `requirements.txt`. The reference
configuration records Python 3.10.0, PyTorch 2.5.1, CUDA 12.4, PyTorch Geometric 2.6.1,
and torch-scatter 2.1.2.

`torch-scatter` and `torch-geometric` wheels must match the installed PyTorch and CUDA
versions. Follow the PyTorch Geometric installation matrix if a normal pip installation
cannot find a compatible wheel.

## Data-processing pipeline

Build pathology hypergraphs:

```bash
python scripts/build_wsi_hypergraphs.py \
  --h5_dir /path/to/h5_features \
  --out_dir /path/to/wsi_hypergraphs
```

Build the pathway template and patient-matched structural files:

```bash
python scripts/build_pathway_hypergraph.py \
  --kegg_csv /path/to/kegg_wide_format.csv \
  --mrna_csv /path/to/tcga_mrna.csv \
  --wsi_hg_dir /path/to/wsi_hypergraphs \
  --template_out /path/to/pathway_node_template.pt \
  --gene_hg_dir /path/to/pathway_structures \
  --mapping_csv /path/to/pathway_node_mapping.csv
```

Create the expression table and pathway-gene mask from the same template:

```bash
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

After installation, the equivalent command starts with `kp2surv` instead of
`python train.py`.

The default configuration uses five folds, hidden dimension 256, dropout 0.35, AdamW
learning rate `2e-4`, batch size 4, 30 epochs, five stage-1 epochs, four stochastic passes
during stage 2, and eight stochastic passes during stage 3.

Cross-validation summaries report the arithmetic mean and sample standard deviation
across folds (`numpy.std(..., ddof=1)`).

## CPTAC external validation

External validation applies the five checkpoints trained for the corresponding TCGA
cancer type directly to CPTAC. It performs no retraining, fine-tuning, parameter update,
or checkpoint selection on CPTAC outcomes. Prepare CPTAC WSI hypergraphs with the same
pathology command and preprocessing settings used for TCGA, then run:

```bash
kp2surv-external \
  --cohort CPTAC \
  --clinical /path/to/cptac_clinical.csv \
  --wsi_dir /path/to/cptac_wsi_hypergraphs \
  --expr /path/to/cptac_case_by_gene_expression.csv \
  --mask /path/to/tcga_pathway_gene_mask.npz \
  --pathway_template /path/to/tcga_pathway_node_template.pt \
  --checkpoint_dir /path/to/tcga_checkpoints \
  --gene_mapping /path/to/optional_gene_mapping.csv \
  --out /path/to/cptac_external_results
```

Without installing the command-line entry point, run the same workflow with
`python -m kp2surv.external` followed by the same arguments.

The expression table is converted to each checkpoint's `gene_order`. Missing genes are
filled with zero before fold-specific TCGA standardization, and inference stops if more
than 10% of checkpoint genes are missing. Each checkpoint supplies its TCGA training-fold
mean and standard deviation, pathway order, model configuration, training stage,
stochastic-pass count, stochastic input-dropout rate, and fitted variance temperature.
CPTAC outcomes are used only to match patients and compute metrics after prediction.
Fold-specific predictions and alignment reports are saved separately; the final C-index
and secondary-metric summaries use the sample standard deviation across the five
checkpoints.

See `docs/external_validation.md` for input schemas, gene mapping, checkpoint contents,
and output files.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The tests cover sample-SD reporting, deterministic sample selection, expression and mask
alignment, checkpoint discovery, the Algorithm 1 variance identity, model forward
execution, and a small training step.

Input data and generated `.pt`, `.h5`, checkpoint, and spreadsheet files are excluded by
`.gitignore` to reduce the risk of committing patient data.
