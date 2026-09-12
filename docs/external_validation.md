# External validation with TCGA checkpoints

## Protocol

External validation uses the five fold-specific checkpoints trained for the matching
TCGA cancer type. A checkpoint is evaluated independently on CPTAC without retraining,
fine-tuning, optimizer steps, or external-data-based checkpoint selection.

For each fold, the evaluator:

1. Restores the model weights, architecture settings, training stage, stochastic-pass
   count, stochastic input-dropout rate, weak-prior configuration, and fitted variance
   temperature from the TCGA checkpoint.
2. Maps CPTAC gene identifiers into the checkpoint gene space when a mapping file is
   supplied.
3. Reorders expression columns to the saved `gene_order`, inserts zero for missing genes
   before standardization, and rejects the input when the missing fraction exceeds 10%.
4. Standardizes expression with that fold's saved TCGA training mean and standard
   deviation. CPTAC-derived normalization statistics are never estimated.
5. Aligns the pathway-gene mask to the same checkpoint gene and pathway order.
6. Uses pathology hypergraphs produced by the same WSI preprocessing program and uses
   the online WSI sampling settings restored from the checkpoint.
7. Uses CPTAC clinical outcomes only for case matching and post-prediction performance
   evaluation.

## Required inputs

- `--clinical`: CSV, TSV, XLS, or XLSX table accepted by `ClinicalDataLoader`. It must
  contain a patient identifier, vital status, and survival-time columns.
- `--wsi_dir`: CPTAC pathology hypergraphs created with
  `scripts/build_wsi_hypergraphs.py` and the same settings used for TCGA.
- `--expr`: numeric case-by-gene CSV on the same expression scale used for TCGA; the
  first column is the case identifier.
- `--mask`: pathway-gene mask used by the TCGA training run. An NPZ mask must include
  `mask`, `pathway_names`, and `gene_names`.
- `--checkpoint_dir`: directory containing `fold0_best.pt` through `fold4_best.pt` by
  default.
- Either `--gene_dir` with prebuilt CPTAC pathway-structure files or
  `--pathway_template` with the TCGA pathway template. When the template is supplied,
  structure-only CPTAC files are created in the output directory.

An optional `--gene_mapping` CSV must contain `source_gene` and `target_gene` columns.
When several source genes map to the same target, their expression values are averaged.

## Command

```bash
kp2surv-external \
  --cohort CPTAC \
  --clinical /data/cptac/clinical.csv \
  --wsi_dir /data/cptac/wsi_hypergraphs \
  --expr /data/cptac/expression.csv \
  --mask /data/tcga/pathway_gene_mask.npz \
  --pathway_template /data/tcga/pathway_node_template.pt \
  --checkpoint_dir /results/tcga/checkpoints \
  --out /results/cptac_external
```

The equivalent module command is `python -m kp2surv.external` followed by the same
arguments.

The default checkpoint pattern is `fold*_best.pt`, the expected count is five, and the
maximum missing-gene fraction is `0.10`. These can be changed with
`--checkpoint_pattern`, `--expected_checkpoints`, and
`--max_missing_gene_fraction` when documenting a different protocol.

## Checkpoint contract

New checkpoints explicitly save the following inference state:

- `model_state_dict`, `fold`, `epoch`, and `stage`;
- `args`, including architecture and online WSI sampling settings;
- `gene_order`, `pathway_names`, `train_gene_mean`, and `train_gene_std`;
- `prior_table` and `sigma_temperature`.

The external evaluator also accepts checkpoints produced by the immediately preceding
repository layout by translating their saved argument keys. A checkpoint without gene
order or fold-specific normalization vectors is rejected because leakage-free external
inference cannot be guaranteed.

## Outputs

- `foldN_CPTAC_predictions.csv`: patient-level predictions from one TCGA fold.
- `foldN_CPTAC_alignment.json`: missing-gene audit for that fold.
- `external_metrics_by_checkpoint.csv`: checkpoint-level metrics and restored settings.
- `external_summary.json`: means and sample standard deviations (`ddof=1`) for all
  reported metrics, plus explicit flags confirming that no retraining, fine-tuning, or
  external checkpoint selection was performed.
