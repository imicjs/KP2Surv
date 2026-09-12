import numpy as np
import pandas as pd
import pytest
import torch

from kp2surv.data import PathwayGeneMask
from kp2surv.external import (
    align_external_expression,
    align_mask_to_checkpoint,
    discover_checkpoints,
)


def test_external_expression_alignment_fills_before_standardization():
    expression = pd.DataFrame(
        {"G1": [4.0], "G3": [8.0]},
        index=["CASE-01-A"],
    )

    aligned, report = align_external_expression(
        expression,
        ["G1", "G2", "G3"],
        max_missing_fraction=0.34,
    )

    assert aligned.columns.tolist() == ["G1", "G2", "G3"]
    np.testing.assert_array_equal(aligned.iloc[0].to_numpy(), [4.0, 0.0, 8.0])
    assert report["missing_genes"] == ["G2"]


def test_external_expression_alignment_enforces_missing_threshold():
    expression = pd.DataFrame({"G1": [4.0]}, index=["CASE-01-A"])

    with pytest.raises(ValueError, match="exceeds threshold"):
        align_external_expression(
            expression,
            ["G1", "G2", "G3"],
            max_missing_fraction=0.10,
        )


def test_external_mask_alignment_uses_checkpoint_orders():
    mask_obj = PathwayGeneMask(
        mask=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        pathway_names=["P2", "P1"],
        gene_names=["G2", "G1"],
    )

    aligned = align_mask_to_checkpoint(
        mask_obj,
        gene_order=["G1", "G2", "G3"],
        checkpoint_pathways=["P1", "P2"],
        device=torch.device("cpu"),
    )

    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert torch.equal(aligned, expected)


def test_checkpoint_discovery_is_numeric_and_requires_five(tmp_path):
    for fold in [4, 1, 3, 0, 2]:
        (tmp_path / f"fold{fold}_best.pt").touch()

    paths = discover_checkpoints(str(tmp_path))

    assert [path.split("fold")[-1].split("_")[0] for path in paths] == [
        "0", "1", "2", "3", "4"
    ]
