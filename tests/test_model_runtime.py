import sys

import numpy as np
import torch
from torch_geometric.data import Batch

from kp2surv.cli import parse_args
from kp2surv.common import HypergraphData
from kp2surv.data import survival_collate
from kp2surv.model import KP2Surv
from kp2surv.training import train_epoch


def synthetic_sample():
    wsi = HypergraphData(
        x=torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                [0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            ],
            dtype=torch.float32,
        ),
        hyperedge_index=torch.tensor(
            [[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long
        ),
        num_nodes=4,
        num_hyperedges=2,
    )
    wsi.hyperedge_type = torch.tensor([0, 1], dtype=torch.long)
    gene = HypergraphData(
        hyperedge_index=torch.tensor(
            [[0, 1, 1, 2], [0, 0, 1, 1]], dtype=torch.long
        ),
        num_nodes=3,
        num_hyperedges=2,
    )
    return {
        "wsi": wsi,
        "gene_tpl": gene,
        "expr": torch.tensor([0.2, -0.1, 0.5, 0.3], dtype=torch.float32),
        "time": torch.tensor(12.0),
        "event": torch.tensor(1.0),
        "case_id": "CASE-01-A",
        "wsi_file": "CASE-01-A.pt",
        "gene_file": "CASE-01-A.pt",
    }


def small_model():
    return KP2Surv(
        wsi_in_dim=6,
        gene_in_dim=4,
        hidden_dim=8,
        num_pathways=3,
        dropout=0.0,
        cross_num_heads=2,
        stochastic_input_dropout=0.1,
    )


def test_model_forward_shapes_and_algorithm1_outputs():
    sample = synthetic_sample()
    model = small_model().eval()
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )

    with torch.no_grad():
        output = model.forward_batch(
            Batch.from_data_list([sample["wsi"]]),
            Batch.from_data_list([sample["gene_tpl"]]),
            sample["expr"].unsqueeze(0),
            mask,
            stochastic_passes=3,
            enable_stochastic_variance=True,
            enable_disagreement=True,
            stochastic_latent=False,
            compute_ctr_loss=False,
            compute_aux_losses=False,
        )

    assert output["mu"].shape == (1, 1)
    assert output["sigma"].shape == (1, 1)
    assert output["branch_disagreement"].shape == (1, 1)
    assert torch.isfinite(output["mu"]).all()
    assert torch.isfinite(output["sigma"]).all()
    assert (output["sigma"] > 0).all()


def test_small_training_step_runs_end_to_end():
    batch = survival_collate([synthetic_sample()])
    model = small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )

    metrics = train_epoch(
        model=model,
        loader=[batch],
        optimizer=optimizer,
        scaler=None,
        mu_g=torch.zeros(4),
        sd_g=torch.ones(4),
        mask=mask,
        device=torch.device("cpu"),
        lambda_ctr=0.0,
        stochastic_passes=1,
        enable_stochastic_variance=False,
        enable_disagreement=True,
        stochastic_latent=False,
        use_amp=False,
    )

    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["nll_fuse"])


def test_training_cli_exposes_revised_stochastic_terms(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kp2surv",
            "--clinical", "clinical.csv",
            "--wsi_dir", "wsi",
            "--gene_dir", "gene",
            "--expr", "expression.csv",
            "--mask", "mask.npz",
            "--out", "output",
        ],
    )

    args = parse_args()

    assert args.stochastic_passes_stage2 == 4
    assert args.stochastic_passes_stage3 == 8
    assert args.stochastic_input_dropout == 0.20
