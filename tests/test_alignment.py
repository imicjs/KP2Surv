import numpy as np
import torch

from kp2surv.common import build_survival_strata, sample_mean_and_std
from kp2surv.metrics import combine_correlated_estimator_variances


def test_joint_event_time_strata_populate_every_fold():
    samples = [
        {"time": float(i + 1 + 0.25 * event), "event": event}
        for event in (0, 1)
        for i in range(60)
    ]
    labels, description = build_survival_strata(samples, n_splits=5, max_time_bins=4)

    assert "event_status" in description
    assert np.bincount(labels).min() >= 5


def test_algorithm1_correlated_variance_identity():
    rng = np.random.default_rng(42)
    mu_dec = rng.normal(size=(1000, 8))
    mu_fh = 0.6 * mu_dec + rng.normal(size=(1000, 8))
    var_stoch_dec = np.var(mu_dec, axis=0)
    var_stoch_fh = np.var(mu_fh, axis=0)
    covariance = np.mean(
        (mu_dec - mu_dec.mean(axis=0)) * (mu_fh - mu_fh.mean(axis=0)),
        axis=0,
    )

    expected = np.var(0.5 * (mu_dec + mu_fh), axis=0)
    actual = combine_correlated_estimator_variances(
        torch.zeros(8),
        torch.zeros(8),
        torch.tensor(var_stoch_dec),
        torch.tensor(var_stoch_fh),
        torch.tensor(covariance),
    ).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_fold_summary_uses_sample_standard_deviation():
    values = np.array([0.61, 0.64, 0.66, 0.67, 0.70])
    mean, standard_deviation = sample_mean_and_std(values)

    assert mean == np.mean(values)
    assert standard_deviation == np.std(values, ddof=1)
    assert standard_deviation != np.std(values, ddof=0)
