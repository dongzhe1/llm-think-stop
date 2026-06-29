"""Regression test: verify entropy NaN fix for masked/special tokens.

LLM logit tensors contain -inf for masked tokens. Without .nan_to_num(),
exp(-inf)*(-inf) = 0*(-inf) = NaN, which propagates through conf_ent into
the feature matrix X and crashes sklearn LogisticRegression.

Scenarios:
    1. Entropy NaN from masked tokens
    2. Pre-sklearn X guard (nan_to_num before StandardScaler)
    3. Full grouped-CV pipeline with NaN entropy
"""
import logging

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

rng = np.random.default_rng(0)
passed = 0


def test_entropy_nan():
    global passed
    logger.info("1. Entropy NaN from masked tokens")
    vocab = 32000
    logits = torch.tensor(rng.standard_normal((4, vocab)).astype(np.float32))
    logits[:, 5000:] = float("-inf")

    lp = torch.log_softmax(logits, dim=-1)
    ent_bad = -(lp.exp() * lp).sum(-1)
    ent_good = -(lp.exp() * lp.nan_to_num(0.0)).sum(-1)

    assert torch.isnan(ent_bad).any(), "Bug not triggered — test setup issue"
    assert not torch.isnan(ent_good).any(), "Fix did not work"
    logger.info("  without fix: NaN=%s", torch.isnan(ent_bad).any().item())
    logger.info("  with fix:    NaN=%s", torch.isnan(ent_good).any().item())
    passed += 1


def test_x_guard():
    global passed
    logger.info("2. NaN in feature matrix X crashes LogisticRegression")

    n, m = 40, 10
    x = rng.standard_normal((n * m, 10)).astype(np.float32)
    x[5, 3] = float("nan")
    y = rng.integers(0, 2, n * m)
    groups = np.repeat(np.arange(n), m)

    try:
        sc = StandardScaler().fit(x)
        LogisticRegression(max_iter=200).fit(sc.transform(x), y)
        logger.info("  UNEXPECTED: no error without fix")
    except ValueError as e:
        logger.info("  Without fix -> ValueError: %s", str(e)[:60])

    x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(x_clean)
    LogisticRegression(max_iter=200).fit(sc.transform(x_clean), y)
    logger.info("  With nan_to_num fix -> OK")
    passed += 1


def test_pipeline():
    global passed
    logger.info("3. Full grouped-CV pipeline with entropy NaN")

    n, m = 60, 10
    conf_ent_raw = torch.tensor(rng.standard_normal((n, m)).astype(np.float32))
    conf_ent_raw[::5, :] = float("nan")

    try:
        conf_ent = conf_ent_raw.numpy()
        x_bad = rng.standard_normal((n * m, 10)).astype(np.float32)
        x_bad[:, 3] = conf_ent.reshape(-1)
        y = rng.integers(0, 2, n * m)
        groups = np.repeat(np.arange(n), m)
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(x_bad, y, groups):
            sc = StandardScaler().fit(x_bad[tr])
            LogisticRegression(max_iter=200).fit(sc.transform(x_bad[tr]), y[tr])
        logger.info("  UNEXPECTED: no error without fix")
    except ValueError as e:
        logger.info("  Without fix -> ValueError: %s", str(e)[:60])

    conf_ent = conf_ent_raw.nan_to_num(0.0).numpy()
    x_good = rng.standard_normal((n * m, 10)).astype(np.float32)
    x_good[:, 3] = conf_ent.reshape(-1)
    x_good = np.nan_to_num(x_good, nan=0.0, posinf=0.0, neginf=0.0)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(x_good, y, groups):
        sc = StandardScaler().fit(x_good[tr])
        x_tr = np.nan_to_num(sc.transform(x_good[tr]), nan=0.0, posinf=0.0, neginf=0.0)
        x_te = np.nan_to_num(sc.transform(x_good[te]), nan=0.0, posinf=0.0, neginf=0.0)
        LogisticRegression(max_iter=200).fit(x_tr, y[tr])
    logger.info("  With all fixes -> OK")
    passed += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_entropy_nan()
    test_x_guard()
    test_pipeline()
    logger.info("\n%d/3 tests passed.", passed)
