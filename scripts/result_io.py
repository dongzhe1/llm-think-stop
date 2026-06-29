"""Result persistence helpers for the reasoning-budget probes.

Each probe writes a self-contained run directory so all downstream analysis
(bootstrap CIs, baselines, ablations, conformal) runs offline on CPU from
the saved arrays. Layout of a run dir::

    results/<probe>/<tag>/
        meta.json          Run configuration
        raw.npz            Per-question matrices (correct, think_lens, p_stop, X, y, ...)
        frontier_fixed.csv     Budget, accuracy, avg_tokens
        frontier_adaptive.csv  Threshold, accuracy, avg_tokens, adapt_gain
        conformal.csv          Alpha, tau, risk, saving_pct
        summary.json       Headline scalars
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def make_run_dir(out_dir: str, data_name: str, model: str, n: int) -> Path:
    model_tag = os.path.basename(model.rstrip("/")) or "model"
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{data_name}_{model_tag}_n{n}_{stamp}"
    run = Path(out_dir) / tag
    run.mkdir(parents=True, exist_ok=True)
    return run


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _str_matrix(ans) -> np.ndarray:
    ans = np.asarray(ans, dtype=object)
    return np.array([[str(x) for x in row] for row in ans])


def dump_raw(path: Path, *, budgets, gold, **arrays) -> None:
    payload = {
        "budgets": np.asarray(budgets, dtype=int),
        "gold": np.array([str(g) for g in gold]),
    }
    for k, v in arrays.items():
        if v is None:
            continue
        v = np.asarray(v, dtype=object) if v.dtype == object else np.asarray(v)
        if getattr(v, "dtype", None) == object:
            v = _str_matrix(v)
        payload[k] = v
    np.savez_compressed(path, **payload)
