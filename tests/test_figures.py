"""Tests — figures write both SVG and PNG (synthetic predictions)."""
import numpy as np
import pandas as pd

from uda import figures

_BACKBONES = ["vgg19", "resnet50"]


def _setup(tmp_path):
    rng = np.random.default_rng(0)
    pred = tmp_path / "predictions"
    pred.mkdir()
    names = []
    metric_rows = []
    for strat in ("augmented", "image", "patient"):
        for bb in _BACKBONES:
            name = f"replication_{bb}_{strat}"
            n = 200
            true = rng.uniform(20, 160, n)
            pd.DataFrame(
                {
                    "image_id": ["a"] * n,
                    "patient_id": [0] * n,
                    "rotation_deg": [0] * n,
                    "split": ["test"] * n,
                    "theta_true": true,
                    "theta_pred": true + rng.normal(0, 3, n),
                }
            ).to_csv(pred / f"{name}.csv", index=False)
            if strat == "augmented":
                names.append(name)
            metric_rows.append(
                {
                    "name": name,
                    "backbone": bb,
                    "split_strategy": strat,
                    "target": "raw",
                    "seed": 42,
                    "era": "replication",
                    "n_test": n,
                    "mae": 3.0,
                    "rmse": 3.9,
                    "me": 0.0,
                    "mape": 4.0,
                    "r2": 0.98,
                }
            )
    mcsv = tmp_path / "metrics.csv"
    pd.DataFrame(metric_rows).to_csv(mcsv, index=False)
    return names, str(pred), str(mcsv)


def test_figure_4_writes_svg_and_png(tmp_path):
    names, pred, mcsv = _setup(tmp_path)
    svg, png = figures.figure_4_pred_vs_actual(names, pred, mcsv, str(tmp_path / "figs"))
    assert svg.exists() and png.exists()


def test_figure_5_writes_svg_and_png(tmp_path):
    names, pred, mcsv = _setup(tmp_path)
    svg, png = figures.figure_5_error_vs_angle(names, pred, mcsv, str(tmp_path / "figs"))
    assert svg.exists() and png.exists()


def _era_cv_csv(tmp_path):
    """Synthetic era2019_cv.csv with cmp_<bb>_<protocol> + modern frozen rows."""
    rows = []
    for bb in ("densenet201", "resnet50", "vgg19", "xception", "inceptionv3"):
        for protocol, base in (("patient", 13.0), ("image", 5.0)):
            rows.append({
                "name": f"cmp_{bb}_{protocol}", "backbone": bb, "pooling": "grid2",
                "target": "raw", "richer_aug": "off", "era": "replication", "k": 5,
                "mae_mean": base * 0.7, "mae_std": 1.5, "rmse_mean": base, "rmse_std": 2.0,
                "me_mean": 0.0, "me_std": 1.0, "mape_mean": base, "mape_std": 2.0,
                "r2_mean": 0.9, "r2_std": 0.05, "protocol": protocol,
            })
    for bb in ("convnext_tiny", "efficientnetb0"):
        rows.append({
            "name": f"f_{bb}", "backbone": bb, "pooling": "grid2", "target": "raw",
            "richer_aug": "off", "era": "modern", "k": 5, "mae_mean": 12.0, "mae_std": 2.0,
            "rmse_mean": 16.0, "rmse_std": 3.0, "me_mean": -3.0, "me_std": 5.0,
            "mape_mean": 17.0, "mape_std": 3.0, "r2_mean": 0.78, "r2_std": 0.1,
            "protocol": float("nan"),
        })
    p = tmp_path / "era2019_cv.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_protocol_comparison_writes_svg_and_png(tmp_path):
    era = _era_cv_csv(tmp_path)
    svg, png = figures.figure_protocol_comparison(era, str(tmp_path / "figs"))
    assert svg.exists() and png.exists()


def test_architecture_bakeoff_writes_svg_and_png(tmp_path):
    era = _era_cv_csv(tmp_path)
    svg, png = figures.figure_architecture_bakeoff(era, str(tmp_path / "figs"))
    assert svg.exists() and png.exists()
