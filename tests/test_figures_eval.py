"""Evaluation-rigor figure smoke tests — each writes a non-empty SVG + PNG (synthetic).

Mirrors ``tests/test_figures.py``: small synthetic inputs in ``tmp_path``, assert
both files exist and are non-empty, two paths returned, and rendering is
deterministic (identical bytes across two runs). No Keras backend, matplotlib Agg.
"""
import numpy as np
import pandas as pd

from uda import figures


def _oof_csv(path, *, n_patients=12, n_rot=5, seed=0):
    """Write a small OOF-schema CSV: image_id, patient_id, rotation_deg, theta_*.

    One image per patient, ``n_rot`` rotations each; ``theta_true = base + rot``
    and ``theta_pred`` is the base estimate (constant per image) plus the rotation
    plus small noise — the rotation-augmented schema the evaluation modules expect.
    """
    rng = np.random.default_rng(seed)
    rots = np.linspace(-40, 40, n_rot)
    rows = []
    for p in range(n_patients):
        base = float(rng.uniform(30, 150))
        est = base + float(rng.normal(0, 4))  # per-image base estimate
        img = f"img_{p:02d}"
        for r in rots:
            rows.append(
                {
                    "image_id": img,
                    "patient_id": p,
                    "rotation_deg": float(r),
                    "theta_true": base + r,
                    "theta_pred": est + r + float(rng.normal(0, 2)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def _both_nonempty(svg, png):
    return (
        svg.exists()
        and png.exists()
        and svg.stat().st_size > 0
        and png.stat().st_size > 0
    )


def test_bland_altman_writes_svg_and_png(tmp_path):
    pred = tmp_path / "oof.csv"
    _oof_csv(pred)
    svg, png = figures.figure_bland_altman(
        pred, agg="patient", out_dir=str(tmp_path / "figs")
    )
    assert _both_nonempty(svg, png)
    assert svg.suffix == ".svg" and png.suffix == ".png"


def test_bland_altman_sample_agg(tmp_path):
    pred = tmp_path / "oof.csv"
    _oof_csv(pred)
    out = figures.figure_bland_altman(
        pred, agg="sample", out_dir=str(tmp_path / "figs")
    )
    assert len(out) == 2 and _both_nonempty(*out)


def test_calibration_writes_svg_and_png(tmp_path):
    pred = tmp_path / "oof.csv"
    _oof_csv(pred, n_patients=20)
    svg, png = figures.figure_calibration(pred, out_dir=str(tmp_path / "figs"))
    assert _both_nonempty(svg, png)
    assert svg.suffix == ".svg" and png.suffix == ".png"


def test_protocol_comparison_writes_svg_and_png(tmp_path):
    # Synthesise the era2019_cv.csv rows the figure reads (cmp_<bb>_<protocol>),
    # so the smoke test never touches results/ on disk.
    era = tmp_path / "era2019_cv.csv"
    rows = []
    for bb in ("densenet201", "resnet50"):
        for protocol, base in (("patient", 13.0), ("image", 5.0)):
            rows.append({
                "name": f"cmp_{bb}_{protocol}", "backbone": bb, "k": 5,
                "mae_mean": base * 0.7, "mae_std": 1.5,
                "mape_mean": base, "mape_std": 2.0, "protocol": protocol,
            })
    pd.DataFrame(rows).to_csv(era, index=False)

    svg, png = figures.figure_protocol_comparison(str(era), out_dir=str(tmp_path / "figs"))
    assert _both_nonempty(svg, png)
    assert svg.suffix == ".svg" and png.suffix == ".png"


def test_figures_return_two_paths(tmp_path):
    pred = tmp_path / "oof.csv"
    _oof_csv(pred, n_patients=20)
    for out in (
        figures.figure_bland_altman(pred, out_dir=str(tmp_path / "f1")),
        figures.figure_calibration(pred, out_dir=str(tmp_path / "f2")),
    ):
        assert isinstance(out, tuple) and len(out) == 2


def test_bland_altman_deterministic(tmp_path):
    pred = tmp_path / "oof.csv"
    _oof_csv(pred)
    svg1, png1 = figures.figure_bland_altman(pred, out_dir=str(tmp_path / "a"))
    svg2, png2 = figures.figure_bland_altman(pred, out_dir=str(tmp_path / "b"))
    # The rendered figure is a pure function of the input: the PNG raster is
    # byte-identical across runs. (The SVG differs only in matplotlib's embedded
    # wall-clock <dc:date> and per-run random clip-path element ids, neither of
    # which is figure content — so determinism is asserted on the PNG bytes.)
    assert png1.read_bytes() == png2.read_bytes()
    assert svg1.read_text().startswith("<?xml") and svg1.stat().st_size > 0
