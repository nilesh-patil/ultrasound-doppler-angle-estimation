"""Figure generation — one shared theme, web-ready SVG + PNG output.

Every figure is produced by a function here (never hand
edited) so the paper and the website can regenerate them from ``results/``. This
module starts with the paper's **Figure 2** (the augmentation sweep); Figures 4 & 5
(prediction-vs-actual and error-vs-angle) and the leakage-gap figure render from
``results/`` once it exists.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from uda.evaluation import agreement as _agreement  # noqa: E402
from uda.evaluation import conformal as _conformal  # noqa: E402
from uda.evaluation import tta as _tta  # noqa: E402
from uda.evaluation.calibration import conformal_calibration as _conformal_calibration  # noqa: E402
from uda.config import ExperimentConfig  # noqa: E402
from uda.data import augment as _augment  # noqa: E402
from uda.data import images as _images  # noqa: E402
from uda.data import labels as _labels  # noqa: E402

#: One shared style for every figure (consistent across paper + site).
THEME: dict = {
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 8,
    "axes.titlepad": 2.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "image.cmap": "gray",
}


def _apply_theme() -> None:
    plt.rcParams.update(THEME)


def save_figure(fig, stem: str, out_dir: str | Path = "results/figures") -> tuple[Path, Path]:
    """Write ``fig`` as both SVG (vector) and PNG (raster) under ``out_dir``.

    Returns the ``(svg_path, png_path)`` written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    svg = out / f"{stem}.svg"
    png = out / f"{stem}.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    return svg, png


def _labels_lookup(cfg: ExperimentConfig) -> dict[str, float]:
    """Map image_id -> base theta, building labels.csv from Results.txt if needed."""
    repo_root = Path(__file__).resolve().parents[2]
    results_txt = repo_root / "data" / "Results.txt"
    df = _labels.parse_results(results_txt)
    return dict(zip(df["image_id"], df["theta_deg"]))


def figure_2_augmentation(
    cfg: ExperimentConfig,
    image_id: str = "09-41-06_1",
    out_dir: str | Path = "results/figures",
) -> tuple[Path, Path]:
    """Reproduce the paper's Figure 2: one base image across the rotation sweep.

    Shows the full ``rotation_angles(cfg.data)`` sweep (default 25 frames) for a
    single base image as a grid, each panel titled with its rotation and the
    resulting (rotated) Doppler-angle label — making the augmentation + label
    arithmetic visible at a glance.
    """
    _apply_theme()

    repo_root = Path(__file__).resolve().parents[2]
    images_dir = repo_root / cfg.data.images_dir
    path = images_dir / f"{image_id}.jpg"
    gray = _images.load_image_gray(path)
    base_theta = _labels_lookup(cfg)[image_id]

    frames = list(_augment.augment_image(gray, base_theta, cfg.data))
    n = len(frames)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.9))
    axes = axes.ravel()
    for ax, (frame, new_theta, rot) in zip(axes, frames):
        ax.imshow(frame, vmin=0.0, vmax=1.0)
        ax.set_title(f"{rot:+d}° → θ={new_theta:.0f}°")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    paths = save_figure(fig, "figure2_augmentation", out_dir)
    plt.close(fig)
    return paths


_BACKBONE_ORDER = ["vgg19", "resnet50", "densenet201", "xception", "inceptionv3"]


def _load_predictions(name: str, pred_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(pred_dir) / f"{name}.csv")


def _grid(n: int):
    cols = min(n, 5) if n else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6), squeeze=False)
    return fig, axes.ravel()


def figure_4_pred_vs_actual(
    names: list[str],
    pred_dir: str | Path = "results/predictions",
    metrics_csv: str | Path = "results/metrics.csv",
    out_dir: str | Path = "results/figures",
    stem: str = "figure4_pred_vs_actual",
) -> tuple[Path, Path]:
    """Paper Figure 4: predicted-vs-actual scatter per backbone, R² annotated.

    Panels are ordered best-to-worst by R² and share one common axis range
    (the true-angle span), so the identity line and scatter spread are directly
    comparable across backbones. No in-figure suptitle — the LaTeX caption titles
    the figure.
    """
    _apply_theme()
    mtab = pd.read_csv(metrics_csv).set_index("name")
    # Load once; order panels best-to-worst by R² for a readable progression.
    loaded = [(name, _load_predictions(name, pred_dir)) for name in names]

    def _r2(name):
        return float(mtab.loc[name, "r2"]) if name in mtab.index else float("nan")

    loaded.sort(key=lambda t: _r2(t[0]), reverse=True)
    # One shared, square axis range over the true-angle span (predictions are
    # clipped to it) so every panel is on the same scale.
    tmin = min(float(df.theta_true.min()) for _, df in loaded)
    tmax = max(float(df.theta_true.max()) for _, df in loaded)
    pad = 0.04 * (tmax - tmin)
    axlo, axhi = tmin - pad, tmax + pad

    fig, axes = _grid(len(loaded))
    for ax, (name, df) in zip(axes, loaded):
        ax.plot([axlo, axhi], [axlo, axhi], "k--", lw=0.8, zorder=1)
        ax.scatter(df.theta_true, df.theta_pred, s=5, alpha=0.35, zorder=2)
        bb = mtab.loc[name, "backbone"] if name in mtab.index else name
        r2 = _r2(name)
        mae = mtab.loc[name, "mae"] if name in mtab.index else float("nan")
        ax.set_title(f"{bb}\nR²={r2:.3f}  MAE={mae:.2f}°")
        ax.set_xlabel("actual θ (°)")
        ax.set_ylabel("predicted θ (°)")
        ax.set_xlim(axlo, axhi)
        ax.set_ylim(axlo, axhi)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[len(loaded):]:
        ax.axis("off")
    fig.tight_layout()
    paths = save_figure(fig, stem, out_dir)
    plt.close(fig)
    return paths


def figure_5_error_vs_angle(
    names: list[str],
    pred_dir: str | Path = "results/predictions",
    metrics_csv: str | Path = "results/metrics.csv",
    out_dir: str | Path = "results/figures",
    stem: str = "figure5_error_vs_angle",
) -> tuple[Path, Path]:
    """Paper Figure 5: mean |error| vs observed angle (the 60–120° dip)."""
    _apply_theme()
    mtab = pd.read_csv(metrics_csv).set_index("name")
    bins = np.arange(0, 181, 15)
    centers = (bins[:-1] + bins[1:]) / 2.0
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axvspan(60, 120, color="0.92", zorder=0, label="60°–120° (clinical mid-range)")
    for name in names:
        df = _load_predictions(name, pred_dir)
        err = np.abs(df.theta_pred.to_numpy() - df.theta_true.to_numpy())
        idx = np.digitize(df.theta_true.to_numpy(), bins) - 1
        idx = np.clip(idx, 0, len(centers) - 1)
        mean_abs = [err[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(centers))]
        bb = mtab.loc[name, "backbone"] if name in mtab.index else name
        ax.plot(centers, mean_abs, marker="o", ms=3, lw=1.2, label=bb)
    ax.set_xlabel("observed Doppler angle θ (°)")
    ax.set_ylabel("mean |error| (°)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    paths = save_figure(fig, stem, out_dir)
    plt.close(fig)
    return paths


def figure_protocol_comparison(
    era_cv_csv: str | Path = "results/era2019_cv.csv",
    out_dir: str | Path = "results/figures",
    stem: str = "figure_protocol_comparison",
) -> tuple[Path, Path]:
    """Image-level vs patient-level MAE and MAPE per frozen backbone.

    Both protocols are reported as **5-fold cross-validation** means with $\\pm$1
    standard-deviation whiskers (rows ``cmp_<backbone>_{image,patient}`` in
    ``era2019_cv.csv``), so each bar is a robust estimate rather than a single
    high-variance split. Backbones are ordered by patient-level MAE. R² is not
    shown: on a single cross-subject fold it is dominated by variance and not a
    stable per-backbone summary.
    """
    _apply_theme()
    df = pd.read_csv(era_cv_csv)
    cmp = df[df["name"].astype(str).str.startswith("cmp_")].copy()
    if cmp.empty:
        raise ValueError(
            "no cmp_<backbone>_<protocol> rows in era2019_cv.csv; "
            "run the frozen-CV comparison step first"
        )

    def _row(backbone, protocol, col):
        sel = cmp[(cmp["backbone"] == backbone) & (cmp["protocol"] == protocol)]
        return float(sel[col].iloc[0]) if len(sel) else np.nan

    backbones = [b for b in _BACKBONE_ORDER if b in set(cmp["backbone"])]
    backbones.sort(key=lambda b: _row(b, "patient", "mae_mean"))
    x = np.arange(len(backbones))
    width = 0.38
    blue, orange = "#3b6ea5", "#dd8452"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, ylab in ((ax1, "mae", "MAE (°)"), (ax2, "mape", "MAPE (%)")):
        img = [_row(b, "image", f"{metric}_mean") for b in backbones]
        pat = [_row(b, "patient", f"{metric}_mean") for b in backbones]
        img_sd = [_row(b, "image", f"{metric}_std") for b in backbones]
        pat_sd = [_row(b, "patient", f"{metric}_std") for b in backbones]
        ax.bar(x - width / 2, img, width, yerr=img_sd, capsize=3,
               color=blue, label="image-level sampling")
        ax.bar(x + width / 2, pat, width, yerr=pat_sd, capsize=3,
               color=orange, label="patient-level sampling")
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels(backbones, rotation=30, ha="right")
        ax.legend(fontsize=7)
    fig.tight_layout()
    paths = save_figure(fig, stem, out_dir)
    plt.close(fig)
    return paths


def figure_architecture_bakeoff(
    era_cv_csv: str | Path = "results/era2019_cv.csv",
    out_dir: str | Path = "results/figures",
    stem: str = "figure_architecture_bakeoff",
) -> tuple[Path, Path]:
    """Frozen-backbone bake-off under patient 5-fold CV (newer ≠ better).

    Horizontal bars of patient-level 5-fold-CV MAPE (mean $\\pm$1 std) for every
    frozen backbone, coloured by family: the five classic ImageNet networks (the
    ``cmp_<backbone>_patient`` rows), ConvNeXt, and EfficientNet/EfficientNetV2 (the
    ``f_*`` / ``e2_*`` rows). All are frozen with the grid-pooling head, so the
    comparison is apples-to-apples. Lower is better; the classic family wins.
    """
    _apply_theme()
    df = pd.read_csv(era_cv_csv)

    def _family(row) -> str:
        bb = str(row["backbone"])
        if "convnext" in bb:
            return "ConvNeXt"
        if "efficientnet" in bb:
            return "EfficientNet / V2"
        return "classic ImageNet"

    # Classic = the frozen patient-CV comparison rows; modern = the f_*/e2_* rows.
    classic = df[df["name"].astype(str).str.fullmatch(r"cmp_.+_patient")].copy()
    modern = df[df["name"].astype(str).str.match(r"(f_|e2_)")].copy()
    modern = modern[modern["protocol"].isna() | (modern["protocol"] != "image")]
    rows = pd.concat([classic, modern], ignore_index=True)
    rows = rows.drop_duplicates("backbone")
    rows["family"] = rows.apply(_family, axis=1)
    rows["label"] = rows["backbone"]
    # Add DenseNet201's MATCHED era-2019 extraction (base_densenet201) so the
    # comparison against the era-2019 modern bars is visibly apples-to-apples, not
    # only disclosed in the caption.
    base = df[df["name"].astype(str) == "base_densenet201"].copy()
    if not base.empty:
        base = base.iloc[[0]].copy()
        base["family"] = "DenseNet201 (era-2019, matched)"
        base["label"] = "densenet201 (era-2019)"
        rows = pd.concat([rows, base[rows.columns]], ignore_index=True)
    rows = rows.sort_values("mape_mean", ascending=False)

    palette = {
        "classic ImageNet": "#3b6ea5",
        "DenseNet201 (era-2019, matched)": "#9ec1e0",
        "ConvNeXt": "#dd8452",
        "EfficientNet / V2": "#8c8c8c",
    }
    colors = [palette[f] for f in rows["family"]]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.barh(y, rows["mape_mean"], xerr=rows["mape_std"], capsize=3,
            color=colors, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=8)
    ax.set_xlabel("patient 5-fold CV MAPE (%)  — lower is better")
    # One legend entry per family. Placed upper-right (over the short, low-MAPE
    # bars) so it never overlaps the long bottom bars' error whiskers.
    from matplotlib.patches import Patch
    handles = [Patch(color=palette[k], label=k) for k in palette]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    paths = save_figure(fig, stem, out_dir)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Evaluation-rigor figures (Keras-free; reuse uda.{agreement,
# calibration,conformal,tta}). Each renders from saved predictions and returns
# the (svg, png) paths written by ``save_figure``.
# ---------------------------------------------------------------------------


def figure_bland_altman(
    pred_csv: str | Path,
    agg: str = "sample",
    out_dir: str | Path = "results/figures",
    stem: str = "figure_bland_altman",
) -> tuple[Path, Path]:
    """Bland–Altman agreement: signed (method − reference) vs the mean axis.

    Scatters the per-pair difference on the **method − reference** convention
    (``theta_pred`` − ``theta_true``, signed 180-wrap), so a *negative* bias means
    the model reads *lower* than the single MATLAB-GUI reference. Horizontal lines
    mark the bias and the 95% limits of agreement. Default ``agg="sample"`` reports
    the per-image statistics (n=2100) that the headline table and prose quote; this
    is *method vs single reference* agreement, **not** inter-observer. All
    statistics come from :func:`uda.evaluation.agreement.agreement_from_csv`, whose
    native convention is reference − method; we negate to method − reference here
    so the sign matches "the model reads lower".
    """
    _apply_theme()
    out = _agreement.agreement_from_csv(pred_csv, agg=agg)
    mean_axis = np.asarray(out["mean_axis"], dtype=float)
    # agreement_from_csv returns reference - method; negate to method - reference.
    diff = -np.asarray(out["diff"], dtype=float)
    bias = -out["bias"]
    lo = -out["loa_upper"]
    hi = -out["loa_lower"]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.scatter(mean_axis, diff, s=10, alpha=0.45, zorder=2, color="#3b6ea5")
    ax.axhline(bias, color="0.15", lw=1.4, zorder=3, label=f"bias = {bias:+.2f}°")
    ax.axhline(
        hi, color="#b5403a", lw=1.0, ls="--", zorder=3,
        label=f"+1.96 SD = {hi:+.2f}°",
    )
    ax.axhline(
        lo, color="#b5403a", lw=1.0, ls="--", zorder=3,
        label=f"−1.96 SD = {lo:+.2f}°",
    )
    ax.axhline(0.0, color="0.7", lw=0.6, zorder=1)
    ax.set_xlabel("mean of reference & method  (θ°, Bland–Altman axis)")
    ax.set_ylabel("method − reference  (Δθ°, signed 180-wrap)")
    _ntitle = (
        f"n = {out['n']} patient-level OOF predictions"
        if agg == "sample"
        else f"n = {out['n']} ({agg}-aggregated)"
    )
    ax.legend(fontsize=7, loc="upper right", title=_ntitle)
    fig.tight_layout()
    paths = save_figure(fig, stem, out_dir)
    plt.close(fig)
    return paths


def figure_calibration(
    pred_csv: str | Path,
    out_dir: str | Path = "results/figures",
    stem: str = "figure_calibration",
    alphas=None,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Conformal calibration: nominal vs empirical coverage (patient-disjoint).

    Builds a **patient-disjoint** calibration/test residual split exactly as
    :func:`uda.evaluation.conformal.evaluate_conformal` does (the first fold of a 2-fold
    :class:`PatientLevelSplit` over ``patient_id``, signed-wrapped residuals), then
    sweeps an ``alpha`` grid through
    :func:`uda.evaluation.calibration.conformal_calibration`. Plots empirical vs nominal
    coverage against the ``y=x`` diagonal; valid split-conformal intervals sit on
    or **above** the diagonal (empirical ``>=`` nominal).
    """
    _apply_theme()
    if alphas is None:
        alphas = np.array([0.05, 0.1, 0.2, 0.3])
    alphas = np.asarray(alphas, dtype=float)

    # Patient-disjoint cal/test partition — identical to ``evaluate_conformal``.
    from uda.config import SplitConfig
    from uda.data.splits import PatientLevelSplit

    df = pd.read_csv(pred_csv)
    labels = df.drop_duplicates(subset="image_id")[["image_id", "patient_id"]]
    cfg = SplitConfig(strategy="patient", n_folds=2, seed=seed)
    cal_ids, test_ids = next(PatientLevelSplit().split(labels, cfg))
    cal_rows = df[df["image_id"].isin(set(cal_ids))]
    test_rows = df[df["image_id"].isin(set(test_ids))]

    cal_resid = _conformal._signed_wrap(
        cal_rows["theta_true"].to_numpy(dtype=float)
        - cal_rows["theta_pred"].to_numpy(dtype=float)
    )
    test_resid = _conformal._signed_wrap(
        test_rows["theta_true"].to_numpy(dtype=float)
        - test_rows["theta_pred"].to_numpy(dtype=float)
    )

    curve = _conformal_calibration(cal_resid, test_resid, alphas=alphas)
    nominal = curve["nominal"]
    empirical = curve["empirical"]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, zorder=1, label="y = x (perfect)")
    ax.plot(
        nominal, empirical, marker="o", ms=5, lw=1.4, color="#3b6ea5", zorder=3,
        label="split-conformal (patient-disjoint)",
    )
    for nom, emp in zip(nominal, empirical):
        ax.annotate(
            f"{emp:.3f}", (nom, emp), textcoords="offset points",
            xytext=(6, -8), fontsize=6, color="0.3",
        )
    # Zoom to the data range (plus a margin) so the small over-coverage gap above
    # the diagonal is visible, instead of fixing the axes to the empty 0..1 box.
    lo = max(0.0, float(min(nominal.min(), empirical.min())) - 0.05)
    hi = min(1.0, float(max(nominal.max(), empirical.max())) + 0.03)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("nominal coverage  (1 − α)")
    ax.set_ylabel("empirical coverage  (held-out test)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    paths = save_figure(fig, stem, out_dir)
    plt.close(fig)
    return paths


