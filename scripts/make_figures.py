"""Regenerate every paper/site figure from ``results/`` (reproducible, Keras-free).

This is the single entry point behind the paper's claim that all figures are
generated from code. It rebuilds, into ``results/figures/`` (SVG + PNG):

  figure2_augmentation          one base image across the rotation sweep
  figure4_pred_vs_actual        predicted-vs-actual scatter, shared axes
  figure5_error_vs_angle        |error| vs observed angle (the 60-120 deg dip)
  figure_protocol_comparison    image- vs patient-level MAE/MAPE, 5-fold CV +-std
  figure_architecture_bakeoff   frozen-backbone bake-off, patient 5-fold CV +-std
  figure_tuning_history         Optuna best-so-far objective per backbone
  figure_bland_altman           method-minus-reference agreement (per-image)
  figure_calibration            conformal nominal-vs-empirical coverage

The Grad-CAM overlay (``gradcam_densenet201_*.png``) needs the TensorFlow backend
and is produced separately by the deploy/Grad-CAM path; it is not rebuilt
here. Run:  ``pixi run python scripts/make_figures.py``
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
PRED = ROOT / "results" / "predictions"
TUN = ROOT / "results" / "tuning"
ERA_CV = ROOT / "results" / "era2019_cv.csv"
METRICS = ROOT / "results" / "metrics.csv"
OOF = PRED / "tuned_densenet201_oof.csv"

# The five paper backbones under image-level (augmented-corpus) sampling — the
# SAME predictions that back Table 1 (replication.tex), so Figs 4 & 5 annotate the
# exact MAE/R2 the replication table reports.
_REPL_IMAGE = [f"replication_{b}_augmented" for b in
               ("densenet201", "resnet50", "vgg19", "xception", "inceptionv3")]


def main() -> None:
    import pandas as pd

    from uda.config import load_config
    from uda import figures as F

    fig_dir = str(FIG)

    # Fig 2 — augmentation sweep (needs any replication config for the data knobs).
    cfg = load_config(ROOT / "configs" / "replication_densenet201.yaml")
    F.figure_2_augmentation(cfg, out_dir=fig_dir)

    # Figs 4 & 5 — image-level replication scatter + error-vs-angle.
    names = [n for n in _REPL_IMAGE if (PRED / f"{n}.csv").exists()]
    if names:
        F.figure_4_pred_vs_actual(names, str(PRED), str(METRICS), fig_dir)
        F.figure_5_error_vs_angle(names, str(PRED), str(METRICS), fig_dir)

    # Protocol comparison + bake-off — both from the frozen 5-fold CV rows.
    F.figure_protocol_comparison(str(ERA_CV), fig_dir)
    F.figure_architecture_bakeoff(str(ERA_CV), fig_dir)

    # Tuning history — best-so-far Optuna objective per backbone.
    _tuning_history(pd, fig_dir)

    # Clinical-eval figures from the tuned DenseNet201 OOF predictions.
    if OOF.exists():
        F.figure_bland_altman(str(OOF), out_dir=fig_dir)
        F.figure_calibration(str(OOF), out_dir=fig_dir)

    # Grad-CAM montage from the available overlay PNGs (a multi-example panel with
    # a saliency-scale reference, replacing the single anecdotal frame).
    _gradcam_montage(fig_dir)

    print(f"figures written -> {FIG}")


def _gradcam_montage(fig_dir: str) -> None:
    """Assemble the available Grad-CAM overlays into one labelled montage with a
    saliency colorbar, so the attention claim rests on several examples, not one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from PIL import Image

    stems = [
        "gradcam_densenet201_09-41-06_1",
        "gradcam_densenet201_09-46-10_1",
        "gradcam_densenet201_09-53-51",
    ]
    imgs = [(s, FIG / f"{s}.png") for s in stems if (FIG / f"{s}.png").exists()]
    if not imgs:
        return
    from uda.figures import _apply_theme, save_figure

    _apply_theme()
    fig, axes = plt.subplots(1, len(imgs), figsize=(3.0 * len(imgs), 3.3))
    if len(imgs) == 1:
        axes = [axes]
    for ax, (stem, path) in zip(axes, imgs):
        ax.imshow(Image.open(path))
        ax.set_title(stem.replace("gradcam_densenet201_", "frame "), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    # Shared saliency colorbar (the overlays use the 'jet' map: blue=low, red=high).
    sm = cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap="jet")
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Grad-CAM saliency (normalized)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    save_figure(fig, "gradcam_montage", fig_dir)
    plt.close(fig)


def _tuning_history(pd, fig_dir: str) -> None:
    """Best-so-far Optuna objective vs trial, per backbone (patient protocol)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from uda.figures import _apply_theme, save_figure

    _apply_theme()
    backbones = ["densenet201", "resnet50", "vgg19", "xception", "inceptionv3"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    plotted = False
    for b in backbones:
        # Prefer the per-protocol patient file; fall back to the legacy single file.
        for cand in (TUN / f"tune_{b}_patient.csv", TUN / f"tune_{b}.csv"):
            if cand.exists():
                df = pd.read_csv(cand).sort_values("number")
                best = df["value"].cummin()
                line, = ax.plot(df["number"], best, lw=1.8, label=b)
                ax.scatter([df["number"].iloc[-1]], [best.iloc[-1]], s=22,
                           color=line.get_color(), zorder=5)
                plotted = True
                break
    if plotted:
        ax.set_xlabel("Optuna trial")
        ax.set_ylabel("best-so-far CV objective (MAPE, %)")
        ax.legend(fontsize=7, ncol=2)
        save_figure(fig, "figure_tuning_history", fig_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
