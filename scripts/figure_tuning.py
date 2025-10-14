"""Figure: Optuna tuning history — best patient-CV MAPE so far vs trial, per backbone.

Reads ``results/tuning/tune_<backbone>.csv`` (one row per trial) and plots the
running minimum (best-so-far) of the objective against trial number, with a marker
at each backbone's final best. Reuses the shared figure theme so it matches the
rest of the paper/site figures. Run: ``pixi run python scripts/figure_tuning.py``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from uda.figures import _apply_theme, save_figure

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "results" / "tuning"
BACKBONES = ["densenet201", "resnet50", "vgg19", "xception", "inceptionv3"]


def main() -> None:
    _apply_theme()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    plotted = 0
    for b in BACKBONES:
        path = TUNING / f"tune_{b}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).sort_values("number")
        best_so_far = df["value"].cummin()
        ax.plot(df["number"], best_so_far, marker="", linewidth=1.8, label=b)
        ax.scatter([df["number"].iloc[-1]], [best_so_far.iloc[-1]], s=22, zorder=5)
        ax.annotate(
            f"{best_so_far.iloc[-1]:.2f}",
            (df["number"].iloc[-1], best_so_far.iloc[-1]),
            textcoords="offset points",
            xytext=(4, 0),
            fontsize=7,
            va="center",
        )
        plotted += 1

    if plotted == 0:
        raise SystemExit("no results/tuning/tune_*.csv found — run scripts/run_tuning.py first")

    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("best patient 5-fold MAPE so far (%)")
    ax.set_title("Hyperparameter search (TPE) — convergence per frozen backbone")
    ax.grid(True, alpha=0.3)
    ax.legend(title="backbone", fontsize=7, title_fontsize=7, loc="upper right")
    svg, png = save_figure(fig, "figure_tuning_history")
    print("wrote", svg)
    print("wrote", png)


if __name__ == "__main__":
    main()
