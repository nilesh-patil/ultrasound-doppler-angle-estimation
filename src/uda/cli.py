"""Command-line surface for the uda pipeline (wired to pixi tasks).

Subcommands mirror the project's use-case-oriented task table:
``fetch-data`` (extract the canonical images from ``data/raw/Data.zip``),
``labels`` (parse ``data/Results.txt`` + patient_id -> ``data/labels.csv``),
``augment`` (build the corpus and report its size), ``figure2`` (regenerate the
augmentation figure), plus the training/eval/replicate subcommands.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cmd_fetch_data(args: argparse.Namespace) -> None:
    root = _repo_root()
    archive = root / "data" / "raw" / "Data.zip"
    dst = root / "data" / "images"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if name.lower().endswith(".jpg"):
                with zf.open(info) as fsrc, open(dst / name, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                n += 1
    print(f"fetch-data: extracted {n} base images -> {dst.relative_to(root)}")


def cmd_labels(args: argparse.Namespace) -> None:
    from uda.data.labels import build_labels_csv

    root = _repo_root()
    out = root / "data" / "labels.csv"
    df = build_labels_csv(root / "data" / "Results.txt", out)
    n_groups = df["patient_id"].nunique()
    print(
        f"labels: wrote {len(df)} rows -> {out.relative_to(root)} "
        f"({n_groups} patient groups)"
    )


def cmd_augment(args: argparse.Namespace) -> None:
    from uda.config import load_config
    from uda.data.augment import rotation_angles
    from uda.data.dataset import build_corpus

    root = _repo_root()
    cfg = load_config(args.config)
    corpus = build_corpus(cfg, max_images=args.max_images)
    total = corpus.x_train.shape[0] + corpus.x_test.shape[0]
    r = len(rotation_angles(cfg.data))
    n_base = corpus.meta["image_id"].nunique()
    print(
        f"augment[{cfg.name}]: {n_base} base x {r} rot = {total} samples "
        f"(train {corpus.x_train.shape[0]}, test {corpus.x_test.shape[0]}); "
        f"x frame {corpus.x_train.shape[1:]}"
    )
    out = root / "results" / "corpus_meta.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus.meta.to_csv(out, index=False)
    print(f"  meta -> {out.relative_to(root)}")


def cmd_figure2(args: argparse.Namespace) -> None:
    from uda.config import load_config
    from uda.figures import figure_2_augmentation

    cfg = load_config(args.config)
    svg, png = figure_2_augmentation(cfg, image_id=args.image_id)
    print(f"figure2 -> {svg}")
    print(f"         {png}")


def _run_one(cfg, max_images, results_dir):
    from uda.data.dataset import build_corpus
    from uda.evaluation.evaluate import evaluate
    from uda.training.train import train

    corpus = build_corpus(cfg, max_images=max_images)
    res = train(cfg, corpus, out_dir=str(results_dir))
    row = evaluate(
        cfg, res.y_test_true_deg, res.y_test_pred_deg, res.test_meta, out_dir=str(results_dir)
    )
    return row


def cmd_train(args: argparse.Namespace) -> None:
    from uda.config import load_config

    root = _repo_root()
    cfg = load_config(args.config)
    if args.strategy:
        cfg = cfg.model_copy(deep=True)
        cfg.split.strategy = args.strategy
        cfg.name = f"{cfg.name}_{args.strategy}"
    row = _run_one(cfg, args.max_images, root / "results")
    print(
        f"{cfg.name}: MAE={row['mae']:.3f} RMSE={row['rmse']:.3f} ME={row['me']:.2f} "
        f"MAPE={row['mape']:.2f} R2={row['r2']:.3f} (n_test={row['n_test']})"
    )


def cmd_replicate(args: argparse.Namespace) -> None:
    import glob

    from uda import figures
    from uda.config import load_config

    root = _repo_root()
    results = root / "results"
    config_paths = sorted(glob.glob(str(root / "configs" / "replication_*.yaml")))
    by_strategy: dict[str, list[str]] = {s: [] for s in args.strategies}

    for cpath in config_paths:
        base = load_config(cpath)
        for strat in args.strategies:
            cfg = base.model_copy(deep=True)
            cfg.split.strategy = strat
            cfg.name = f"{base.name}_{strat}"
            print(f"[replicate] {cfg.name} ...", flush=True)
            row = _run_one(cfg, args.max_images, results)
            print(
                f"    MAE={row['mae']:.3f}  RMSE={row['rmse']:.3f}  "
                f"R2={row['r2']:.3f}  (n_test={row['n_test']})",
                flush=True,
            )
            by_strategy[strat].append(cfg.name)

    metrics_csv = str(results / "metrics.csv")
    pred_dir = str(results / "predictions")
    fig_dir = str(results / "figures")
    paper_names = by_strategy.get("augmented") or by_strategy.get("image")
    if paper_names:
        figures.figure_4_pred_vs_actual(paper_names, pred_dir, metrics_csv, fig_dir)
        figures.figure_5_error_vs_angle(paper_names, pred_dir, metrics_csv, fig_dir)
    print("[replicate] figures written")


def cmd_tune(args: argparse.Namespace) -> None:
    from uda.config import load_config
    from uda.training.experiment import cv_mape_objective
    from uda.training.tuning import run_study

    root = _repo_root()
    cfg = load_config(args.config).model_copy(deep=True)
    cfg.split.strategy = "patient"  # tuning is scored on the honest protocol
    study_name = args.study or f"tune_{cfg.backbone.name}"

    objective = cv_mape_objective(cfg, k=args.k, max_images=args.max_images)
    print(
        f"[tune] {study_name}: {args.trials} trials, patient {args.k}-fold, dims={args.dims}",
        flush=True,
    )
    res = run_study(
        cfg,
        objective,
        n_trials=args.trials,
        study_name=study_name,
        out_dir=str(root / "results" / "tuning"),
        configs_dir=str(root / "configs"),
        seed=cfg.seed,
        dims=args.dims,
    )
    print(f"[tune] best MAPE={res['best_value']:.3f}  -> {res['best_config_path']}")
    print(f"       trials -> {res['trials_csv']}")
    print(f"       params: {res['best_params']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="uda")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "fetch-data", help="extract base images from data/raw/Data.zip -> data/images"
    ).set_defaults(
        func=cmd_fetch_data
    )
    sub.add_parser("labels", help="parse Results.txt -> data/labels.csv").set_defaults(
        func=cmd_labels
    )

    sa = sub.add_parser("augment", help="build the corpus and report its size")
    sa.add_argument("--config", required=True)
    sa.add_argument("--max-images", type=int, default=None)
    sa.set_defaults(func=cmd_augment)

    sf = sub.add_parser("figure2", help="regenerate the augmentation figure")
    sf.add_argument("--config", required=True)
    sf.add_argument("--image-id", default="09-41-06_1")
    sf.set_defaults(func=cmd_figure2)

    st = sub.add_parser("train", help="train+eval one config")
    st.add_argument("--config", required=True)
    st.add_argument("--strategy", choices=["image", "patient", "augmented"], default=None)
    st.add_argument("--max-images", type=int, default=None)
    st.set_defaults(func=cmd_train)

    sr = sub.add_parser("replicate", help="train+eval all backbones, then figures")
    sr.add_argument("--strategies", nargs="+", default=["augmented", "image", "patient"])
    sr.add_argument("--max-images", type=int, default=None)
    sr.set_defaults(func=cmd_replicate)

    su = sub.add_parser("tune", help="Optuna TPE search over head+optimizer (patient CV)")
    su.add_argument("--config", required=True)
    su.add_argument("--trials", type=int, default=40)
    su.add_argument("--k", type=int, default=5, help="patient CV folds for the objective")
    su.add_argument("--study", default=None, help="study name (default tune_<backbone>)")
    su.add_argument("--dims", choices=["head", "head+pool+target"], default="head")
    su.add_argument("--max-images", type=int, default=None)
    su.set_defaults(func=cmd_tune)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
