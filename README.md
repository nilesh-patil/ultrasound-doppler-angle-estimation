# Ultrasound Doppler Angle Estimation

Learning the **Doppler angle** θ — the angle between the ultrasound beam and the
vessel flow — directly from a single grayscale B-mode image, with **no color
Doppler and no segmentation**. Spectral-Doppler velocity is recovered from

```
f_d = 2·f₀·v·cos(θ) / c        ⟹        v ∝ 1 / cos(θ)
```

so an error in θ propagates straight into the reported velocity. The angle is set
manually by the sonographer, which makes it a leading, operator-dependent source
of velocity error. This is the motivation for estimating it automatically from the
image.

This repository is a clean, tested, reproducible re-implementation and extension of:

> **N. Patil, A. Anand — "Automated Ultrasound Doppler Angle Estimation Using Deep
> Learning."** *Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)*, 2019.
> doi:10.1109/EMBC.2019.8857587

It replicates the five-backbone transfer-learning result and isolates the design
choice that makes it work (**orientation-preserving grid pooling**). On top of that
it tunes and ensembles the estimator and adds a clinical-grade evaluation. The
original 2017–2018 notebooks and weights are preserved untouched under
[`legacy/`](legacy/); the manuscript is in [`paper/`](paper/); the full results
narrative is in [`docs/analysis.md`](docs/analysis.md).

## Headline results

| Estimator | Image-level sampling | Patient-level sampling |
|---|---|---|
| Frozen DenseNet201 (grid pooling) | 5.84 % MAPE · 3.77° MAE | — |
| Tuned DenseNet201 | 4.03 % · 3.00° | 10.80 % · 8.62° |
| **Tuned stacked ensemble (5 backbones)** | **2.79 % · 1.96° · R²=0.995** | **8.53 % · 5.93° · R²=0.952** |

The tuned ensemble **reproduces and improves on** the original paper's best single
model (≈ 4.03 % MAPE / 2.87° MAE, image-level), and additionally reports the
previously-unmeasured patient-level (cross-subject) number. Clinical-grade
evaluation adds split-conformal 90 % intervals of ±20.5° (valid empirical
coverage), a Bland–Altman method-vs-reference analysis (+4.3° bias), and
rotation test-time augmentation that nearly halves the base-image error.

## Two sampling protocols

The same 2,100-image corpus is scored two complementary ways, each answering a
different question, and each tuned to its own best:

| Protocol | Partition | Question |
|---|---|---|
| **Image-level** | random split over the augmented corpus (the original study's protocol) | accuracy across the full population of orientations and imaging conditions |
| **Patient-level** | grouped split by volunteer (`GroupKFold`) — no subject in both folds | generalization to a previously **unseen patient** |

Neither is "wrong"; a complete picture reports both. The ~3× spread between them
quantifies how much within-population accuracy is anatomy-specific.

## Environments (one model codebase, three backends)

Model code is written **once** with Keras 3; the backend is chosen per machine by
`KERAS_BACKEND`, set automatically by the pixi environment you select.

| Command prefix | Backend | Use |
|---|---|---|
| `pixi run …`            | **JAX · CPU** (`KERAS_BACKEND=jax`) | default; reproducible reference, CI |
| `pixi run -e mac-gpu …` | **TensorFlow · Metal** (`KERAS_BACKEND=tensorflow`) | Apple-silicon GPU |
| `pixi run -e cuda …`    | **JAX · CUDA** (`KERAS_BACKEND=jax`) | Linux + NVIDIA training |

Keras 3 keeps the five pretrained ImageNet backbones (VGG19, ResNet50,
DenseNet201, Xception, InceptionV3) from `keras.applications` and stays one rewrite
away from the original Keras code (the strongest replication claim) at the cost
of a thin abstraction layer over JAX.

## Quickstart

```bash
pixi install            # solve + install the default (JAX-CPU) env
pixi run test           # run the test suite
pixi run all            # labels → augment → 5-backbone replicate → figures
```

## Project layout

```
src/uda/            # typed, tested library
  config.py seed.py cli.py figures.py deploy.py
  data/             # images, labels, augmentation, splits, corpus assembly
  models/           # backbone registry, shallow head, assembled estimator, angle targets
  training/         # trainer, experiment runner, cross-validation, Optuna tuning
  evaluation/       # metrics, ensembling, conformal, calibration, agreement, TTA, nested CV, uncertainty
  interpret/        # Grad-CAM, structure-tensor geometric prior, fusion
configs/            # one YAML per experiment (backbone × protocol × target)
scripts/            # tuning sweep, OOF ensemble, paper-table + figure generators
data/               # canonical images/ + labels.csv + Results.txt; raw archives under data/raw/
results/            # metrics.csv + predictions/ + figures/  (regenerated, never hand-edited)
tests/              # data, label-math, leakage-assertion, target round-trip, eval-module contracts
legacy/             # the original 2018 notebooks + README (untouched, output-stripped)
paper/              # LaTeX manuscript; tables generated from results/ by scripts/gen_paper_tables.py
docs/               # analysis.md — the results & comparative-analysis narrative
```

## Reproduce it end to end

```bash
pixi install
pixi run all                                                     # replicate → figures
pixi run python scripts/run_tuning.py --protocol both --trials 20 --k 5
pixi run python scripts/oof_ensemble.py --protocol image
pixi run python scripts/oof_ensemble.py --protocol patient
pixi run python scripts/gen_paper_tables.py                      # regenerate paper tables from results/
```

Every number in [`docs/analysis.md`](docs/analysis.md) and [`paper/`](paper/) is
regenerated from `results/`; nothing is hand-typed. Seed 42; default backend
JAX-CPU. The default `pixi run test` stays green by skipping the TensorFlow-only
Grad-CAM/deploy tests (`skipif KERAS_BACKEND != "tensorflow"`); run those under
`pixi run -e mac-gpu`.
