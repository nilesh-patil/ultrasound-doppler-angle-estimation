---
layout: default
title: Every number here is regenerated from code
nav: reproduce
mathjax: false
description: How to reproduce the Doppler-angle estimator end to end — the pixi quickstart, the full label-to-tables pipeline, the three-backend environments, the repository layout, and the data provenance and honest scope.
---

<p class="kicker">Reproduce</p>

# Every number here is regenerated from code

<p class="lede">There is one model codebase, written once with Keras 3, and one results directory that the manuscript, the analysis narrative, and this website all read from. Nothing on this site is hand-typed; every table and figure is regenerated from <code>results/</code> by code, at seed 42, on the JAX-CPU reference backend. This page is the recipe.</p>

## Reproduce it end to end {#reproduce}

The pipeline runs under [pixi](https://pixi.sh): `pixi install` solves and installs the default JAX-CPU environment, and the tasks below carry it from raw images to the regenerated paper tables. The same commands appear verbatim in the analysis narrative (§8) and the repository README.

<div class="table-wrap" role="region" aria-label="Quickstart commands" tabindex="0">

```bash
pixi install            # solve + install the default (JAX-CPU) env
pixi run test           # run the test suite
pixi run all            # labels → 5-backbone replication
```

</div>

The full run, from labels through tuning, out-of-fold ensembling, and the regenerated paper tables:

<div class="table-wrap" role="region" aria-label="End-to-end reproduction commands" tabindex="0">

```bash
pixi install
pixi run all                                                     # labels → augment → replicate → figures
pixi run python scripts/run_tuning.py --protocol both --trials 20 --k 5
pixi run python scripts/oof_ensemble.py --protocol image
pixi run python scripts/oof_ensemble.py --protocol patient
pixi run python scripts/gen_paper_tables.py                      # regenerate paper tables from results/
```

</div>

Every number in `docs/analysis.md` and in `paper/` is regenerated from `results/` by `scripts/gen_paper_tables.py`; the `results/` directory itself is regenerated and never hand-edited. The default `pixi run test` stays green by skipping the TensorFlow-only Grad-CAM and deploy tests (`skipif KERAS_BACKEND != "tensorflow"`); those run under the Apple-silicon GPU environment below.

## One model codebase, three backends {#backends}

The model code is written once with Keras 3, and the backend is selected per machine by the `KERAS_BACKEND` environment variable, which each pixi environment sets automatically. The same five pretrained ImageNet backbones from `keras.applications` (VGG19, ResNet50, DenseNet201, Xception, InceptionV3) are used across all three, which keeps the implementation one rewrite away from the original Keras code.

<div class="table-wrap" role="region" aria-label="The three pixi environments and their Keras backends" tabindex="0">

<table class="data">
  <thead>
    <tr><th scope="col">Command prefix</th><th scope="col">Backend</th><th scope="col">Use</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row"><code>pixi run …</code></th><td><strong>JAX · CPU</strong> (<code>KERAS_BACKEND=jax</code>)</td><td>default; reproducible reference, CI</td></tr>
    <tr><th scope="row"><code>pixi run -e mac-gpu …</code></th><td><strong>TensorFlow · Metal</strong> (<code>KERAS_BACKEND=tensorflow</code>)</td><td>Apple-silicon GPU</td></tr>
    <tr><th scope="row"><code>pixi run -e cuda …</code></th><td><strong>JAX · CUDA</strong> (<code>KERAS_BACKEND=jax</code>)</td><td>Linux + NVIDIA training</td></tr>
  </tbody>
</table>

</div>

All runs use seed 42. The JAX-CPU environment is the reference target for reproduction and continuous integration; the Apple-silicon (TensorFlow-Metal) and Linux-NVIDIA (JAX-CUDA) environments exist for hardware-accelerated training and for the Grad-CAM and deploy paths that require the TensorFlow backend.

## The repository layout {#layout}

The library is typed and tested; configuration lives in one YAML file per experiment; the canonical data, results, and the manuscript each have their own tree.

<div class="table-wrap" role="region" aria-label="Repository layout" tabindex="0">

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
docs/               # analysis.md — the results & comparative-analysis narrative; this website
```

</div>

## The data {#data}

The cohort is the one introduced in the EMBC 2019 study, drawn from the public SPLab (Brno) ultrasound database: 84 longitudinal common-carotid artery (CCA) B-mode images from about ten young, healthy volunteers (mean age 27.5 years), usually eight images per volunteer. Acquisition used a single Sonix OP scanner with linear-array transducers (10 and 14 MHz), with the subject supine and the right CCA imaged in longitudinal view at roughly 390×330 pixels.

Each image is rotated over [−60°, +60°] in 5° steps to give 25 oriented views and approximately 2,100 labelled examples; the rotation supplies an exact angle label for each view. These 2,100 examples are coupled copies of 84 base images, not 2,100 independent acquisitions. The supervised target is the Doppler angle θ. Because the database carries no clinical Doppler-angle annotation, the reference θ was measured offline by one operator in a custom MATLAB interface, drawing a line along the vessel wall and recording its inclination. This single hand-drawn reading is the only ground truth available per image, and its own measurement error is unknown.

## Honest scope and limitations {#limitations}

The scope of what these numbers support is narrow, and the limitations travel with the results throughout this site.

The cohort is small and single-center: about ten volunteers, 84 base images, one scanner, the carotid only. Patient five-fold cross-validation rests on a few held-out subjects per fold (around two per test fold), so the cross-subject standard deviation is non-negligible and external validity to other anatomies, ages, or pathologies is untested.

There is one reference reading per image, so the agreement reported on the Clinical page is method-versus-reference, not inter-observer; the reference itself is a noisy estimate of the true angle.

Transfer learning is essential on this hardware. A from-scratch CNN fails to converge — image-level MAPE 101.6% with R² −6.14, and patient-level MAPE 100.8% with R² −5.68 — and end-to-end DenseNet201 fine-tuning exhausts the Apple-silicon GPU even at batch size 16. Both reinforce that frozen transfer is the right tool for this dataset.

The backbones are frozen only. End-to-end fine-tuning, transformer and self-supervised encoders (ViT, DINOv2), and ultrasound foundation models (USFM, MedSAM) are deferred to CUDA-class hardware and are not claimed here.

## The prior version {#prior-version}

This work re-implements and extends:

> N. Patil, A. Anand. "Automated Ultrasound Doppler Angle Estimation Using Deep Learning." <em>Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)</em>, 2019, pp. 28–31. doi:[10.1109/EMBC.2019.8857587](https://doi.org/10.1109/EMBC.2019.8857587)

The original 2017–2018 notebooks and weights are preserved untouched, output-stripped, under `legacy/`. The extension adds dual-protocol (image-level and patient-level) evaluation, Optuna head tuning, a five-backbone tuned stacked ensemble, a clinical-grade evaluation (split-conformal intervals, a Bland–Altman analysis, rotation test-time augmentation), Grad-CAM attribution, and a classical structure-tensor angle prior with its learned-classical fusion.

## Cite and source {#cite}

The code, data, regenerated results, and the full manuscript are public:

- Repository and data: [GitHub — ultrasound-doppler-angle-estimation](https://github.com/nilesh-patil/ultrasound-doppler-angle-estimation)
- Manuscript (PDF): [`paper/main.pdf`]({{ '/assets/paper/main.pdf' | relative_url }})
- Results and comparative-analysis narrative: [`docs/analysis.md`]({{ 'analysis.md' | relative_url }})
- Project and author notes: [About]({{ '/about/' | relative_url }})

<p><a href="{{ '/about/' | relative_url }}">About this project →</a></p>
