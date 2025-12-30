---
layout: default
title: About this project
nav: about
mathjax: false
description: About "Reading the Doppler angle" — a research essay by Nilesh Patil re-implementing and extending the EMBC 2019 Doppler-angle estimator, with citation, paper PDF, source repository, and build colophon.
---

<p class="kicker">About</p>

# About this project

<p class="lede">"Reading the Doppler angle" is a research essay on estimating the Doppler beam-to-vessel angle θ directly from a single grayscale B-mode carotid image — no color Doppler, no segmentation. It is a clean, tested, reproducible re-implementation and extension of a 2019 conference paper, with every number on this site regenerated from code.</p>

## The project {#project}

Spectral-Doppler velocity is recovered from the Doppler equation, so the reported velocity scales as the reciprocal of the cosine of the insonation angle θ. That angle is set by hand at the scanner, which makes it a leading, operator-dependent source of velocity error. This project asks whether θ can be read straight off the B-mode image instead.

The work reproduces a five-backbone transfer-learning estimator, isolates the orientation-preserving grid-pooling design choice that makes it work, and then tunes and ensembles it and adds a clinical-grade evaluation. The headline tuned stacked ensemble reaches 2.79% MAPE / 1.96° MAE / R² 0.995 at the image level and 8.53% MAPE / 5.93° MAE / R² 0.952 at the patient level; it reproduces and improves on the original paper's best single model and adds the previously-unmeasured patient-level number. The [Reproduce]({{ '/reproduce/' | relative_url }}) page documents how to regenerate all of it, along with the honest scope and limitations.

## The author {#author}

This research essay was written by Nilesh Patil, first author of the original EMBC 2019 paper. The re-implementation, dual-protocol evaluation, tuning, ensembling, and clinical-grade analysis collected here are his own work.

## The prior version {#prior-version}

This work re-implements and extends:

> N. Patil, A. Anand. "Automated Ultrasound Doppler Angle Estimation Using Deep Learning." <em>Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)</em>, 2019, pp. 28–31. doi:[10.1109/EMBC.2019.8857587](https://doi.org/10.1109/EMBC.2019.8857587)

The original 2017–2018 notebooks and weights are preserved untouched under `legacy/` in the repository.

## Links {#links}

- Manuscript (PDF): [`paper/main.pdf`]({{ '/assets/paper/main.pdf' | relative_url }})
- Source code, data, and regenerated results: [GitHub — ultrasound-doppler-angle-estimation](https://github.com/nilesh-patil/ultrasound-doppler-angle-estimation)
- Results and comparative-analysis narrative: [Results &amp; analysis]({{ '/analysis/' | relative_url }})
- How to reproduce every number: [Reproduce]({{ '/reproduce/' | relative_url }})

## Colophon {#colophon}

This site is a static Jekyll site, built by GitHub Pages, set in Newsreader. The interactive elements — the Doppler-angle explainer, the results charts, and the precomputed prediction explorer — progressively enhance the static SVG figures and degrade to them when JavaScript is unavailable.

Every number on this site is regenerated from `results/` by code (seed 42, Keras 3 / JAX). Nothing is hand-typed.

<p><a href="{{ '/' | relative_url }}">← Back to Overview</a></p>
