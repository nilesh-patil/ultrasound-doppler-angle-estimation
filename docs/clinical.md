---
layout: default
title: What a clinic would actually need
nav: clinical
toc: true
mathjax: true
scripts:
  - data.js
  - charts.js
  - demo.js
description: Clinical-grade evaluation of the tuned DenseNet201 — split-conformal prediction intervals (±20.50° at 90%), a Bland–Altman method-vs-reference agreement analysis (−4.31° bias), rotation test-time augmentation (7.80°→4.72° base-image MAE), and Grad-CAM attributions on the vessel wall.
---

<p class="kicker">Clinical evaluation</p>

# What a clinic would actually need

<p class="lede">A clinic does not act on a single number. It acts on a number with a band around it, an honest account of how far it sits from the reference it would replace, and some assurance that the model is reading the anatomy rather than an artefact. This page reports four such checks on the tuned DenseNet201, all recomputed from the saved patient-level out-of-fold predictions.</p>

These results are a research-grade characterisation, not a deployment claim. They are computed on the same small single-center cohort — roughly ten volunteers, 84 base images, one Sonix OP scanner, common carotid artery only — and against a single MATLAB-GUI reference reading per image. Read them as evidence about the estimator's behaviour, not as a clearance for clinical use.

## A band, not just a number {#conformal}

A point estimate of the insonation angle is only useful if its uncertainty is legible. We wrap the tuned DenseNet201 in a split-conformal procedure: hold out a calibration set, take the empirical quantile of the absolute residuals, and emit a symmetric prediction interval whose half-width is that quantile. For a target miscoverage $$\alpha$$ and calibration residuals $$r_i = \lvert \theta_{\text{true},i} - \theta_{\text{pred},i}\rvert$$, the half-width is

$$
q_{1-\alpha} = \operatorname{Quantile}_{1-\alpha}\big(\{r_i\}\big), \qquad
\text{interval} = \big[\,\hat{\theta} - q_{1-\alpha},\; \hat{\theta} + q_{1-\alpha}\,\big].
$$

The split is **patient-disjoint** (a group-conformal split: 900 calibration rows and 1200 test rows, with no patient appearing in both), so the interval is distribution-free and finite-sample valid. At a single seed-42 split the empirical coverage meets or exceeds the nominal level at every band, which is why the intervals are wide but honest.

<div class="table-wrap" role="region" aria-label="Split-conformal prediction intervals" tabindex="0">
<table class="data">
  <thead>
    <tr><th scope="col">Nominal level</th><th scope="col">Interval half-width</th><th scope="col">Empirical coverage</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">80%</th><td>±15.01°</td><td>89.7%</td></tr>
    <tr class="is-best"><th scope="row">90%</th><td>±20.50°</td><td>95.2%</td></tr>
    <tr><th scope="row">95%</th><td>±26.03°</td><td>97.8%</td></tr>
  </tbody>
</table>
</div>

The headline band is **±20.50° at the 90% level**, with empirical coverage of 95.2% — at or above nominal. The interval is honest precisely because it is wide: roughly forty degrees of total width is the price of distribution-free validity on a cohort this size.

<details class="caveat">
<summary>The conformal split sits on top of pooled out-of-fold predictions, which can make the bands look mildly optimistic.</summary>
The OOF predictions on which these intervals are calibrated were produced across the cross-validation folds rather than on a single fully held-out test set, and the same pooled-OOF predictions feed the stacked ensemble reported on the <a href="{{ '/results/' | relative_url }}#ensemble">Results page</a>. Stacking and calibrating on out-of-fold predictions can be mildly optimistic, so treat the empirical coverage as an estimate from a single seed-42 patient-disjoint split (900 calibration rows, 1200 test rows), not a guarantee.
</details>

<figure class="fig widget wide" id="chart-conformal-calibration-fig">
  <p class="fig__num">Figure 1</p>
  <img class="fallback" src="{{ '/assets/figures/figure_calibration.svg' | relative_url }}" alt="Calibration plot: a step curve of empirical coverage against nominal level, with the empirical curve sitting on or above the diagonal at the 80, 90, and 95 percent levels." />
  <div class="widget__canvas live" id="chart-conformal-calibration" hidden></div>
  <figcaption>Split-conformal coverage meets or exceeds the nominal level at every band; the empirical values (89.7%, 95.2%, 97.8%) sit at or above the diagonal, so the prediction intervals are wide but valid.</figcaption>
</figure>

## Agreement, not inter-observer {#agreement}

The estimator would, in practice, stand in for a sonographer's manual angle setting, so the right question is how far it sits from the reference reading it would replace. A Bland–Altman analysis plots the signed difference against the mean of the two readings. Per sample, across all 2,100 views, the bias is **−4.31°**, with 95% limits of agreement of **−24.25° … +15.63°** (n=2,100). Aggregated per patient, the bias is **−4.56°**, with limits of agreement of **−16.87° … +7.75°** (n=12). The bias is small and negative: the model reads slightly lower than the reference.

<details class="caveat">
<summary>This is a method-vs-reference comparison against a single reading, not an inter-observer agreement study.</summary>
There is exactly one MATLAB-GUI reference reading (<code>theta_true</code>) per image, so the Bland–Altman analysis measures how the model agrees with that single reference — not how two independent readers agree with each other. The reference is itself a noisy estimate of the true angle, and we cannot separate model error from reference error. The differences are also signed with a 180°-periodic wrap, and the per-sample n=2,100 counts 25 coupled rotation copies of each of the 84 base images, not 2,100 independent measurements.
</details>

<figure class="fig widget wide" id="chart-bland-altman-fig">
  <p class="fig__num">Figure 2</p>
  <img class="fallback" src="{{ '/assets/figures/figure_bland_altman.svg' | relative_url }}" alt="Bland–Altman plot: signed difference between model and reference angle on the vertical axis against their mean on the horizontal axis, with a bias line near minus four degrees and dashed 95 percent limits-of-agreement lines above and below." />
  <div class="widget__canvas live" id="chart-bland-altman" hidden></div>
  <figcaption>Model and reference agree to within a small negative bias of −4.31°; the 95% limits of agreement span −24.25° to +15.63°. The interactive view plots a downsampled set of the per-sample differences, with a per-patient toggle for the twelve patient-proxy means.</figcaption>
</figure>

## De-rotation nearly halves the per-image error {#tta}

The augmentation that builds the training corpus is also a tool at inference. Each of the 84 base images carries 25 rotation views; median rotation test-time augmentation de-rotates every view back to the base frame and reduces them circularly (180°-periodic, seam-safe). On the base images this takes the raw base-image MAE from **7.80° down to 4.72°** under the circular median (and to 5.31° under the circular mean) — roughly halving the per-image error.

<div class="callout">
  <p class="callout__title">Rotation TTA, base-image MAE</p>
  <p>7.80° &rarr; <strong>4.72°</strong> (circular median over rotations); 7.80° &rarr; 5.31° (circular mean). Measured on the 84 de-rotated base images.</p>
</div>

<details class="caveat">
<summary>The 7.80° base-image figure is on a different footing from the headline patient-level numbers.</summary>
The 7.80° → 4.72° reduction is measured on the 84 de-rotated base frames, which is not the same population as the patient-level pooled-OOF evaluation. The 7.80° base-image MAE is the like-for-like starting point for this de-rotation experiment, and it should not be read directly against the 5.93° / 8.53% patient-level headline reported on the <a href="{{ '/results/' | relative_url }}">Results page</a>.
</details>

## It looks at the vessel wall {#gradcam}

A model that reports an angle should be reading the anatomy that defines the flow axis. Grad-CAM attributions for the tuned DenseNet201 localise on the vessel wall — the longitudinal common-carotid boundary that fixes the orientation — consistent with the network keying on the geometry rather than on an artefact or label leak.

<figure class="fig wide" id="fig-gradcam-montage">
  <p class="fig__num">Figure 3</p>
  <img class="fallback" src="{{ '/assets/figures/gradcam_montage.svg' | relative_url }}" alt="Grad-CAM montage: a row of carotid B-mode frames overlaid with warm heatmaps, each concentrated along the long bright vessel-wall boundary that runs across the frame." />
  <figcaption>Grad-CAM attributions concentrate along the vessel wall, the structure that defines the flow axis the model is estimating.</figcaption>
</figure>

<div class="fig-row">
  <figure class="fig">
    <img src="{{ '/assets/figures/gradcam_densenet201_09-41-06_1.png' | relative_url }}" alt="Grad-CAM overlay on carotid B-mode image 09-41-06_1: the heatmap sits on the vessel-wall boundary." />
    <figcaption>Image 09-41-06_1.</figcaption>
  </figure>
  <figure class="fig">
    <img src="{{ '/assets/figures/gradcam_densenet201_09-46-10_1.png' | relative_url }}" alt="Grad-CAM overlay on carotid B-mode image 09-46-10_1: the heatmap sits on the vessel-wall boundary." />
    <figcaption>Image 09-46-10_1.</figcaption>
  </figure>
  <figure class="fig">
    <img src="{{ '/assets/figures/gradcam_densenet201_09-53-51.png' | relative_url }}" alt="Grad-CAM overlay on carotid B-mode image 09-53-51: the heatmap sits on the vessel-wall boundary." />
    <figcaption>Image 09-53-51.</figcaption>
  </figure>
</div>

<details class="caveat">
<summary>These attributions are illustrative, not a cohort-wide attribution study.</summary>
The panels above are three hand-picked frames at the coarse Grad-CAM grid resolution. They are qualitative evidence that the network attends to the vessel wall on these examples; they are not a quantitative, cohort-wide attribution analysis, and they do not certify that the model never relies on an artefact elsewhere.
</details>

## Explore the predictions {#explorer}

The explorer below is driven by the real saved out-of-fold predictions for the tuned DenseNet201 on the study images. Pick one of the 84 base images, scrub the rotation, and read the reference angle, the model's estimate, the signed residual, and the ±20.50° conformal band on a number line. Nothing here is computed in the browser beyond the signed error $$\theta_{\text{pred}} - \theta_{\text{true}}$$; the angles and the band are the precomputed values from the evaluation.

<details class="caveat">
<summary>These are precomputed out-of-fold predictions, not live in-browser inference.</summary>
The explorer reads precomputed real predictions exported from <code>results/predictions/tuned_densenet201_oof.csv</code> — the same pooled-OOF predictions used throughout this page. It is a viewer over saved numbers, not a live model; running the network in the browser on a freshly uploaded image is out of scope for this version.
</details>

<section id="prediction-demo" class="demo" aria-label="Precomputed prediction explorer">
  <figure class="fig widget" id="demo-fig">
    <p class="fig__num">Figure 4</p>
    <img class="fallback" src="{{ '/assets/figures/figure4_pred_vs_actual.svg' | relative_url }}" alt="Scatter of predicted versus actual insonation angle, with points clustered along the identity line, shown as the static fallback for the prediction explorer." />
    <div class="demo__grid live" hidden>
      <div class="demo__preview">
        <img class="demo__img" id="demo-thumb" src="{{ '/assets/images/bmode/09-49-17_1.jpg' | relative_url }}" alt="Selected carotid B-mode study image preview." />
      </div>
      <div class="demo__panel">
        <div class="controls">
          <div class="control">
            <label class="control__label" for="demo-image-select">Study image</label>
            <select id="demo-image-select"></select>
          </div>
          <div class="control">
            <label class="control__label" for="demo-rotation-scrub">Rotation view</label>
            <input type="range" id="demo-rotation-scrub" min="-60" max="60" step="5" value="0" aria-valuetext="0 degrees" />
            <span class="control__value" id="demo-rotation-value">0°</span>
          </div>
        </div>
        <dl class="demo__stats" id="demo-result" aria-live="polite"></dl>
        <div class="widget__canvas" id="demo-band" role="img" aria-label="Number line showing the reference angle, the model estimate, and the ±20.50° conformal band."></div>
      </div>
    </div>
    <figcaption>Precomputed real out-of-fold predictions for the tuned DenseNet201 on the 84 study images, with the ±20.50° conformal band drawn around each estimate.</figcaption>
  </figure>
</section>

---

<p class="read-on"><a href="{{ '/reproduce/' | relative_url }}">Next: Reproduce &rarr;</a></p>
