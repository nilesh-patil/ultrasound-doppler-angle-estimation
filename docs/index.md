---
layout: default
title: Reading the Doppler angle off the image
nav: overview
mathjax: true
scripts:
  - data.js
  - doppler-explainer.js
description: A deep-learning estimator that reads the Doppler beam-to-vessel angle directly from a single grayscale B-mode carotid image, without color Doppler or segmentation. It reaches 1.96° MAE / 2.79% MAPE / R²=0.995 image-level and 5.93° / 8.53% / R²=0.952 patient-level.
---

<p class="kicker">Ultrasound · Doppler angle · deep learning</p>

# Reading the Doppler angle off the image

<p class="lede">A tuned stacked ensemble of five frozen ImageNet backbones reads the Doppler
beam-to-vessel angle straight off a single grayscale B-mode image at
<strong>1.96° MAE / 2.79% MAPE / R²=0.995</strong> across the augmented corpus, and at
<strong>5.93° MAE / 8.53% MAPE / R²=0.952</strong> when it is asked to generalize to an
unseen patient — a roughly 3× gap that is the honest signature of a small cohort. It
reproduces and improves on the EMBC 2019 paper's best single model and adds the
patient-level number that work never reported.</p>

The angle is read with no color Doppler and no segmentation. The orientation is
already present in the frozen ImageNet features; the load-bearing design choice is a
small grid-pooling head that keeps it, where a conventional rotation-invariant pooling
throws it away.

## Why a few degrees of angle is a velocity problem {#why-angle}

Spectral-Doppler velocity is recovered from the Doppler equation,

$$ f_d = \frac{2\,f_0\,v\cos\theta}{c} \quad\Longrightarrow\quad v \propto \frac{1}{\cos\theta}, $$

where $$f_d$$ is the measured frequency shift, $$f_0$$ the transmit frequency,
$$c$$ the speed of sound, and $$\theta$$ the angle between the ultrasound beam and
the vessel flow. Because velocity scales as $$1/\cos\theta$$, an error in $$\theta$$
propagates straight into the reported velocity, and the sensitivity $$\tan\theta$$
grows sharply as $$\theta$$ approaches the steep angles used in practice: near 60°,
each additional degree of angle error moves the reported velocity by about three
percent.

The angle is set manually by the sonographer, which makes it a leading,
operator-dependent source of velocity error. Reading it automatically from the image
removes that step from the human, which is the motivation for the whole estimator.

<details class="caveat">
<summary>What the ~35% accreditation figure does and does not say</summary>
Improper angle correction has been flagged in up to roughly 35% of vascular-lab
accreditation applications (Saad 2009). That is a literature figure describing the field
at large, not a measurement on this cohort, and it motivates the problem rather than
quantifying anything this estimator was tested against.
</details>

## Rotate the beam and watch the velocity move {#explainer}

The slider below sweeps the ultrasound beam across a real B-mode image to set an
illustrative beam-to-vessel angle $$\theta$$, and shows, live, the fractional velocity
error a clinic would inherit from it. The beam overlay is a synthetic teaching tool for
the $$1/\cos\theta$$ relationship; it is not the rotation-augmentation grid the model
was trained on.

<figure class="fig widget" id="chart-explainer">
  <img class="fallback" loading="lazy" decoding="async" src="{{ '/assets/images/bmode/09-49-17_1.jpg' | relative_url }}"
       alt="Longitudinal grayscale B-mode image of a common carotid artery — the static fallback for the interactive Doppler-angle explainer." />
  <div class="live" hidden>
    <section id="doppler-explainer" class="explainer" data-base="{{ '/assets/data/' | relative_url }}">
      <div class="explainer__stage" role="img"
           aria-label="Longitudinal grayscale B-mode image of a common carotid artery, with a beam-to-vessel angle overlay set by the slider."
           style="background-image: url('{{ '/assets/images/bmode/09-49-17_1.jpg' | relative_url }}');">
        <svg id="explainer-overlay" class="explainer__overlay" viewBox="0 0 100 100"
             role="presentation" aria-hidden="true"></svg>
      </div>
      <div class="controls">
        <label class="control" for="explainer-rotation">
          <span class="control__label">Beam-to-vessel angle θ</span>
          <input id="explainer-rotation" type="range" min="0" max="89" step="1" value="60"
                 aria-describedby="explainer-theta-readout" />
        </label>
      </div>
      <div id="explainer-theta-readout" class="explainer__readout">
        <span class="readout__theta">θ = 60°</span>
        <span class="readout__err">velocity error 0.0%</span>
        <span id="explainer-velocity-gauge" class="readout__mult">velocity multiplier ×2.00</span>
      </div>
      <svg id="explainer-cos-curve" class="explainer__curve" viewBox="0 0 100 60"
           role="img" aria-label="The 1/cos θ velocity-multiplier curve, with a marker at the angle set by the slider."></svg>
    </section>
  </div>
  <figcaption>
    <span class="fig__num">Figure 1.</span> The reported velocity is proportional to
    \( 1/\cos\theta \), so it rises slowly near small angles and steeply past 60°. The
    table below this figure shows the same relationship without scripting; the slider
    is an illustrative model, not the trained estimator.
  </figcaption>
</figure>

<div class="table-wrap explainer__static-table" role="region" aria-label="Velocity multiplier and fractional velocity error by Doppler angle, relative to a 60° reference" tabindex="0">
<table class="data" id="explainer-fallback-table">
  <caption>Velocity multiplier \( 1/\cos\theta \) and the fractional velocity error relative to a 60° reference.</caption>
  <thead>
    <tr><th scope="col">θ (degrees)</th><th scope="col">Velocity multiplier ×</th><th scope="col">Error vs 60°</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">45</th><td>1.41</td><td>−29.3%</td></tr>
    <tr><th scope="row">50</th><td>1.56</td><td>−22.2%</td></tr>
    <tr><th scope="row">55</th><td>1.74</td><td>−12.8%</td></tr>
    <tr><th scope="row">60</th><td>2.00</td><td>0.0%</td></tr>
    <tr><th scope="row">65</th><td>2.37</td><td>+18.3%</td></tr>
    <tr><th scope="row">70</th><td>2.92</td><td>+46.2%</td></tr>
    <tr><th scope="row">75</th><td>3.86</td><td>+93.2%</td></tr>
    <tr><th scope="row">80</th><td>5.76</td><td>+187.9%</td></tr>
  </tbody>
</table>
</div>

## Headline results {#headline}

The tuned stacked ensemble reproduces and improves on the EMBC 2019 paper's best
single model (≈ 2.87° MAE / 4.03% MAPE, image-level) and additionally reports the
previously-unmeasured patient-level number.

<div class="kpi-row">
  <div class="kpi"><span class="kpi__value">1.96°</span><span class="kpi__label">Image-level MAE</span></div>
  <div class="kpi"><span class="kpi__value">2.79%</span><span class="kpi__label">Image-level MAPE</span></div>
  <div class="kpi"><span class="kpi__value">5.93°</span><span class="kpi__label">Patient-level MAE</span></div>
  <div class="kpi"><span class="kpi__value">8.53%</span><span class="kpi__label">Patient-level MAPE</span></div>
</div>

<div class="table-wrap" role="region" aria-label="Headline results across the image-level and patient-level sampling protocols" tabindex="0">
<table class="data">
  <caption>Headline accuracy under each sampling protocol, with MAPE and MAE shown
  separately. Image-level rows are five-fold cross-validation means or pooled
  out-of-fold; patient-level rows are pooled out-of-fold.</caption>
  <thead>
    <tr>
      <th scope="col" rowspan="2">Estimator</th>
      <th scope="colgroup" colspan="2" style="text-align:center">Image-level sampling</th>
      <th scope="colgroup" colspan="2" style="text-align:center; border-left:1px solid var(--hairline)">Patient-level sampling</th>
    </tr>
    <tr>
      <th scope="col">MAPE (%)</th>
      <th scope="col">MAE (°)</th>
      <th scope="col" style="border-left:1px solid var(--hairline)">MAPE (%)</th>
      <th scope="col">MAE (°)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Frozen DenseNet201 (grid pooling)</th>
      <td>5.84</td>
      <td>3.77</td>
      <td style="border-left:1px solid var(--hairline)">—</td>
      <td>—</td>
    </tr>
    <tr>
      <th scope="row">Tuned DenseNet201</th>
      <td>4.03</td>
      <td>3.00</td>
      <td style="border-left:1px solid var(--hairline)">10.80</td>
      <td>8.62</td>
    </tr>
    <tr class="is-best">
      <th scope="row">Tuned stacked ensemble (5 backbones)</th>
      <td>2.79</td>
      <td>1.96</td>
      <td style="border-left:1px solid var(--hairline)">8.53</td>
      <td>5.93</td>
    </tr>
  </tbody>
</table>
</div>

## Two ways to be right {#two-protocols}

The same 2,100-image corpus is scored two complementary ways: an image-level protocol
that splits the augmented images at random (the original study's lens), and a
patient-level protocol that splits by volunteer with `GroupKFold` so no subject appears
in both folds. Neither is wrong — the image-level number measures accuracy across the
full population of orientations, the patient-level number measures generalization to an
unseen patient, and the [Method]({{ '/method/' | relative_url }}#protocols) and
[Results]({{ '/results/' | relative_url }}#protocols) pages report both.

<details class="caveat">
<summary>Why the patient-level number is the harder, more honest one</summary>
The patient-level result rests on a small, single-center cohort (~10 volunteers, 84
base images, one scanner, carotid only), with only about two held-out subjects per
fold, so its cross-subject variance is non-negligible and its external validity to
other anatomies or scanners is untested. The roughly 3× spread between the two
protocols quantifies how much of the within-population accuracy is anatomy-specific.
</details>

## What this is not {#not}

This is a research re-implementation, not a clinical tool: nothing here is validated
for diagnosis, deployment, or velocity or stenosis reporting, the backbones are frozen
rather than fine-tuned, and a from-scratch CNN on this data fails outright. The 2,100
labelled examples are 25 rotated copies of 84 base images, not 2,100 independent
samples, and the agreement analysis compares the model against a single reference
reading per image rather than against a second observer. The full
[scope and limitations]({{ '/reproduce/' | relative_url }}#limitations) are on the
Reproduce page.

## Read on {#read-on}

- [How the estimator is built]({{ '/method/' | relative_url }}) — the data, the two protocols, the frozen backbone, and the grid-pooling thesis.
- [Five backbones, two protocols, one ensemble]({{ '/results/' | relative_url }}) — replication, the tuned best estimator, and the comparative analysis.
- [What a clinic would actually need]({{ '/clinical/' | relative_url }}) — conformal intervals, Bland–Altman agreement, test-time augmentation, and Grad-CAM.

<p class="next-link"><a href="{{ '/method/' | relative_url }}">Next → How the estimator is built</a></p>
