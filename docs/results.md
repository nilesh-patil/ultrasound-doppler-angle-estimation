---
layout: default
title: Five backbones, two protocols, one ensemble
nav: results
toc: true
mathjax: true
scripts:
  - data.js
  - charts.js
description: "Replication and tuned results for reading the Doppler angle off a B-mode image: five frozen ImageNet backbones, image-level and patient-level sampling, and a stacked ensemble reaching 2.79% MAPE image-level and 8.53% patient-level."
---

<p class="kicker">Results</p>

# Five backbones, two protocols, one ensemble

<p class="lede">We score the same frozen-feature estimator two ways. Image-level sampling asks how well the model reads the angle across the full population of orientations the augmentation spans; patient-level sampling holds out whole volunteers and asks how well it generalizes to an unseen person. The two answers differ by roughly threefold, and that gap is the result.</p>

The headline is a tuned, stacked ensemble of five frozen ImageNet backbones: 2.79% MAPE / 1.96&deg; MAE / R&sup2; 0.995 image-level, and 8.53% MAPE / 5.93&deg; MAE / R&sup2; 0.952 patient-level. Below we build to that number from a faithful replication of the EMBC 2019 setup, tune each protocol to its own best, and then take apart what moves the error: the sampling lens, head tuning, ensembling, the choice of backbone, and a classical geometric baseline.

## Faithful replication {#replication}

We first reproduce the EMBC 2019 regime: frozen ImageNet backbones with the orientation-preserving grid-pooling head, scored under image-level sampling on the rotation-augmented corpus, with no fine-tuning.

<div class="table-wrap" role="region" aria-label="Faithful replication, image-level sampling" tabindex="0">
<table class="data">
<caption>Faithful replication under image-level sampling: frozen ImageNet backbones with the grid-pooling head, no fine-tuning.</caption>
<thead>
<tr><th scope="col">Backbone (frozen)</th><th scope="col">MAE (&deg;)</th><th scope="col">RMSE (&deg;)</th><th scope="col">MAPE (%)</th><th scope="col">R&sup2;</th></tr>
</thead>
<tbody>
<tr class="is-best"><th scope="row">DenseNet201</th><td>3.77</td><td>4.75</td><td>5.84</td><td>0.982</td></tr>
<tr><th scope="row">InceptionV3</th><td>8.36</td><td>10.44</td><td>11.70</td><td>0.911</td></tr>
<tr><th scope="row">Xception</th><td>7.91</td><td>11.20</td><td>12.00</td><td>0.898</td></tr>
<tr><th scope="row">ResNet50</th><td>8.97</td><td>11.62</td><td>12.75</td><td>0.890</td></tr>
<tr><th scope="row">VGG19</th><td>11.67</td><td>17.77</td><td>18.01</td><td>0.743</td></tr>
</tbody>
</table>
</div>

Frozen DenseNet201 reaches 5.84% MAPE / 3.77&deg; MAE, reproducing the regime of the paper's best single model (&approx; 2.87&deg; MAE / 4.03% MAPE) within tolerance and with no fine-tuning. The per-backbone ranking differs from the 2019 study, which reported VGG19 as strongest: with a shared head, pooling, and training budget across backbones, DenseNet201 lands first and VGG19 last here. The replication reproduces the single-digit-degree regime, not the original ordering.

<details class="caveat">
<summary>Why this is not yet a real-world accuracy number</summary>
This table is image-level sampling on the rotation-augmented corpus, where the 2,100 examples are coupled copies of 84 base images, not 2,100 independent observations. It measures how well the model reads the angle across orientations, not how it generalizes to an unseen patient; for that, read the patient-level column below.
</details>

## The best estimator under each protocol {#best-estimator}

Each protocol is tuned to its own best head and optimizer configuration via Optuna TPE (head depth and width, dropout, L2, BatchNorm, learning rate, batch size, patience), scored on that protocol's five-fold cross-validation over cached frozen features; a single feature extraction per backbone serves both protocols. Per-backbone rows are five-fold CV means; the ensemble rows are pooled out-of-fold (OOF) over the five tuned models.

<div class="table-wrap" role="region" aria-label="Optuna-tuned best estimator under each sampling protocol" tabindex="0">
<table class="data">
<caption>Doppler-angle estimation under two sampling protocols, each Optuna-tuned to its own best configuration.</caption>
<thead>
<tr><th scope="col">Model (Optuna-tuned)</th><th scope="col">Image MAE&deg;</th><th scope="col">Image MAPE%</th><th scope="col">Image R&sup2;</th><th scope="col">Patient MAE&deg;</th><th scope="col">Patient MAPE%</th><th scope="col">Patient R&sup2;</th></tr>
</thead>
<tbody>
<tr><th scope="row">DenseNet201</th><td>3.00</td><td>4.03</td><td>0.988</td><td>8.62</td><td>10.80</td><td>0.886</td></tr>
<tr><th scope="row">ResNet50</th><td>3.64</td><td>4.84</td><td>0.981</td><td>9.79</td><td>13.30</td><td>0.842</td></tr>
<tr><th scope="row">VGG19</th><td>4.22</td><td>5.79</td><td>0.976</td><td>10.21</td><td>14.97</td><td>0.871</td></tr>
<tr><th scope="row">InceptionV3</th><td>4.68</td><td>6.59</td><td>0.970</td><td>10.66</td><td>14.66</td><td>0.856</td></tr>
<tr><th scope="row">Xception</th><td>4.75</td><td>6.59</td><td>0.970</td><td>10.88</td><td>14.76</td><td>0.851</td></tr>
<tr><th scope="row">Ensemble (mean)</th><td>2.09</td><td>3.03</td><td>0.994</td><td>6.89</td><td>9.89</td><td>0.932</td></tr>
<tr class="is-best"><th scope="row">Ensemble (stacked)</th><td>1.96</td><td>2.79</td><td>0.995</td><td>5.93</td><td>8.53</td><td>0.952</td></tr>
</tbody>
</table>
</div>

The member rows and the ensemble rows aggregate differently: member rows are per-fold CV means, while ensemble rows are pooled OOF. Aggregated the same pooled-OOF way, the single tuned DenseNet201 is 7.80&deg; MAE / 10.14% MAPE patient-level, the like-for-like predecessor of the ensemble rows; that is the row the ensembling gain should be read against.

The scatter below pairs the stacked ensemble's predictions against the reference angle, image by image.

<figure class="fig widget" id="fig-pred-vs-actual">
  <img class="fallback" src="{{ '/assets/figures/figure4_pred_vs_actual.svg' | relative_url }}" alt="Scatter of predicted Doppler angle against reference angle for the stacked ensemble; points cluster tightly along the identity line across the full range of reference Doppler angles, roughly twenty to one hundred sixty degrees.">
  <div class="widget__canvas live" id="chart-pred-vs-actual" hidden></div>
  <figcaption><span class="fig__num">Figure 1.</span> Predicted versus reference angle for the stacked ensemble. Points hug the identity line across the full angular range; the residual scatter widens modestly at the extremes.</figcaption>
</figure>

## Two protocols, two questions {#protocols}

Holding the frozen grid-pooling model fixed and changing only the sampling protocol, every backbone loses accuracy under the stricter cross-subject split, by margins that separate the backbones cleanly.

<div class="table-wrap" role="region" aria-label="Per-backbone accuracy across image-level and patient-level sampling" tabindex="0">
<table class="data">
<caption>Per-backbone frozen grid-pooling accuracy under image-level and patient-level five-fold cross-validation.</caption>
<thead>
<tr><th scope="col">Backbone (frozen)</th><th scope="col">Image MAPE %</th><th scope="col">Image R&sup2;</th><th scope="col">Patient MAPE %</th><th scope="col">Patient R&sup2;</th></tr>
</thead>
<tbody>
<tr class="is-best"><th scope="row">DenseNet201</th><td>4.58</td><td>0.987</td><td>12.59</td><td>0.869</td></tr>
<tr><th scope="row">ResNet50</th><td>6.90</td><td>0.964</td><td>16.12</td><td>0.738</td></tr>
<tr><th scope="row">Xception</th><td>7.14</td><td>0.967</td><td>17.11</td><td>0.784</td></tr>
<tr><th scope="row">InceptionV3</th><td>7.85</td><td>0.959</td><td>17.29</td><td>0.743</td></tr>
<tr><th scope="row">VGG19</th><td>8.64</td><td>0.942</td><td>20.60</td><td>0.626</td></tr>
</tbody>
</table>
</div>

The tuned stacked ensemble reaches 2.79% MAPE / 1.96&deg; MAE under image-level sampling and 8.53% / 5.93&deg; under patient-level, roughly a 3&times; MAPE spread. This is the expected signature of a small cohort (~10 volunteers): the model interpolates the augmented-image manifold very well, while cross-subject generalization is intrinsically harder and carries genuine per-fold variance. DenseNet201 holds both the smallest protocol gap and the highest patient-level R&sup2; (0.869), which is why it is carried forward; the ordering among the weaker four shifts with the lens, while the choice of lead backbone does not.

<figure class="fig widget" id="fig-protocol-comparison">
  <img class="fallback" src="{{ '/assets/figures/figure_protocol_comparison.svg' | relative_url }}" alt="Grouped bar chart of MAPE per backbone under image-level and patient-level sampling; every backbone's patient-level bar is roughly double its image-level bar, with DenseNet201 lowest in both.">
  <div class="widget__canvas live" id="chart-protocol-comparison" hidden></div>
  <figcaption><span class="fig__num">Figure 2.</span> Image-level versus patient-level MAPE per backbone. Patient-level error is roughly double image-level for every backbone, and the spread measures how much within-population accuracy is anatomy-specific rather than a defect of either protocol.</figcaption>
</figure>

The same protocol gap shows up as a function of the true angle: error grows toward the extreme insonation angles, where the geometry is least forgiving.

<figure class="fig widget" id="fig-error-vs-angle">
  <img class="fallback" src="{{ '/assets/figures/figure5_error_vs_angle.svg' | relative_url }}" alt="Plot of absolute angle error against the reference angle; mean error per ten-degree bin is smallest in the mid-angle band and rises toward the extreme reference angles at both ends.">
  <div class="widget__canvas live" id="chart-error-vs-angle" hidden></div>
  <figcaption><span class="fig__num">Figure 3.</span> Error as a function of the reference angle. The estimator is most accurate near the center of the augmented range and degrades toward the extreme orientations.</figcaption>
</figure>

## Frozen versus tuned {#frozen-vs-tuned}

Tuning the head helps under both protocols. For DenseNet201, like-for-like under five-fold CV against the grid-pooling frozen baseline: image-level 4.58% &rarr; 4.03% MAPE and patient-level 12.59% &rarr; 10.80%. The gains are modest per model but consistent, and tuning makes the five members well-calibrated enough that a plain mean ensemble works, where an untuned mean ensemble was effectively useless.

<figure class="fig" id="fig-tuning-history">
  <img src="{{ '/assets/figures/figure_tuning_history.svg' | relative_url }}" alt="Optuna tuning-history plot showing validation MAE decreasing across trials as the search converges on better head configurations.">
  <figcaption><span class="fig__num">Figure 4.</span> Optuna tuning history. Validation error falls and then plateaus as the TPE search converges; the tuning is in-sample, so the per-model gains are read against the frozen baseline rather than as held-out improvements.</figcaption>
</figure>

## Ensembling is the biggest lever {#ensemble}

Ensembling moves the error more than any single other choice. Best single model to stacked ensemble: image-level 4.03% &rarr; 2.79% MAPE (3.00&deg; &rarr; 1.96&deg; MAE) and patient-level 10.80% &rarr; 8.53% MAPE (8.62&deg; &rarr; 5.93&deg; MAE). Read against the like-for-like pooled-OOF predecessor, the single tuned DenseNet201 at 7.80&deg; MAE / 10.14% MAPE patient-level, the genuine ensembling gain is about 1.9&deg;.

The five backbones make partly independent errors, so combining them — a plain mean, or a Ridge stacker over OOF predictions — recovers accuracy a single model cannot. Under the honest patient-level protocol, the tuned stacked ensemble is the first configuration to break below 10% MAPE (8.53% pooled OOF), improving on the untuned ensemble.

<details class="caveat">
<summary>Stacking on out-of-fold predictions can be mildly optimistic</summary>
The ensemble rows are pooled out-of-fold predictions, and the Ridge stacker is fit on those same OOF predictions. This reuse can make the stacked number mildly optimistic relative to a fully nested protocol, so the 8.53% patient-level figure is best read as an upper-bound-leaning estimate rather than a guaranteed held-out accuracy.
</details>

<figure class="fig">
  <img src="{{ '/assets/figures/figure4_pred_vs_actual.svg' | relative_url }}" alt="Predicted-versus-reference scatter for the ensemble, with most points lying on or near the identity diagonal across the full angular range.">
  <figcaption><span class="fig__num">Figure 5.</span> The ensemble's predictions against the reference. Combining five partly independent estimators tightens the scatter that a single model leaves on the table.</figcaption>
</figure>

## Newer is not better {#backbones}

A frozen bake-off under patient five-fold CV (grid pooling, untuned heads) spans the classic ImageNet encoders and the modern ConvNeXt and EfficientNet families. The classic backbones clearly beat the modern ones.

<div class="table-wrap" role="region" aria-label="Backbone bake-off by family, patient five-fold cross-validation" tabindex="0">
<table class="data">
<caption>Frozen backbone bake-off by family, patient five-fold cross-validation, grid pooling, untuned heads.</caption>
<thead>
<tr><th scope="col">Tier</th><th scope="col">Backbones</th><th scope="col">MAPE (%), patient 5-fold</th></tr>
</thead>
<tbody>
<tr class="is-best"><th scope="row">Classic ImageNet</th><td>DenseNet201 (best, 14.13), ResNet50, VGG19, InceptionV3, Xception</td><td>~14</td></tr>
<tr><th scope="row">ConvNeXt</th><td>ConvNeXt-Base (15.65), ConvNeXt-Tiny (16.07)</td><td>~16</td></tr>
<tr><th scope="row">EfficientNet / V2</th><td>B0&ndash;B3 and V2-B0&ndash;B3 families</td><td>~17&ndash;21</td></tr>
</tbody>
</table>
</div>

Within the classic group the differences sit inside the per-fold standard deviation; across families, the older encoders win. Every row in this bake-off shares one feature-extraction batch, so DenseNet201's 14.13% here is the like-for-like point against the modern backbones, whereas its replication-extraction value above is 12.59%. With only 84 base images, larger and ImageNet-stronger encoders have nothing to grip. DenseNet201 — the best R&sup2;, the replication winner, and the strongest after tuning — is carried forward.

<figure class="fig">
  <img src="{{ '/assets/figures/figure_architecture_bakeoff.svg' | relative_url }}" alt="Bar chart of patient five-fold MAPE across backbone families; DenseNet201 is lowest, the ConvNeXt pair is in the middle, and the EfficientNet and EfficientNetV2 families are highest, with error bars spanning the per-fold standard deviation.">
  <figcaption><span class="fig__num">Figure 6.</span> Backbone bake-off, patient five-fold MAPE. The classic ImageNet encoders lead the modern ConvNeXt and EfficientNet families; the interactive version of this chart lives on the <a href="{{ '/method/' | relative_url }}#grid-pooling">Method page</a>.</figcaption>
</figure>

<details class="caveat">
<summary>One backbone was extracted on a different runtime</summary>
ConvNeXt fails to run on TF-Metal because of an XLA op-support gap, so it was extracted on JAX-CPU. The bake-off otherwise shares a single feature-extraction batch per row, which is why DenseNet201's bake-off value (14.13%) and its replication-extraction value (12.59%) differ.
</details>

## Learned versus classical, and fusion {#learned-vs-classical}

A purely classical, image-only structure-tensor angle prior reaches 3.16&deg; MAE on the narrow base-angle band, and a circular fusion of the learned and classical estimates reaches 2.72&deg; MAE, improving on either alone. The learned model and a hand-crafted geometric cue capture partly complementary information.

<details class="caveat">
<summary>The fusion number is in-sample, on a narrow band of angles</summary>
The 3.16&deg; structure-tensor and 2.72&deg; fusion figures are computed in-sample on the narrow band of base angles, not on a held-out split, and they are recomputed at runtime rather than stored in <code>results/</code>. They show that classical geometry carries complementary signal, not that fusion is a validated held-out improvement.
</details>

<p class="next-link"><a href="{{ '/clinical/' | relative_url }}">Next: What a clinic would actually need &rarr;</a></p>
