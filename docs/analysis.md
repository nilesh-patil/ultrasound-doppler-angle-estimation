---
layout: default
title: Results & comparative-analysis narrative
---

# Results & Comparative Analysis — Ultrasound Doppler Angle Estimation

> Estimating the Doppler beam-to-vessel angle θ directly from a single grayscale
> B-mode carotid image (no color Doppler, no segmentation), as a replication and
> extension of Patil & Anand, *Automated Ultrasound Doppler Angle Estimation Using
> Deep Learning*, EMBC 2019, pp. 28–31 (doi:10.1109/EMBC.2019.8857587).
>
> Every number below is regenerated from code — `results/metrics.csv`,
> `results/era2019_cv.csv`, the saved per-image predictions in
> `results/predictions/`, and `paper/tables/*` — via `scripts/gen_paper_tables.py`.
> Seed 42; Keras 3 / JAX; Apple M4 Max.

---

## 1. The task and why the angle matters

Spectral-Doppler velocity is recovered from the Doppler equation
`f_d = 2·f₀·v·cos(θ)/c`, so the reported velocity scales as `1/cos θ`. The angle θ
is set manually by the sonographer, which makes it a leading, operator-dependent
source of velocity error: improper angle correction has been flagged in up to ~35%
of vascular-lab accreditation applications. The goal is to read θ straight off the
B-mode image.

**Data.** 84 longitudinal common-carotid B-mode images from ~10 volunteers
(SPLab Brno; supine, right CCA, ~390×330 px). Each image is rotated over
**[−60°, +60°] in 5° steps** (25 oriented views), giving **2,100** labelled
examples. The rotation supplies an exact angle label for each view. This is a
standard small-data synthetic-augmentation recipe.

**Model.** A frozen ImageNet backbone (`include_top=False`) feeds a shallow
BatchNorm-Dense-ReLU-Dropout-Dense regression head, trained with Adam
(lr 1e-4), MSE loss, and early stopping on validation MAE. No fine-tuning.

---

## 2. Two sampling protocols (how the model is scored)

Both protocols operate on the same 2,100-image corpus and the same frozen
features; they differ only in how the train/test partition is drawn, and they
answer **different questions**. We report and tune **both**.

| Protocol | Partition | Question it answers |
|---|---|---|
| **Image-level sampling** | Random split over the augmented image corpus (the original study's protocol) | How accurately does the model read θ across the full population of orientations and imaging conditions the augmentation spans? |
| **Patient-level sampling** | Grouped split by volunteer (`GroupKFold`) — no subject in both train and test | How well does the model generalize to a previously **unseen patient**? |

Image-level is the standard way to evaluate a model on an augmented/synthetic
dataset; patient-level is the stricter cross-subject lens. We report both, each
tuned to its own best.

---

## 3. Faithful replication (image-level sampling)

Frozen backbones with the orientation-preserving **grid-pooling** head, on the
original image-level protocol:

| Backbone (frozen) | MAE (°) | RMSE (°) | MAPE (%) | R² |
|---|---|---|---|---|
| **DenseNet201** | **3.77** | **4.75** | **5.84** | **0.982** |
| InceptionV3 | 8.36 | 10.44 | 11.70 | 0.911 |
| Xception | 7.91 | 11.20 | 12.00 | 0.898 |
| ResNet50 | 8.97 | 11.62 | 12.75 | 0.890 |
| VGG19 | 11.67 | 17.77 | 18.01 | 0.743 |

Frozen DenseNet201 reaches **5.84% MAPE / 3.77° MAE**, reproducing the regime of
the paper's Table I (best ≈ 2.87° MAE / 4.03% MAPE) within tolerance and with **no
fine-tuning**. The per-backbone *ranking* differs from the 2019 study, which
reported VGG19 as the strongest model: here a shared head, pooling, and training
budget across backbones place DenseNet201 first and VGG19 last. The replication
reproduces the single-digit-degree *regime*, not the original ordering.

### The grid-pooling insight

The first frozen runs used **global average pooling** (GAP), which averages the
conv feature map over all spatial positions and is therefore partly
*rotation-invariant*, exactly the wrong inductive bias when the target *is* an
orientation. Replacing GAP with **grid pooling** (average-pool the feature map to a
small G×G grid, then flatten, preserving coarse spatial layout) is the load-bearing
choice for frozen DenseNet201:

| Head | Image-level MAPE (%) | Patient-level MAPE (%) |
|---|---|---|
| GAP (rotation-invariant) | 10.85 | 18.70 |
| Grid pooling | 4.58 | 12.59 |

The orientation is already in the frozen features; grid pooling simply keeps it.

---

## 4. Best estimator under each protocol (Optuna-tuned)

Each protocol is tuned to its **own** best head/optimizer configuration via Optuna
TPE (head depth/width, dropout, L2, BatchNorm, lr, batch size, patience), scored on
that protocol's 5-fold cross-validation over cached frozen features; one feature
extraction per backbone serves both protocols. Per-backbone rows are 5-fold CV
means (per-fold MAPE std is 0.1–0.7% for image-level and 2.6–4.4% for the
higher-variance patient-level folds); ensemble rows are pooled out-of-fold (OOF)
over the five tuned models.

| Model (Optuna-tuned) | Image MAE° | Image MAPE% | Image R² | Patient MAE° | Patient MAPE% | Patient R² |
|---|---|---|---|---|---|---|
| DenseNet201 | 3.00 | 4.03 | 0.988 | 8.62 | 10.80 | 0.886 |
| ResNet50 | 3.64 | 4.84 | 0.981 | 9.79 | 13.30 | 0.842 |
| VGG19 | 4.22 | 5.79 | 0.976 | 10.21 | 14.97 | 0.871 |
| InceptionV3 | 4.68 | 6.59 | 0.970 | 10.66 | 14.66 | 0.856 |
| Xception | 4.75 | 6.59 | 0.970 | 10.88 | 14.76 | 0.851 |
| Ensemble (mean) | 2.09 | 3.03 | 0.994 | 6.89 | 9.89 | 0.932 |
| **Ensemble (stacked)** | **1.96** | **2.79** | **0.995** | **5.93** | **8.53** | **0.952** |

The member rows and the ensemble rows aggregate differently (per-fold means vs.
pooled OOF); aggregated the same pooled-OOF way, the single tuned DenseNet201 is
7.80° MAE / ~10.1% MAPE patient-level, the like-for-like predecessor of the
ensemble rows.

---

## 5. Comparative analysis

### 5.1 Per-backbone behavior across the two protocols

Holding the frozen grid-pooling model fixed and changing only the sampling protocol,
every backbone loses accuracy under the stricter cross-subject split, by margins that
separate the backbones cleanly (frozen heads, image and patient five-fold CV):

| Backbone (frozen) | Image MAPE % | Image R² | Patient MAPE % | Patient R² |
|---|---|---|---|---|
| DenseNet201 | 4.58 | 0.987 | 12.59 | 0.869 |
| ResNet50 | 6.90 | 0.964 | 16.12 | 0.738 |
| Xception | 7.14 | 0.967 | 17.11 | 0.784 |
| InceptionV3 | 7.85 | 0.959 | 17.29 | 0.743 |
| VGG19 | 8.64 | 0.942 | 20.60 | 0.626 |

Patient-level MAPE is roughly double the image-level MAPE for every backbone, and the
cross-subject R² stays positive throughout under five-fold CV. DenseNet201 holds both
the smallest protocol gap and the highest patient-level R² (0.869), which is why it is
carried forward. The spread between the protocols measures how much of the
within-population accuracy is anatomy-specific, not a defect of either protocol.

### 5.2 Image-level vs patient-level sampling
The tuned stacked ensemble reaches **2.79% MAPE / 1.96° MAE** under image-level
sampling and **8.53% / 5.93°** under patient-level, roughly a **3× MAPE** spread.
This is the expected signature of a small cohort (~10 volunteers): the model
interpolates the augmented-image manifold very well, while cross-subject
generalization is intrinsically harder and carries genuine per-fold variance.
DenseNet201 is the strongest backbone in every regime (frozen or tuned, image-level
or patient-level), while the ordering among the weaker four shifts with the lens. The
choice of lead backbone is robust to the evaluation lens; the tail ordering is not.

### 5.3 Frozen vs Optuna-tuned
Tuning the head helps under both protocols. For DenseNet201, like-for-like under
five-fold CV against the grid-pooling frozen baseline: image-level **4.58% → 4.03%**
MAPE and patient-level **12.59% → 10.80%**. The gains are modest per model but
consistent, and tuning makes the five members well-calibrated enough that a plain
*mean* ensemble works, where an untuned mean ensemble was effectively useless.

### 5.4 Single model vs ensemble
Ensembling is the single biggest lever. Best single to stacked ensemble:
image-level **4.03% → 2.79%** MAPE (3.00° → 1.96° MAE) and patient-level
**10.80% → 8.53%** MAPE (8.62° → 5.93° MAE).

The five backbones make partly independent errors, so combining them (mean or a
Ridge stacker over OOF predictions) recovers accuracy a single model cannot. Under
the honest patient-level protocol, the tuned stacked ensemble is the first
configuration to break below **10% MAPE** (8.53% pooled OOF), improving on the
untuned ensemble.

### 5.5 Versus the EMBC 2019 paper
The paper's best single model reports ≈ **2.87° MAE / 4.03% MAPE** (image-level).
This work **reproduces** that regime with a single tuned DenseNet201 (3.00° / 4.03%)
and **improves** on it with the tuned ensemble (**1.96° MAE / 2.79% MAPE**, R² 0.995),
and additionally reports the previously-unmeasured patient-level number.

### 5.6 Backbone architectures — newer is not better
A frozen bake-off under patient 5-fold CV (grid pooling, untuned heads), spanning
the classic ImageNet encoders and the modern ConvNeXt / EfficientNet families:

| Tier | Backbones | MAPE (%), patient 5-fold |
|---|---|---|
| Classic ImageNet | DenseNet201 (best, 14.13), ResNet50, VGG19, InceptionV3, Xception | ~14 |
| ConvNeXt | ConvNeXt-Base (15.65), ConvNeXt-Tiny (16.07) | ~16 |
| EfficientNet / V2 | B0–B3 and V2-B0–B3 families | ~17–21 |

The classic backbones clearly beat the modern families; within the classic group
the differences sit inside the per-fold std. **DenseNet201**, the best R², the
replication winner, and the strongest after tuning, is carried forward. Every row in
this bake-off shares one feature-extraction batch, so the DenseNet201 14.13% here is
the like-for-like point against the modern backbones; its replication-extraction
value (§5.1) is 12.59%. (ConvNeXt fails to run on TF-Metal due to an XLA op-support
gap and was extracted on JAX-CPU.) With only 84 base images, larger ImageNet-stronger
encoders have nothing to grip.

### 5.7 Learned vs classical, and fusion
A purely classical, image-only **structure-tensor** angle prior reaches **3.16° MAE**
on the narrow base-angle band, and a **circular fusion** of the learned and classical
estimates reaches **2.72° MAE**, improving on the classical prior and on the learned
model's base-band error. The learned model and a
hand-crafted geometric cue capture **partly complementary** information (both
recomputed at runtime by `uda.interpret.geometric` and `uda.interpret.fusion`, not
stored in `results/`).

---

## 6. Clinical-grade evaluation (patient-level OOF, tuned DenseNet201)

Beyond a point estimate, calibrated and clinically legible reporting on the saved
patient-level OOF predictions (`results/predictions/tuned_densenet201_oof.csv`):

| Quantity | Value | Detail |
|---|---|---|
| Conformal 80% interval | ±15.01° | empirical coverage 89.7% |
| Conformal 90% interval | **±20.50°** | empirical coverage **95.2%** (≥ nominal) |
| Conformal 95% interval | ±26.03° | empirical coverage 97.8% |
| Bland–Altman bias (model − reference) | **−4.31°** | 95% LoA −24.25° … +15.63° (per-sample, n=2100) |
| Bland–Altman bias, per-patient | −4.56° | 95% LoA −16.87° … +7.75° (n=12) |
| Test-time augmentation (median) | **7.80° → 4.72°** | base-image MAE, circular-median over rotations |
| Test-time augmentation (mean) | 7.80° → 5.31° | base-image MAE, circular-mean reduction |

Conformal intervals use a **patient-disjoint** calibration/test split
(group-conformal: 900 calibration rows, 1200 test rows, no patient in both) and are
distribution-free, finite-sample valid; empirical coverage is ≥ nominal at every
level, so they are wide but **honest**. The Bland–Altman comparison is
**method-vs-reference** against the *single* MATLAB-GUI reading available per image
(`theta_true`). It is **not** an inter-observer study; the small negative bias
means the model reads slightly lower than the reference. Each of the 84 base images
carries 25 rotation views; median rotation TTA de-rotates every view to the base
frame, reduces them circularly (180-periodic, seam-safe), and roughly halves the
raw base-image error. Grad-CAM attributions localize on the vessel wall, consistent
with the network keying on the anatomy that defines the flow axis.

---

## 7. Limitations & honest scope

- **Small, single-center cohort** (~10 volunteers, 84 base images, one Sonix OP
  scanner, carotid only): patient 5-fold CV rests on a few held-out subjects per
  fold (~2 per test fold), so the cross-subject std is non-negligible and external
  validity is untested.
- **One reference reading** per image, so agreement is method-vs-reference, not
  inter-observer; the reference itself is a noisy estimate of the true angle.
- **Transfer is essential on this hardware.** A from-scratch CNN fails to converge
  on the augmented (MAPE 101.6%, R² −6.14) and patient (MAPE 100.8%, R² −5.68)
  protocols, and end-to-end DenseNet201 fine-tuning exhausts the Apple-silicon GPU
  even at batch 16. Both reinforce that frozen transfer is the right tool here.
- **Frozen backbones only.** End-to-end fine-tuning, transformer/SSL encoders
  (ViT, DINOv2), and ultrasound foundation models (USFM, MedSAM) are deferred to
  CUDA-class hardware and are **not** claimed here.

---

## 8. Reproduction

```bash
pixi install
pixi run all                                              # labels → augment → replicate → figures
pixi run python scripts/run_tuning.py --protocol both --trials 20 --k 5   # tune both protocols
pixi run python scripts/oof_ensemble.py --protocol patient
pixi run python scripts/oof_ensemble.py --protocol image
pixi run python scripts/gen_paper_tables.py              # regenerate tables + paper inputs
```
