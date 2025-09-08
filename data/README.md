# data/

Canonicalized dataset for the replication.

## Provenance
- **Source:** SPLab Brno ultrasound database — <http://splab.cz/en/download/databaze/ultrasound>
- **Content:** 84 B-mode images of the common carotid artery (longitudinal section),
  10 volunteers (~8 images each), Sonix OP scanner, 10/14 MHz linear arrays, ~390×330 px.
- **Citation:** N. Patil, A. Anand, "Automated Ultrasound Doppler Angle Estimation
  Using Deep Learning," *EMBC* 2019; doi:10.1109/EMBC.2019.8857587.

## Files
| Path | What |
|---|---|
| `images/` | the 84 canonical base images (copied verbatim from `00.data/raw/`, 1:1 with `Results.txt`) |
| `labels.csv` | built by `pixi run labels`: `image_id, patient_id, theta_deg` |

`../Results.txt` (tab-separated, CRLF) is the raw ground truth: column 1 = filename,
**column 2 = θ (the Doppler angle, degrees)**; columns 3–6 are auxiliary vessel-wall
geometry recorded by the original MATLAB measurement GUI (not used for the target).
The augmented 2100-image corpus is **regenerated** by `pixi run augment`, never stored.
