#!/usr/bin/env python3
"""Export site data JSON for the project website.

Stdlib only (csv + json). Reads from results/, writes derived chart JSON to
docs/assets/data/. Errors loudly if a referenced row/column is missing so the
site can never silently fabricate a number.

This exporter owns every JSON in docs/assets/data/ EXCEPT demo_predictions.json
(owned by the demo builder). Bias / LoA / conformal / headline values are
emitted as PRECOMPUTED CONSTANTS verified against the build-brief ledger; they
come from disjoint splits and are NOT recomputed from the full OOF file.

Run: python3 scripts/export_site_data.py
"""

import csv
import json
import os
import sys

# ---------------------------------------------------------------------------
# Paths (resolved relative to repo root = parent of this script's dir)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS = os.path.join(ROOT, "results")
METRICS_CSV = os.path.join(RESULTS, "metrics.csv")
ERA_CSV = os.path.join(RESULTS, "era2019_cv.csv")
OOF_CSV = os.path.join(RESULTS, "predictions", "tuned_densenet201_oof.csv")
OUT_DIR = os.path.join(ROOT, "docs", "assets", "data")


def die(msg):
    sys.stderr.write("export_site_data.py ERROR: %s\n" % msg)
    sys.exit(1)


def read_csv(path):
    if not os.path.exists(path):
        die("missing source file: %s" % path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        die("empty source file: %s" % path)
    return rows


def to_num(value):
    """Parse a CSV cell to int or float; empty -> None."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except ValueError:
        return value
    if f == int(f) and "." not in value and "e" not in value.lower():
        return int(f)
    return f


def require_row(rows, key_field, key, label):
    for r in rows:
        if r.get(key_field) == key:
            return r
    die("%s: no row where %s == %r in source" % (label, key_field, key))


def require_field(row, field, label):
    if field not in row or row[field] == "" or row[field] is None:
        die("%s: missing/empty column %r in row %r" % (label, field, row))
    return row[field]


def r2(x):
    """Round display values to <= 2 decimals."""
    return round(float(x), 2)


def write_json(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# 1. metrics.json  (clean JSON array of every metrics.csv row, numbers typed)
# ---------------------------------------------------------------------------
def export_metrics(metrics_rows):
    NUMERIC = {"seed", "n_test", "mae", "rmse", "me", "mape", "r2"}
    out = []
    for r in metrics_rows:
        rec = {}
        for k, v in r.items():
            rec[k] = to_num(v) if k in NUMERIC else v
        out.append(rec)
    return write_json("metrics.json", out)


# ---------------------------------------------------------------------------
# 2. headline.json  (verified ledger constants)
# ---------------------------------------------------------------------------
def export_headline():
    obj = {
        "image_level": {"mape": 2.79, "mae": 1.96, "r2": 0.995},
        "patient_level": {"mape": 8.53, "mae": 5.93, "r2": 0.952},
        "paper_best": {"mae": 2.87, "mape": 4.03},
        "single_tuned_densenet201_image": {"mae": 3.00, "mape": 4.03, "r2": 0.988},
    }
    return write_json("headline.json", obj)


# ---------------------------------------------------------------------------
# 3. protocol_comparison.json  (cmp_<backbone>_<protocol> rows from era CSV)
# ---------------------------------------------------------------------------
def export_protocol_comparison(era_rows):
    # backbone -> (label, lead)
    BACKBONES = [
        ("densenet201", "DenseNet201", True),
        ("xception", "Xception", False),
        ("resnet50", "ResNet50", False),
        ("inceptionv3", "InceptionV3", False),
        ("vgg19", "VGG19", False),
    ]
    out_backbones = []
    for bb, label, lead in BACKBONES:
        entry = {"backbone": bb, "label": label, "lead": lead}
        for proto in ("image", "patient"):
            name = "cmp_%s_%s" % (bb, proto)
            row = require_row(era_rows, "name", name, "protocol_comparison")
            entry[proto] = {
                "mape": r2(require_field(row, "mape_mean", name)),
                "mae": r2(require_field(row, "mae_mean", name)),
                "r2": round(float(require_field(row, "r2_mean", name)), 3),
                "mape_std": r2(require_field(row, "mape_std", name)),
            }
        out_backbones.append(entry)
    # Sort ascending by patient MAPE.
    out_backbones.sort(key=lambda e: e["patient"]["mape"])
    obj = {"metric_default": "mape", "backbones": out_backbones}
    return write_json("protocol_comparison.json", obj)


# ---------------------------------------------------------------------------
# 4. architecture_bakeoff.json  (patient 5-fold CV-mean bake-off)
# ---------------------------------------------------------------------------
def export_architecture_bakeoff(era_rows):
    # (era CSV name, display label, tier, lead)
    ROWS = [
        ("base_densenet201", "DenseNet201", "Classic ImageNet", True),
        ("f_convnext_base", "ConvNeXt-Base", "ConvNeXt", False),
        ("f_convnext_tiny", "ConvNeXt-Tiny", "ConvNeXt", False),
        ("f_effv2b1", "EfficientNetV2-B1", "EfficientNet / V2", False),
        ("f_effv2b2", "EfficientNetV2-B2", "EfficientNet / V2", False),
        ("e2_effb0", "EfficientNet-B0", "EfficientNet / V2", False),
        ("f_effv2b0", "EfficientNetV2-B0", "EfficientNet / V2", False),
        ("e2_effb3", "EfficientNet-B3", "EfficientNet / V2", False),
        ("e2_effb1", "EfficientNet-B1", "EfficientNet / V2", False),
        ("f_effv2b3", "EfficientNetV2-B3", "EfficientNet / V2", False),
        ("e2_effb2", "EfficientNet-B2", "EfficientNet / V2", False),
    ]
    out_rows = []
    for name, label, tier, lead in ROWS:
        row = require_row(era_rows, "name", name, "architecture_bakeoff")
        bb = require_field(row, "backbone", name)
        out_rows.append({
            "backbone": bb,
            "label": label,
            "tier": tier,
            "mape": r2(require_field(row, "mape_mean", name)),
            "mape_std": r2(require_field(row, "mape_std", name)),
            "mae": r2(require_field(row, "mae_mean", name)),
            "r2": round(float(require_field(row, "r2_mean", name)), 3),
            "lead": lead,
        })
    # Sort ascending by MAPE (best/lead first).
    out_rows.sort(key=lambda r: r["mape"])
    obj = {"reference_backbone": "densenet201", "rows": out_rows}
    return write_json("architecture_bakeoff.json", obj)


# ---------------------------------------------------------------------------
# OOF helpers (shared by pred_vs_actual + bland_altman)
# ---------------------------------------------------------------------------
def load_oof():
    rows = read_csv(OOF_CSV)
    needed = {"image_id", "patient_id", "rotation_deg", "theta_true", "theta_pred"}
    missing = needed - set(rows[0].keys())
    if missing:
        die("OOF file missing columns: %s" % sorted(missing))
    out = []
    for r in rows:
        out.append({
            "image_id": r["image_id"],
            "patient": int(float(r["patient_id"])),
            "rot": int(float(r["rotation_deg"])),
            "true": float(r["theta_true"]),
            "pred": float(r["theta_pred"]),
        })
    return out


def signed_err(pred, true):
    """Signed error = theta_pred - theta_true (the sole permitted recompute)."""
    return pred - true


# ---------------------------------------------------------------------------
# 5. pred_vs_actual.json  (2100 OOF points + 10-deg error bins + constants)
# ---------------------------------------------------------------------------
def export_pred_vs_actual(oof):
    points = []
    for r in oof:
        err = signed_err(r["pred"], r["true"])
        points.append({
            "id": r["image_id"],
            "patient": r["patient"],
            "rot": r["rot"],
            "true": round(r["true"], 4),
            "pred": round(r["pred"], 4),
            "err": round(err, 4),
            "base": r["rot"] == 0,
        })
    if len(points) != 2100:
        die("pred_vs_actual: expected 2100 OOF points, got %d" % len(points))

    # 10-degree error bins over theta_true range.
    bins = []
    lo = 0
    while lo < 90:
        hi = lo + 10
        member_errs = [p["err"] for p in points if lo <= p["true"] < hi]
        if member_errs:
            bins.append({
                "lo": lo,
                "hi": hi,
                "mean_err": round(sum(member_errs) / len(member_errs), 4),
                "n": len(member_errs),
            })
        lo = hi

    obj = {
        "conformal_halfwidth": 20.50,
        "identity": True,
        "bias_line": -4.31,
        "points": points,
        "error_bins": bins,
    }
    return write_json("pred_vs_actual.json", obj)


# ---------------------------------------------------------------------------
# 6. bland_altman.json  (per-sample ~600 downsample + per-patient; precomputed
#    bias/LoA constants from ledger)
# ---------------------------------------------------------------------------
def export_bland_altman(oof):
    # Per-sample points: mean = (pred+true)/2, diff = method - reference = pred - true.
    all_points = []
    for r in oof:
        diff = signed_err(r["pred"], r["true"])
        mean = (r["pred"] + r["true"]) / 2.0
        all_points.append({
            "mean": round(mean, 4),
            "diff": round(diff, 4),
            "id": r["image_id"],
            "rot": r["rot"],
        })

    # Downsample to ~600: keep ALL 84 base (rot==0) + a stratified sample of the
    # rotated points (deterministic stride, no RNG dependency).
    base = [p for p in all_points if p["rot"] == 0]
    rotated = [p for p in all_points if p["rot"] != 0]
    target_rotated = 600 - len(base)
    if target_rotated < 0:
        target_rotated = 0
    if rotated and target_rotated > 0:
        stride = max(1, len(rotated) // target_rotated)
        sampled = rotated[::stride][:target_rotated]
    else:
        sampled = []
    per_sample_points = base + sampled

    # Per-patient points: aggregate the 12 patient-proxy groups.
    by_patient = {}
    for r in oof:
        by_patient.setdefault(r["patient"], []).append(r)
    per_patient_points = []
    for pid in sorted(by_patient):
        recs = by_patient[pid]
        mean = sum((x["pred"] + x["true"]) / 2.0 for x in recs) / len(recs)
        diff = sum(signed_err(x["pred"], x["true"]) for x in recs) / len(recs)
        per_patient_points.append({
            "patient": pid,
            "mean": round(mean, 4),
            "diff": round(diff, 4),
        })
    if len(per_patient_points) != 12:
        die("bland_altman: expected 12 patient groups, got %d"
            % len(per_patient_points))

    obj = {
        "per_sample": {
            "bias": -4.31,
            "loa_lo": -24.25,
            "loa_hi": 15.63,
            "n": 2100,
            "points": per_sample_points,
        },
        "per_patient": {
            "bias": -4.56,
            "loa_lo": -16.87,
            "loa_hi": 7.75,
            "n": 12,
            "points": per_patient_points,
        },
    }
    return write_json("bland_altman.json", obj)


# ---------------------------------------------------------------------------
# 7. conformal.json  (3 levels; verified ledger constants from disjoint split)
# ---------------------------------------------------------------------------
def export_conformal():
    obj = {
        "levels": [
            {"nominal": 0.80, "halfwidth": 15.01, "empirical": 0.897},
            {"nominal": 0.90, "halfwidth": 20.50, "empirical": 0.952},
            {"nominal": 0.95, "halfwidth": 26.03, "empirical": 0.978},
        ]
    }
    return write_json("conformal.json", obj)


# ---------------------------------------------------------------------------
def main():
    metrics_rows = read_csv(METRICS_CSV)
    era_rows = read_csv(ERA_CSV)
    oof = load_oof()

    written = []
    written.append(export_metrics(metrics_rows))
    written.append(export_headline())
    written.append(export_protocol_comparison(era_rows))
    written.append(export_architecture_bakeoff(era_rows))
    written.append(export_pred_vs_actual(oof))
    written.append(export_bland_altman(oof))
    written.append(export_conformal())

    sys.stdout.write("Wrote %d JSON files:\n" % len(written))
    for p in written:
        sys.stdout.write("  %s\n" % p)


if __name__ == "__main__":
    main()
