"""Typed, validated experiment configuration — the single source of truth.

Loaded from YAML and fully validated: unknown keys are rejected and ranges are
checked. Imported by every other ``uda`` module so the whole project shares one schema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    """Strict base: reject unknown keys so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    images_dir: Path = Path("data/images")
    labels_csv: Path = Path("data/labels.csv")
    rotation_min_deg: int = -60
    rotation_max_deg: int = 60
    rotation_step_deg: int = 5  # -> 25 rotations per image at the [-60, 60] span
    rotation_mode: Literal["reflect", "constant", "nearest", "wrap"] = "reflect"
    clahe: bool = True
    clahe_backend: Literal["skimage", "opencv"] = "skimage"
    clahe_clip_limit: float = 0.03
    normalize: bool = True
    wrap_0_180: bool = True
    # Richer augmentation. All default OFF so the baseline corpus is
    # byte-for-byte unchanged; applied AFTER rotation+CLAHE+normalize in
    # uda.data.augment.augment_image.
    flip_h: bool = False  # horizontal (left-right) mirror; angle-preserving
    flip_v: bool = False  # vertical (up-down) mirror; angle-preserving
    gamma_jitter: float = 0.0  # max |log-gamma| jitter; gamma drawn per-call, 0=off
    translate_frac: float = 0.0  # max shift as a fraction of H/W, in [0, 1); 0=off
    speckle_std: float = 0.0  # std of additive Gaussian speckle, in [0, 1); 0=off

    @field_validator("rotation_step_deg")
    @classmethod
    def _step_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("rotation_step_deg must be > 0")
        return v

    @field_validator("gamma_jitter")
    @classmethod
    def _gamma_jitter_nonneg(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("gamma_jitter must be >= 0")
        return v

    @field_validator("translate_frac")
    @classmethod
    def _translate_frac_unit(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError("translate_frac must be in [0, 1)")
        return v

    @field_validator("speckle_std")
    @classmethod
    def _speckle_std_unit(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError("speckle_std must be in [0, 1)")
        return v

    @model_validator(mode="after")
    def _check_span(self) -> "DataConfig":
        if self.rotation_min_deg > self.rotation_max_deg:
            raise ValueError("rotation_min_deg must be <= rotation_max_deg")
        return self

    @property
    def n_rotations(self) -> int:
        span = self.rotation_max_deg - self.rotation_min_deg
        return span // self.rotation_step_deg + 1


class SplitConfig(_Base):
    # "augmented" = the paper's protocol (random 80/20 over the 2100 augmented
    # images; rotated copies of a base image leak across the split -> reproduces
    # Table I). "image" = split base images then augment (no rotated-copy leak).
    # "patient" = split by volunteer (leakage-free).
    strategy: Literal["image", "patient", "augmented"] = "image"
    test_size: float = 0.2
    n_folds: Optional[int] = None
    seed: int = 42

    @field_validator("test_size")
    @classmethod
    def _test_size_unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        return v


class BackboneConfig(_Base):
    name: Literal[
        "vgg19",
        "resnet50",
        "densenet201",
        "xception",
        "inceptionv3",
        # EfficientNet B0-B3 (ImageNet) + a from-scratch CNN.
        "efficientnetb0",
        "efficientnetb1",
        "efficientnetb2",
        "efficientnetb3",
        # Modern backbones (ConvNeXt + EfficientNetV2).
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "efficientnetv2b0",
        "efficientnetv2b1",
        "efficientnetv2b2",
        "efficientnetv2b3",
        "cnn_scratch",
    ]
    weights: Optional[Literal["imagenet"]] = "imagenet"
    trainable: bool = False
    # avg/max = global pooling (rotation-invariant); avgmax = their concat;
    # grid2/grid3 = spatial-pyramid avg-pool to G×G then flatten (keeps orientation).
    pooling: Literal["avg", "max", "none", "avgmax", "grid2", "grid3"] = "avg"


class HeadConfig(_Base):
    hidden_units: list[int] = Field(default_factory=lambda: [256])
    dropout: float = 0.5
    batchnorm: bool = True
    activation: str = "relu"
    final_activation: Literal["linear", "relu"] = "linear"
    l2: float = 0.0  # L2 weight decay on Dense kernels (regularizes the head)


class TargetConfig(_Base):
    kind: Literal["raw", "sincos2theta"] = "raw"


class TrainConfig(_Base):
    optimizer: Literal["adam"] = "adam"
    lr: float = 1e-4
    loss: Literal["mse"] = "mse"
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    monitor: str = "val_mae"
    # Monte-Carlo dropout passes at eval. 0 = off (point predictions,
    # current behavior); >0 = that many stochastic forward passes to produce a
    # predictive mean + std.
    mc_samples: int = 0

    @field_validator("mc_samples")
    @classmethod
    def _mc_samples_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("mc_samples must be >= 0")
        return v


class ExperimentConfig(_Base):
    name: str
    seed: int = 42
    era: Literal["replication", "era2019", "modern"] = "replication"
    backbone: BackboneConfig
    head: HeadConfig = Field(default_factory=HeadConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate a YAML experiment config."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(raw)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Serialize a config back to YAML (JSON-mode dump makes Paths strings)."""
    data = cfg.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
