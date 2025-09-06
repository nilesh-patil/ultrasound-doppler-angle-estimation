"""Tests — config schema, validation, round-trip."""
import pytest
from pydantic import ValidationError

from uda.config import DataConfig, ExperimentConfig, dump_config, load_config


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(name="t", backbone={"name": "vgg19"})


def test_defaults_and_nesting():
    c = _cfg()
    assert c.backbone.name == "vgg19"
    assert c.backbone.trainable is False
    assert c.data.clahe_backend == "skimage"
    assert c.split.strategy == "image"
    assert c.target.kind == "raw"
    assert c.train.lr == 1e-4


def test_n_rotations_default_is_25():
    assert DataConfig().n_rotations == 25


def test_round_trip(tmp_path):
    c = _cfg()
    p = tmp_path / "exp.yaml"
    dump_config(c, p)
    assert load_config(p) == c


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig(name="t", backbone={"name": "vgg19"}, bogus=1)


def test_bad_backbone_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig(name="t", backbone={"name": "alexnet"})


def test_bad_test_size_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig(name="t", backbone={"name": "vgg19"}, split={"test_size": 1.5})


def test_rotation_step_must_be_positive():
    with pytest.raises(ValidationError):
        DataConfig(rotation_step_deg=0)
