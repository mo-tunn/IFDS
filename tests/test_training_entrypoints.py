"""
@file test_training_entrypoints.py
@brief Tests for optional training entrypoints using lightweight fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.training.train_lstm as train_lstm_module
import src.training.train_xception as train_xception_module


class FakeEfficientNetModel:
    def build_model(self, imagenet_weights: bool = False):
        assert imagenet_weights is True
        return type("BuiltModel", (), {"name": "fake_efficientnet"})()


def test_train_lstm_placeholder_builds_model_and_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(train_lstm_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_lstm_module, "EfficientNetForensicModel", lambda: FakeEfficientNetModel())

    with pytest.raises(NotImplementedError, match="fake_efficientnet"):
        train_lstm_module.train()

    assert (tmp_path / "models").exists()


class FakeDatasetBuilder:
    def build_tf_datasets(self):
        return "train", "val", "test"


class FakeXceptionWrapper:
    def build_model(self, imagenet_weights: bool = False):
        assert imagenet_weights is True
        return FakeTrainableModel()


class FakeTrainableModel:
    def __init__(self) -> None:
        self.fit_args = None
        self.saved_to = None
        self.evaluated_on = None

    def fit(self, train_ds, validation_data, epochs: int) -> None:
        self.fit_args = (train_ds, validation_data, epochs)

    def save(self, path: str) -> None:
        self.saved_to = path

    def evaluate(self, test_ds) -> None:
        self.evaluated_on = test_ds


def test_train_xception_runs_training_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, FakeTrainableModel] = {}

    def wrapper_factory() -> FakeXceptionWrapper:
        wrapper = FakeXceptionWrapper()
        original_build = wrapper.build_model

        def build_model(imagenet_weights: bool = False):
            model = original_build(imagenet_weights)
            captured["model"] = model
            return model

        wrapper.build_model = build_model
        return wrapper

    monkeypatch.setattr(train_xception_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_xception_module, "XCEPTION_MODEL_PATH", tmp_path / "models" / "xception.h5")
    monkeypatch.setattr(train_xception_module, "EPOCHS_XCEPTION", 3)
    monkeypatch.setattr(train_xception_module, "ForensicDatasetBuilder", lambda: FakeDatasetBuilder())
    monkeypatch.setattr(train_xception_module, "XceptionForensicModel", wrapper_factory)

    train_xception_module.train()

    model = captured["model"]
    assert model.fit_args == ("train", "val", 3)
    assert model.saved_to.endswith("xception.h5")
    assert model.evaluated_on == "test"
