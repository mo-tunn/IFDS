"""
@file test_ai_models.py
@brief Tests for AI wrappers without real TensorFlow weights.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.ai_models.efficientnet_lstm import EfficientNetLSTMModel
from src.ai_models.efficientnet_model import EfficientNetForensicModel
from src.ai_models.xception_model import XceptionForensicModel


class DummyXceptionBackend:
    """@brief Minimal backend exposing Keras-like predict."""

    def predict(self, batch, verbose: int = 0):
        return np.array([[0.82]], dtype=np.float32)


class DummyEfficientNetBackend:
    """@brief Minimal backend exposing Keras-like predict."""

    def predict(self, batch, verbose: int = 0):
        return np.array([[0.91]], dtype=np.float32)


def test_xception_predict_with_mock_backend() -> None:
    model = XceptionForensicModel()
    model.model = DummyXceptionBackend()
    image = np.zeros((224, 224, 3), dtype=np.float32)

    result = model.predict(image)

    assert result.model_name == "Xception"
    assert result.is_forged is True
    assert result.class_label == "Tampered"
    assert result.confidence > 0.8


def test_efficientnet_predict_with_mock_backend() -> None:
    model = EfficientNetForensicModel()
    model.model = DummyEfficientNetBackend()
    image = np.zeros((224, 224, 3), dtype=np.float32)

    result = model.predict(image)

    assert result.model_name == "EfficientNet CNN"
    assert result.is_forged is True
    assert result.class_label == "Tampered"
    assert result.confidence > 0.9


def test_predict_requires_loaded_model() -> None:
    with pytest.raises(RuntimeError, match="Xception modeli yüklenmedi"):
        XceptionForensicModel().predict(np.zeros((224, 224, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="EfficientNet modeli yüklenmedi"):
        EfficientNetForensicModel().predict(np.zeros((224, 224, 3), dtype=np.float32))


class FakeCallableLayer:
    def __init__(self, name: str = "layer") -> None:
        self.name = name
        self.trainable = True

    def __call__(self, inputs=None, *args, **kwargs):
        return inputs if inputs is not None else self


class FakeLayers(types.SimpleNamespace):
    class BatchNormalization(FakeCallableLayer):
        pass

    def __getattr__(self, name: str):
        if name == "BatchNormalization":
            return self.BatchNormalization
        return lambda *args, **kwargs: FakeCallableLayer(kwargs.get("name", name))


class FakeOptimizers:
    class Adam:
        def __init__(self, learning_rate: float) -> None:
            self.learning_rate = learning_rate


class FakeRegularizers:
    @staticmethod
    def l2(value: float) -> tuple[str, float]:
        return ("l2", value)


class FakeMetrics:
    class AUC:
        def __init__(self, name: str) -> None:
            self.name = name


class FakeBackbone:
    def __init__(self, name: str = "fake_backbone") -> None:
        self.name = name
        self.trainable = False
        self.layers = [FakeCallableLayer(f"layer_{index}") for index in range(25)]
        self.layers[-1] = FakeLayers.BatchNormalization("bn")

    def __call__(self, inputs=None, *args, **kwargs):
        return inputs


class FakeCompiledModel:
    def __init__(self, inputs=None, outputs=None, name: str = "fake_model") -> None:
        self.inputs = inputs
        self.output = outputs
        self.name = name
        self.compiled = False
        self.loaded_weights_path = None
        self.backbone = FakeBackbone("xception_backbone")

    def compile(self, **kwargs) -> None:
        self.compiled = True
        self.compile_kwargs = kwargs

    def load_weights(self, path: str) -> None:
        self.loaded_weights_path = path

    def get_layer(self, name: str):
        if name == "xception_backbone":
            return self.backbone
        raise ValueError(name)


def install_fake_tensorflow(monkeypatch: pytest.MonkeyPatch, fail_imagenet: bool = False) -> types.ModuleType:
    tensorflow = types.ModuleType("tensorflow")
    keras = types.ModuleType("tensorflow.keras")
    applications = types.ModuleType("tensorflow.keras.applications")
    models = types.SimpleNamespace(load_model=lambda path, compile=False: FakeCompiledModel(name="loaded_model"))

    def input_factory(*args, **kwargs):
        return {"shape": kwargs.get("shape"), "name": kwargs.get("name")}

    def xception_factory(*args, **kwargs):
        return FakeBackbone(kwargs.get("name", "xception_backbone"))

    def efficientnet_factory(*args, **kwargs):
        if fail_imagenet and kwargs.get("weights") == "imagenet":
            raise RuntimeError("download failed")
        return FakeBackbone("efficientnetb0")

    keras.Input = input_factory
    keras.Model = FakeCompiledModel
    keras.layers = FakeLayers()
    keras.optimizers = FakeOptimizers
    keras.regularizers = FakeRegularizers
    keras.metrics = FakeMetrics
    keras.models = models
    applications.Xception = xception_factory
    applications.EfficientNetB0 = efficientnet_factory
    applications.EfficientNetB4 = efficientnet_factory
    tensorflow.keras = keras

    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.applications", applications)
    return tensorflow


def test_xception_build_load_unfreeze_and_availability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_fake_tensorflow(monkeypatch)
    weights = tmp_path / "xception.h5"
    weights.write_bytes(b"placeholder")
    model = XceptionForensicModel(weights)

    built = model.build_model(imagenet_weights=False)
    assert built.name == "xception_forensic"
    assert built.compiled is True
    assert model.is_available() is True
    assert model.is_available(tmp_path / "missing.h5") is False

    model.load_weights()
    assert model.model.name == "loaded_model"
    model.unfreeze_for_finetuning()
    assert model.model.compiled is True
    assert model.model.get_layer("xception_backbone").trainable is True


def test_xception_load_weights_falls_back_to_existing_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tf = install_fake_tensorflow(monkeypatch)
    tf.keras.models.load_model = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad model"))
    weights = tmp_path / "weights.h5"
    weights.write_bytes(b"placeholder")
    model = XceptionForensicModel(weights)
    model.model = FakeCompiledModel(name="existing")

    model.load_weights()

    assert model.model.loaded_weights_path == str(weights)


def test_xception_missing_weights_and_unfreeze_guard(tmp_path: Path) -> None:
    model = XceptionForensicModel(tmp_path / "missing.h5")

    with pytest.raises(FileNotFoundError):
        model.load_weights()
    with pytest.raises(RuntimeError, match="build_model"):
        model.unfreeze_for_finetuning()


def test_efficientnet_build_load_and_keras_v3_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_fake_tensorflow(monkeypatch, fail_imagenet=True)
    weights = tmp_path / "efficientnet.h5"
    with h5py.File(weights, "w") as handle:
        handle.create_group("layers")
        handle.create_group("vars")

    model = EfficientNetForensicModel(weights)
    built = model.build_model(imagenet_weights=True)

    assert built.name == "efficientnet_forensic"
    assert model.pretrained_weights_loaded is False
    assert model.backbone_name == "efficientnetb0"
    assert model.is_available() is True
    assert EfficientNetForensicModel._is_keras_v3_weights_file(weights) is True
    assert EfficientNetForensicModel._is_keras_v3_weights_file(tmp_path / "no-file.h5") is False

    model._load_keras_v3_weights_with_compatible_suffix(weights)
    assert str(model.model.loaded_weights_path).endswith(".weights.h5")


def test_efficientnet_load_weights_fallbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tf = install_fake_tensorflow(monkeypatch)
    weights = tmp_path / "efficientnet.h5"
    weights.write_bytes(b"not a keras v3 hdf5")
    model = EfficientNetForensicModel(weights)

    model.load_weights()
    assert model.model.name == "loaded_model"

    tf.keras.models.load_model = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad model"))
    model.model = FakeCompiledModel(name="existing")
    model.load_weights()
    assert model.model.loaded_weights_path == str(weights)

    with pytest.raises(FileNotFoundError):
        EfficientNetForensicModel(tmp_path / "missing.h5").load_weights()


def test_efficientnet_compatible_suffix_requires_built_model(tmp_path: Path) -> None:
    weights = tmp_path / "efficientnet.h5"
    weights.write_bytes(b"placeholder")

    with pytest.raises(RuntimeError, match="build edilmeden"):
        EfficientNetForensicModel(weights)._load_keras_v3_weights_with_compatible_suffix(weights)


def test_unavailable_result_factories_and_legacy_alias() -> None:
    xception = XceptionForensicModel.unavailable_result("missing")
    efficientnet = EfficientNetForensicModel.unavailable_result("missing")

    assert xception.class_label == "Unavailable"
    assert efficientnet.class_label == "Unavailable"
    assert EfficientNetLSTMModel is EfficientNetForensicModel
