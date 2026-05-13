"""
@file test_gradcam.py
@brief Tests for Grad-CAM helper branches without importing real TensorFlow.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from src.ai_models.gradcam import GradCAM


class TensorBox:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class FakeGradientTape:
    gradient_value = np.ones((1, 2, 2, 3), dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def gradient(self, _loss, _conv_outputs):
        return self.gradient_value


class FakeGradModel:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __call__(self, _image_batch):
        conv = np.ones((1, 2, 2, 3), dtype=np.float32)
        predictions = np.array([[0.9]], dtype=np.float32)
        return conv, predictions


class FakeTensorFlow:
    newaxis = np.newaxis
    GradientTape = FakeGradientTape

    class keras:
        Model = FakeGradModel

    @staticmethod
    def reduce_mean(value, axis):
        return np.mean(value, axis=axis)

    @staticmethod
    def squeeze(value):
        return TensorBox(np.squeeze(value))


class FakeOutput:
    shape = (None, 4, 4, 8)


class FakeLayer:
    output = FakeOutput()

    def __init__(self, name: str = "layer") -> None:
        self.name = name

    def __call__(self, value, *args, **kwargs):
        return value


class DropoutLayer(FakeLayer):
    pass


DropoutLayer.__name__ = "Dropout"


class NestedModel:
    def __init__(self, layer: FakeLayer) -> None:
        self.layers = [layer]

    def get_layer(self, name: str):
        if self.layers[0].name == name:
            return self.layers[0]
        raise ValueError(name)


class FallbackModel:
    inputs = ["input"]
    output = "prediction"

    def __init__(self, layer: FakeLayer | None = None) -> None:
        self.layer = layer or FakeLayer("target")
        self.layers = [NestedModel(self.layer)]

    def get_layer(self, name: str):
        if name == self.layer.name:
            return self.layer
        raise ValueError(name)


class BackboneModel:
    inputs = ["input"]

    def __init__(self) -> None:
        self.preprocess = FakeLayer("xception_preprocess")
        self.backbone = FakeLayer("xception_backbone")
        self.head = FakeLayer("head")
        self.dropout = DropoutLayer("dropout")
        self.layers = [self.preprocess, self.backbone, self.dropout, self.head]

    def get_layer(self, name: str):
        if name == "xception_preprocess":
            return self.preprocess
        if name == "xception_backbone":
            return self.backbone
        raise ValueError(name)


def test_generate_builds_heatmap_and_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GradCAM, "_tensorflow", staticmethod(lambda: FakeTensorFlow))
    image_batch = np.ones((1, 8, 8, 3), dtype=np.float32) * 0.5

    heatmap, overlay = GradCAM(FallbackModel(FakeLayer("block14_sepconv2_act"))).generate(image_batch)

    assert heatmap.shape == (2, 2)
    assert overlay.shape == (8, 8, 3)
    assert overlay.dtype == np.uint8


def test_generate_raises_when_gradient_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GradCAM, "_tensorflow", staticmethod(lambda: FakeTensorFlow))
    FakeGradientTape.gradient_value = None
    try:
        with pytest.raises(RuntimeError, match="gradient"):
            GradCAM(FallbackModel(FakeLayer("block14_sepconv2_act"))).generate(
                np.ones((1, 8, 8, 3), dtype=np.float32)
            )
    finally:
        FakeGradientTape.gradient_value = np.ones((1, 2, 2, 3), dtype=np.float32)


def test_build_grad_model_uses_backbone_path() -> None:
    result = GradCAM(BackboneModel())._build_grad_model(FakeTensorFlow)

    assert isinstance(result, FakeGradModel)


def test_find_grad_layer_searches_nested_layers_and_handles_missing() -> None:
    layer = FakeLayer("target")
    gradcam = GradCAM(FallbackModel(layer), last_conv_layer_name="target")

    assert gradcam._find_grad_layer() is layer
    assert gradcam._find_layer(FallbackModel(layer), "target") is layer
    assert GradCAM(FallbackModel(layer), last_conv_layer_name="missing")._find_layer(types.SimpleNamespace(), "x") is None


def test_build_grad_model_reports_missing_layer() -> None:
    model = types.SimpleNamespace(inputs=["input"], output="out", layers=[], get_layer=lambda name: (_ for _ in ()).throw(ValueError(name)))

    with pytest.raises(RuntimeError, match="Grad-CAM layer was not found"):
        GradCAM(model)._build_grad_model(FakeTensorFlow)
