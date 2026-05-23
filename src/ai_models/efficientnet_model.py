"""
@file efficientnet_model.py
@brief EfficientNet CNN wrapper used as the second AI forgery detector.
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path
from shutil import copy2
from typing import Any

import h5py
import numpy as np

from config.settings import (
    CLASS_NAMES,
    EFFICIENTNET_DECISION_THRESHOLD,
    EFFICIENTNET_INPUT_SIZE,
    EFFICIENTNET_MODEL_PATH,
    EFFICIENTNET_VARIANT,
    LEARNING_RATE,
)
from src.ai_models.base_model import AIDetectionResult


class EfficientNetForensicModel:
    """
    @class EfficientNetForensicModel
    @brief Loads a Keras .h5 EfficientNet CNN for image forgery classification.
    """

    def __init__(self, model_path: str | Path = EFFICIENTNET_MODEL_PATH) -> None:
        """
        @brief Initialize model wrapper.
        @param model_path Default .h5 model path.
        """
        self.model_path = Path(model_path)
        self.model: Any | None = None
        self.input_size = EFFICIENTNET_INPUT_SIZE
        self.pretrained_weights_loaded = False
        self.backbone_name = ""

    def build_model(self, imagenet_weights: bool = False) -> Any:
        """
        @brief Build the EfficientNet binary classifier architecture.
        @param imagenet_weights Use ImageNet weights for optional training workflows.
        @return Compiled Keras model.
        @raises ImportError When TensorFlow is not installed.
        """
        tf = self._tensorflow()
        from tensorflow.keras import Model, layers, optimizers, regularizers
        from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4

        backbone_cls = EfficientNetB0 if EFFICIENTNET_VARIANT == "B0" else EfficientNetB4
        weights = "imagenet" if imagenet_weights else None
        try:
            base_model = backbone_cls(
                weights=weights,
                include_top=False,
                input_shape=(*self.input_size, 3),
            )
            self.pretrained_weights_loaded = weights == "imagenet"
        except Exception as exc:
            if weights is None:
                raise
            print(f"{EFFICIENTNET_VARIANT} ImageNet weights could not be downloaded; continuing with weights=None: {exc}")
            base_model = backbone_cls(
                weights=None,
                include_top=False,
                input_shape=(*self.input_size, 3),
            )
            self.pretrained_weights_loaded = False
        self.backbone_name = base_model.name
        base_model.trainable = not self.pretrained_weights_loaded

        inputs = tf.keras.Input(shape=(*self.input_size, 3), name="image")
        x = layers.Rescaling(255.0, name="efficientnet_input_rescale")(inputs)
        x = base_model(x, training=not self.pretrained_weights_loaded)
        x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
        x = layers.BatchNormalization(name="bn_head")(x)
        x = layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name="dense_128",
        )(x)
        x = layers.Dropout(0.45, name="dropout_128")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="forgery_output")(x)

        model = Model(inputs=inputs, outputs=outputs, name="efficientnet_forensic")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        self.model = model
        return model

    def load_weights(self, path: str | Path | None = None) -> None:
        """
        @brief Load a full .h5 model or compatible weights from disk.
        @param path Optional model path override.
        @raises FileNotFoundError When the .h5 file is missing.
        @raises ImportError When TensorFlow is not installed.
        """
        model_path = Path(path) if path is not None else self.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"EfficientNet model weights were not found: {model_path}")

        tf = self._tensorflow()
        try:
            self.model = tf.keras.models.load_model(str(model_path), compile=False)
        except Exception as load_model_error:
            if self.model is None:
                self.build_model(imagenet_weights=False)
            try:
                self.model.load_weights(str(model_path))
            except Exception as load_weights_error:
                if not self._is_keras_v3_weights_file(model_path):
                    raise load_weights_error from load_model_error
                self._load_keras_v3_weights_with_compatible_suffix(model_path)

    def is_available(self, path: str | Path | None = None) -> bool:
        """
        @brief Check whether the configured model file exists.
        @param path Optional model path override.
        @return True when a model file is present.
        """
        model_path = Path(path) if path is not None else self.model_path
        return model_path.exists()

    def predict(self, image: np.ndarray) -> AIDetectionResult:
        """
        @brief Predict whether a normalized RGB image is tampered.
        @param image Float32 RGB image with shape (224, 224, 3).
        @return AIDetectionResult with label and confidence.
        @raises RuntimeError When the model has not been loaded.
        """
        if self.model is None:
            raise RuntimeError("EfficientNet model is not loaded. Call load_weights() first.")
        start = time.perf_counter()
        batch = np.expand_dims(image.astype(np.float32), axis=0)
        prediction = self.model.predict(batch, verbose=0)
        tampered_probability = float(np.asarray(prediction).reshape(-1)[0])
        is_forged = tampered_probability >= EFFICIENTNET_DECISION_THRESHOLD
        confidence = tampered_probability if is_forged else 1.0 - tampered_probability
        return AIDetectionResult(
            model_name="EfficientNet CNN + LSTM",
            is_forged=is_forged,
            confidence=float(confidence),
            class_label=CLASS_NAMES[int(is_forged)],
            processing_time=time.perf_counter() - start,
        )

    @staticmethod
    def unavailable_result(message: str) -> AIDetectionResult:
        """
        @brief Build a non-fatal result for unavailable EfficientNet inference.
        @param message Explanation to show in the UI and report.
        @return AIDetectionResult with error_message populated.
        """
        return AIDetectionResult(
            model_name="EfficientNet CNN",
            is_forged=False,
            confidence=0.0,
            class_label="Unavailable",
            processing_time=0.0,
            error_message=message,
        )

    @staticmethod
    def _tensorflow() -> Any:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow is not installed. Install requirements.txt for AI inference.") from exc
        return tf

    @staticmethod
    def _is_keras_v3_weights_file(model_path: Path) -> bool:
        """
        @brief Detect Keras 3 save_weights HDF5 files copied to a .h5 filename.
        @param model_path HDF5 path.
        @return True when the file has Keras 3 weight-only groups.
        """
        try:
            with h5py.File(model_path, "r") as handle:
                keys = set(handle.keys())
                return {"layers", "vars"}.issubset(keys)
        except OSError:
            return False

    def _load_keras_v3_weights_with_compatible_suffix(self, model_path: Path) -> None:
        """
        @brief Load Keras 3 weight-only HDF5 after copying it to .weights.h5.
        @param model_path Source weights path.
        """
        if self.model is None:
            raise RuntimeError("EfficientNet weights cannot be loaded before the model is built.")
        with tempfile.TemporaryDirectory(prefix="ifds_efficientnet_") as tmp_dir:
            compatible_path = Path(tmp_dir) / "efficientnet_finetuned.weights.h5"
            copy2(model_path, compatible_path)
            self.model.load_weights(str(compatible_path))
