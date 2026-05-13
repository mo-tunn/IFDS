"""
@file xception_model.py
@brief Inference-first Xception wrapper for binary image forgery classification.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import (
    CLASS_NAMES,
    FINE_TUNE_LAYERS,
    LEARNING_RATE,
    XCEPTION_DECISION_THRESHOLD,
    XCEPTION_INPUT_SIZE,
    XCEPTION_MODEL_PATH,
)
from src.ai_models.base_model import AIDetectionResult


class XceptionForensicModel:
    """
    @class XceptionForensicModel
    @brief Loads a Keras .h5 Xception model or weights for image forgery inference.
    """

    def __init__(self, model_path: str | Path = XCEPTION_MODEL_PATH) -> None:
        """
        @brief Initialize model wrapper.
        @param model_path Default .h5 path.
        """
        self.model_path = Path(model_path)
        self.model: Any | None = None
        self.input_size = XCEPTION_INPUT_SIZE

    def build_model(self, imagenet_weights: bool = False) -> Any:
        """
        @brief Build the Xception binary classifier architecture.
        @param imagenet_weights Use ImageNet weights for optional training workflows.
        @return Compiled Keras model.
        @raises ImportError When TensorFlow is not installed.
        """
        tf = self._tensorflow()
        from tensorflow.keras import Model, layers, optimizers, regularizers
        from tensorflow.keras.applications import Xception

        base_model = Xception(
            weights="imagenet" if imagenet_weights else None,
            include_top=False,
            input_shape=(*self.input_size, 3),
            name="xception_backbone",
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(*self.input_size, 3), name="image")
        x = layers.Rescaling(2.0, offset=-1.0, name="xception_preprocess")(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
        x = layers.BatchNormalization(name="bn_head")(x)
        x = layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name="dense_128",
        )(x)
        x = layers.Dropout(0.5, name="dropout_128")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="forgery_output")(x)

        model = Model(inputs=inputs, outputs=outputs, name="xception_forensic")
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
            raise FileNotFoundError(f"Xception model weights were not found: {model_path}")

        tf = self._tensorflow()
        try:
            self.model = tf.keras.models.load_model(str(model_path), compile=False)
        except Exception:
            if self.model is None:
                self.build_model(imagenet_weights=False)
            self.model.load_weights(str(model_path))

    def unfreeze_for_finetuning(self) -> None:
        """
        @brief Unfreeze the last configured Xception backbone layers for fine-tuning.
        @raises RuntimeError When the model has not been built.
        """
        if self.model is None:
            raise RuntimeError("build_model() must be called first.")
        tf = self._tensorflow()
        from tensorflow.keras import optimizers

        base_model = self.model.get_layer("xception_backbone")
        base_model.trainable = True
        for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
            layer.trainable = False
        for layer in base_model.layers[-FINE_TUNE_LAYERS:]:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE / 10),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

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
            raise RuntimeError("Xception model is not loaded. Call load_weights() first.")
        start = time.perf_counter()
        batch = np.expand_dims(image.astype(np.float32), axis=0)
        prediction = self.model.predict(batch, verbose=0)
        tampered_probability = float(np.asarray(prediction).reshape(-1)[0])
        is_forged = tampered_probability >= XCEPTION_DECISION_THRESHOLD
        confidence = tampered_probability if is_forged else 1.0 - tampered_probability
        return AIDetectionResult(
            model_name="Xception",
            is_forged=is_forged,
            confidence=float(confidence),
            class_label=CLASS_NAMES[int(is_forged)],
            processing_time=time.perf_counter() - start,
        )

    @staticmethod
    def unavailable_result(message: str) -> AIDetectionResult:
        """
        @brief Build a non-fatal result for unavailable Xception inference.
        @param message Explanation to show in the UI and report.
        @return AIDetectionResult with error_message populated.
        """
        return AIDetectionResult(
            model_name="Xception",
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
