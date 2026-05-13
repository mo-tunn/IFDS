"""
@file gradcam.py
@brief Grad-CAM visualization for Keras classification models.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class GradCAM:
    """
    @class GradCAM
    @brief Generates a class activation heatmap for an Xception-like Keras model.
    """

    def __init__(self, model: Any, last_conv_layer_name: str = "block14_sepconv2_act") -> None:
        """
        @brief Initialize Grad-CAM generator.
        @param model Loaded Keras model.
        @param last_conv_layer_name Preferred convolution layer name.
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name

    def generate(self, image_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        @brief Generate heatmap and overlay for a single normalized image batch.
        @param image_batch Float32 batch with shape (1, H, W, 3).
        @return Tuple of heatmap and RGB overlay.
        @raises RuntimeError When the required layer cannot be resolved.
        """
        tf = self._tensorflow()
        grad_model = self._build_grad_model(tf)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_batch)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError("Grad-CAM gradient could not be computed.")
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)

        base = np.clip(image_batch[0] * 255, 0, 255).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
        colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(base, 0.62, colored_rgb, 0.38, 0)
        return heatmap, overlay

    def _build_grad_model(self, tf: Any) -> Any:
        """
        @brief Build a Grad-CAM model connected to the outer classifier graph.
        @param tf TensorFlow module.
        @return Keras model returning activation maps and predictions.
        """
        try:
            return self._build_xception_backbone_grad_model(tf)
        except Exception:
            conv_layer = self._find_grad_layer()
            if conv_layer is None:
                raise RuntimeError(f"Grad-CAM layer was not found: {self.last_conv_layer_name}")
            return tf.keras.Model(self.model.inputs, [conv_layer.output, self.model.output])

    def _build_xception_backbone_grad_model(self, tf: Any) -> Any:
        """
        @brief Reconnect the nested Xception backbone for Keras 3 compatibility.
        @param tf TensorFlow module.
        @return Keras model returning backbone feature maps and predictions.
        """
        preprocess = self.model.get_layer("xception_preprocess")
        backbone = self.model.get_layer("xception_backbone")

        inputs = self.model.inputs[0]
        x = preprocess(inputs)
        conv_outputs = backbone(x, training=False)
        x = conv_outputs

        use_head = False
        for layer in self.model.layers:
            if layer is backbone:
                use_head = True
                continue
            if not use_head:
                continue
            if layer is backbone:
                continue
            if layer.__class__.__name__ == "Dropout":
                x = layer(x, training=False)
            else:
                x = layer(x)

        return tf.keras.Model(inputs, [conv_outputs, x])

    def _find_grad_layer(self) -> Any | None:
        """
        @brief Resolve a 4D activation layer connected to the outer model graph.
        @return Keras layer suitable for Grad-CAM.
        """
        for name in ("xception_backbone", self.last_conv_layer_name):
            try:
                layer = self.model.get_layer(name)
            except Exception:
                layer = None
            if layer is not None and self._has_rank4_output(layer):
                return layer

        for layer in reversed(getattr(self.model, "layers", [])):
            if self._has_rank4_output(layer):
                return layer

        found = self._find_layer(self.model, self.last_conv_layer_name)
        if found is not None and self._has_rank4_output(found):
            return found
        return None

    def _find_layer(self, model: Any, name: str) -> Any | None:
        try:
            return model.get_layer(name)
        except Exception:
            pass
        for layer in getattr(model, "layers", []):
            nested = getattr(layer, "layers", None)
            if nested is None:
                continue
            found = self._find_layer(layer, name)
            if found is not None:
                return found
        return None

    @staticmethod
    def _has_rank4_output(layer: Any) -> bool:
        """
        @brief Check whether a layer output looks like an image feature map.
        @param layer Keras layer.
        @return True for rank-4 outputs.
        """
        output = getattr(layer, "output", None)
        shape = getattr(output, "shape", None)
        return shape is not None and len(shape) == 4

    @staticmethod
    def _tensorflow() -> Any:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow is not installed. Install requirements.txt for Grad-CAM.") from exc
        return tf
