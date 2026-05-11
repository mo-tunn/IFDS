"""
@file train_xception.py
@brief Optional Xception training entrypoint.
@usage python -m src.training.train_xception
"""

from __future__ import annotations

from config.settings import EPOCHS_XCEPTION, MODELS_DIR, XCEPTION_MODEL_PATH
from src.ai_models.xception_model import XceptionForensicModel
from src.training.dataset_builder import ForensicDatasetBuilder


def train() -> None:
    """
    @brief Train and save the Xception classifier.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds = ForensicDatasetBuilder().build_tf_datasets()
    model_wrapper = XceptionForensicModel()
    model = model_wrapper.build_model(imagenet_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_XCEPTION)
    model.save(str(XCEPTION_MODEL_PATH))
    model.evaluate(test_ds)


if __name__ == "__main__":
    train()
