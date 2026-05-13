"""
@file train_lstm.py
@brief Optional EfficientNet CNN training placeholder kept for legacy command compatibility.
@usage python -m src.training.train_lstm
"""

from __future__ import annotations

from config.settings import EFFICIENTNET_MODEL_PATH, MODELS_DIR
from src.ai_models.efficientnet_model import EfficientNetForensicModel


def train() -> None:
    """
    @brief Build the EfficientNet CNN architecture for optional training workflows.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = EfficientNetForensicModel().build_model(imagenet_weights=True)
    raise NotImplementedError(
        "Use notebooks/IFDS_Kaggle_EfficientNet_CNN_Training.ipynb for EfficientNet training. "
        f"Model skeleton ready: {model.name}; target path: {EFFICIENTNET_MODEL_PATH}"
    )


if __name__ == "__main__":
    train()
