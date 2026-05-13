"""
@file test_dataset_builder.py
@brief Tests for optional dataset discovery and TensorFlow dataset assembly helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from config.settings import CLASS_AUTHENTIC, CLASS_TAMPERED
from src.training.dataset_builder import DatasetSplit, ForensicDatasetBuilder


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 120, dtype=np.uint8)).save(path)


def test_load_image_paths_discovers_known_dataset_layouts(tmp_path: Path) -> None:
    _write_image(tmp_path / "CASIA2" / "Au" / "authentic.png")
    _write_image(tmp_path / "CASIA2" / "Tp" / "tampered.jpg")
    _write_image(tmp_path / "Coverage" / "original" / "cov_authentic.png")
    _write_image(tmp_path / "Coverage" / "tampered" / "cov_tampered.png")
    _write_image(tmp_path / "Columbia" / "authentic" / "col_authentic.png")
    _write_image(tmp_path / "Columbia" / "spliced" / "col_spliced.png")
    (tmp_path / "CASIA2" / "Au" / "broken.png").write_text("not an image", encoding="utf-8")
    (tmp_path / "CASIA2" / "Au" / "ignored.txt").write_text("text", encoding="utf-8")

    paths, labels = ForensicDatasetBuilder(tmp_path).load_image_paths()

    assert len(paths) == 6
    assert labels.count(CLASS_AUTHENTIC) == 3
    assert labels.count(CLASS_TAMPERED) == 3
    assert all(Path(path).suffix.lower() in {".png", ".jpg"} for path in paths)


def test_split_paths_returns_three_stratified_splits(tmp_path: Path) -> None:
    for index in range(12):
        _write_image(tmp_path / "Au" / f"auth_{index}.png")
        _write_image(tmp_path / "Tp" / f"tamp_{index}.png")

    train, val, test = ForensicDatasetBuilder(tmp_path).split_paths()

    assert isinstance(train, DatasetSplit)
    assert len(train.paths) + len(val.paths) + len(test.paths) == 24
    assert set(train.labels + val.labels + test.labels) == {CLASS_AUTHENTIC, CLASS_TAMPERED}


def test_split_paths_raises_when_no_images(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Veri seti görüntüsü bulunamadı"):
        ForensicDatasetBuilder(tmp_path).split_paths()


def test_build_tf_datasets_reports_missing_tensorflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "tensorflow", None)

    with pytest.raises(ImportError, match="TensorFlow yüklü değil"):
        ForensicDatasetBuilder(tmp_path).build_tf_datasets()


class FakeDataset:
    def __init__(self, items):
        self.items = items
        self.maps = 0
        self.shuffled = False
        self.batched = None
        self.prefetched = None

    def map(self, callback, num_parallel_calls=None):
        self.maps += 1
        path, label = self.items[0]
        callback(path, label)
        return self

    def shuffle(self, size: int, seed: int):
        self.shuffled = (size, seed)
        return self

    def batch(self, batch_size: int):
        self.batched = batch_size
        return self

    def prefetch(self, autotune):
        self.prefetched = autotune
        return self


class FakeTensorFlow:
    float32 = np.float32

    class data:
        AUTOTUNE = "AUTO"
        last_dataset: FakeDataset | None = None

        class Dataset:
            @staticmethod
            def from_tensor_slices(items):
                dataset = FakeDataset(list(zip(*items)))
                FakeTensorFlow.data.last_dataset = dataset
                return dataset

    class io:
        @staticmethod
        def read_file(path):
            return path

    class image:
        @staticmethod
        def decode_image(_image, channels: int, expand_animations: bool):
            return np.ones((4, 4, channels), dtype=np.uint8)

        @staticmethod
        def resize(image, size):
            return np.ones((*size, image.shape[2]), dtype=np.float32)

        @staticmethod
        def random_flip_left_right(image):
            return image

        @staticmethod
        def random_brightness(image, max_delta: float):
            return image

        @staticmethod
        def random_contrast(image, lower: float, upper: float):
            return image

    @staticmethod
    def cast(image, dtype):
        return image.astype(dtype)


def test_create_tf_dataset_applies_augmentation_pipeline() -> None:
    split = DatasetSplit(paths=["a.png"], labels=[CLASS_TAMPERED])
    dataset = ForensicDatasetBuilder()._create_tf_dataset(FakeTensorFlow, split, augment=True)

    assert dataset.maps == 2
    assert dataset.shuffled
    assert dataset.batched == 32
    assert dataset.prefetched == "AUTO"


def test_create_tf_dataset_without_augmentation_skips_shuffle() -> None:
    split = DatasetSplit(paths=["a.png"], labels=[CLASS_AUTHENTIC])
    dataset = ForensicDatasetBuilder()._create_tf_dataset(FakeTensorFlow, split, augment=False)

    assert dataset.maps == 1
    assert dataset.shuffled is False
