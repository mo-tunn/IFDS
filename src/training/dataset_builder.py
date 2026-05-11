"""
@file dataset_builder.py
@brief Optional dataset discovery and TensorFlow dataset construction helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from sklearn.model_selection import train_test_split

from config.settings import (
    BATCH_SIZE,
    CLASS_AUTHENTIC,
    CLASS_TAMPERED,
    RANDOM_STATE,
    RAW_DIR,
    SUPPORTED_FORMATS,
    TRAIN_RATIO,
    VAL_RATIO,
    XCEPTION_INPUT_SIZE,
)


@dataclass(frozen=True)
class DatasetSplit:
    """
    @class DatasetSplit
    @brief File paths and labels for one dataset split.
    """

    paths: list[str]
    labels: list[int]


class ForensicDatasetBuilder:
    """
    @class ForensicDatasetBuilder
    @brief Discovers CASIA v2, Coverage, and Columbia datasets for optional training.
    """

    def __init__(self, raw_dir: str | Path = RAW_DIR) -> None:
        """
        @brief Initialize dataset builder.
        @param raw_dir Root raw dataset directory.
        """
        self.raw_dir = Path(raw_dir)

    def load_image_paths(self) -> tuple[list[str], list[int]]:
        """
        @brief Load all supported dataset image paths and binary labels.
        @return Tuple of paths and labels.
        """
        paths: list[str] = []
        labels: list[int] = []

        self._append_casia(paths, labels)
        self._append_coverage(paths, labels)
        self._append_columbia(paths, labels)
        return paths, labels

    def split_paths(self) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """
        @brief Create stratified train, validation, and test splits.
        @return Train, validation, and test DatasetSplit objects.
        @raises ValueError When no supported dataset images are found.
        """
        paths, labels = self.load_image_paths()
        if not paths:
            raise ValueError(f"Veri seti görüntüsü bulunamadı: {self.raw_dir}")

        x_temp, x_test, y_temp, y_test = train_test_split(
            paths,
            labels,
            test_size=1.0 - TRAIN_RATIO - VAL_RATIO,
            stratify=labels,
            random_state=RANDOM_STATE,
        )
        val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp,
            y_temp,
            test_size=val_ratio_adjusted,
            stratify=y_temp,
            random_state=RANDOM_STATE,
        )
        return (
            DatasetSplit(list(x_train), list(y_train)),
            DatasetSplit(list(x_val), list(y_val)),
            DatasetSplit(list(x_test), list(y_test)),
        )

    def build_tf_datasets(self) -> tuple[Any, Any, Any]:
        """
        @brief Build TensorFlow datasets for optional model training.
        @return Train, validation, and test tf.data.Dataset instances.
        @raises ImportError When TensorFlow is not installed.
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow yüklü değil; eğitim araçları kullanılamaz.") from exc

        train, val, test = self.split_paths()
        return (
            self._create_tf_dataset(tf, train, augment=True),
            self._create_tf_dataset(tf, val, augment=False),
            self._create_tf_dataset(tf, test, augment=False),
        )

    def _create_tf_dataset(self, tf: Any, split: DatasetSplit, augment: bool) -> Any:
        ds = tf.data.Dataset.from_tensor_slices((split.paths, split.labels))

        def load_and_preprocess(path: Any, label: Any) -> tuple[Any, Any]:
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, XCEPTION_INPUT_SIZE)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        def augment_image(image: Any, label: Any) -> tuple[Any, Any]:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            return image, label

        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.shuffle(1000, seed=RANDOM_STATE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def _append_casia(self, paths: list[str], labels: list[int]) -> None:
        au_dirs = self._find_label_dirs("Au")
        tp_dirs = self._find_label_dirs("Tp")
        for directory in au_dirs:
            self._append_directory(paths, labels, directory, CLASS_AUTHENTIC)
        for directory in tp_dirs:
            self._append_directory(paths, labels, directory, CLASS_TAMPERED)

    def _append_coverage(self, paths: list[str], labels: list[int]) -> None:
        coverage = self.raw_dir / "Coverage"
        self._append_directory(paths, labels, coverage / "original", CLASS_AUTHENTIC)
        self._append_directory(paths, labels, coverage / "tampered", CLASS_TAMPERED)

    def _append_columbia(self, paths: list[str], labels: list[int]) -> None:
        columbia = self.raw_dir / "Columbia"
        self._append_directory(paths, labels, columbia / "authentic", CLASS_AUTHENTIC)
        self._append_directory(paths, labels, columbia / "spliced", CLASS_TAMPERED)

    def _append_directory(self, paths: list[str], labels: list[int], directory: Path, label: int) -> None:
        if not directory.exists():
            return
        existing = set(paths)
        for image_path in sorted(directory.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_FORMATS:
                path_str = str(image_path)
                if path_str not in existing and self._is_valid_image_file(image_path):
                    paths.append(path_str)
                    labels.append(label)
                    existing.add(path_str)

    def _find_label_dirs(self, dirname: str) -> list[Path]:
        direct_candidates = [
            self.raw_dir / dirname,
            self.raw_dir / "CASIA2" / dirname,
        ]
        discovered = [path for path in direct_candidates if path.exists() and path.is_dir()]
        discovered.extend(
            path
            for path in self.raw_dir.rglob("*")
            if path.is_dir() and path.name.lower() == dirname.lower()
        )

        unique: list[Path] = []
        seen: set[Path] = set()
        for path in discovered:
            resolved = path.resolve()
            if resolved not in seen:
                unique.append(path)
                seen.add(resolved)
        return unique

    @staticmethod
    def _is_valid_image_file(path: Path) -> bool:
        try:
            with Image.open(path) as image:
                image.verify()
            return True
        except Exception:
            return False
