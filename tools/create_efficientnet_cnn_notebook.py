from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
        "source": dedent(source).strip().splitlines(True),
    }


cells = [
    cell(
        "markdown",
        """
        # IFDS EfficientNet CNN Eğitimi

        Bu notebook IFDS projesi için ikinci AI algoritması olan **EfficientNet CNN** modelini eğitir.

        - Dataset: `divg07/casia-20-image-tampering-detection-dataset`
        - Görüntüler: `CASIA2/Au`, `CASIA2/Tp`
        - Maskeler: `CASIA 2 Groundtruth(5123 files)`
        - Model: EfficientNetB0/B4 tabanlı binary classifier
        - Groundtruth varsa: tampered eğitim örneklerinde mask-guided crop augmentation
        - ImageNet ağırlığı indirilemezse: backbone otomatik trainable kalır; donuk rastgele feature extractor kullanılmaz
        - Output: `/kaggle/working/efficientnet_finetuned.h5`
        - Metrics: `/kaggle/working/efficientnet_training_results.json`
        """,
    ),
    cell(
        "code",
        """
        # Hücre 2 — Kurulum
        import importlib.util, os, subprocess, sys
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        required = {"tensorflow": "tensorflow", "cv2": "opencv-contrib-python", "sklearn": "scikit-learn", "tqdm": "tqdm"}
        missing = [pkg for module, pkg in required.items() if importlib.util.find_spec(module) is None]
        print("Eksik paketler:", missing if missing else "Yok")
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        """,
    ),
    cell(
        "code",
        """
        # Hücre 3 — Import ve sabitler
        from __future__ import annotations
        import gc, json, os, random, time
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        from dataclasses import dataclass
        from pathlib import Path
        from typing import Any

        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from PIL import Image
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_class_weight
        from tqdm.auto import tqdm

        KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset"
        DEFAULT_CASIA_DIR = Path("/kaggle/input/casia-20-image-tampering-detection-dataset")
        ALT_CASIA_DIR = Path("/kaggle/input/datasets/divg07/casia-20-image-tampering-detection-dataset")
        CASIA_DIR = DEFAULT_CASIA_DIR if DEFAULT_CASIA_DIR.exists() else ALT_CASIA_DIR
        if not CASIA_DIR.exists():
            input_root = Path("/kaggle/input")
            candidates = []
            if input_root.exists():
                for candidate in input_root.rglob("*"):
                    if (
                        candidate.is_dir()
                        and candidate.name == "CASIA2"
                        and (candidate / "Au").exists()
                        and (candidate / "Tp").exists()
                    ):
                        candidates.append(candidate.parent)
            if candidates:
                CASIA_DIR = candidates[0]

        OUTPUT_DIR = Path("/kaggle/working")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        EFFICIENTNET_INPUT_SIZE = (224, 224)
        BATCH_SIZE = 32
        EPOCHS_EFFICIENTNET = 20
        LEARNING_RATE = 1e-4
        FINE_TUNE_LAYERS = 20
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
        RANDOM_STATE = 42
        SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"}
        CLASS_AUTHENTIC, CLASS_TAMPERED = 0, 1
        CLASS_NAMES = ["Authentic", "Tampered"]
        CONFIDENCE_THRESHOLD = 0.5
        EFFICIENTNET_VARIANT = "B0"
        USE_MASK_GUIDED_CROPS = True
        REQUIRE_GROUNDTRUTH = True
        REQUIRE_PRETRAINED_BACKBONE = True
        MASK_CROP_PROBABILITY = 0.65
        AUTHENTIC_RANDOM_CROP_PROBABILITY = 0.30
        CROP_MARGIN_RATIO = 0.18
        GROUNDTRUTH_DIR_NAMES = [
            "CASIA 2 Groundtruth(5123 files)",
            "CASIA 2 Groundtruth",
            "CASIA2 Groundtruth",
            "Groundtruth",
            "Gt",
            "GT",
        ]
        EFFICIENTNET_MODEL_PATH = OUTPUT_DIR / "efficientnet_finetuned.h5"
        EFFICIENTNET_WEIGHTS_PATH = OUTPUT_DIR / "efficientnet_finetuned.weights.h5"
        EFFICIENTNET_BEST_PATH = OUTPUT_DIR / "efficientnet_best.keras"
        RESULTS_PATH = OUTPUT_DIR / "efficientnet_training_results.json"
        training_results = {"efficientnet": {}, "files": {}}

        np.random.seed(RANDOM_STATE)
        random.seed(RANDOM_STATE)
        tf.random.set_seed(RANDOM_STATE)

        def file_size_mb(path: str | Path) -> float:
            return round(Path(path).stat().st_size / (1024 * 1024), 2)

        def load_model_compat(path: str | Path) -> tf.keras.Model:
            try:
                return tf.keras.models.load_model(str(path), compile=False, safe_mode=False)
            except TypeError:
                return tf.keras.models.load_model(str(path), compile=False)

        def save_results() -> None:
            RESULTS_PATH.write_text(json.dumps(training_results, indent=2, ensure_ascii=False), encoding="utf-8")
            print("Sonuç dosyası yazıldı:", RESULTS_PATH)

        def find_local_efficientnet_weights(variant: str) -> str | None:
            expected = "efficientnetb0_notop.h5" if variant == "B0" else "efficientnetb4_notop.h5"
            roots = [Path("/kaggle/input"), Path("/root/.keras/models"), Path.home() / ".keras" / "models"]
            for root in roots:
                if not root.exists():
                    continue
                direct = root / expected
                if direct.exists():
                    return str(direct)
                matches = list(root.rglob(expected))
                if matches:
                    return str(matches[0])
            return None

        print("CASIA_DIR:", CASIA_DIR)
        print("Dataset URL:", KAGGLE_DATASET_URL)
        print("OUTPUT_DIR:", OUTPUT_DIR)
        print("EFFICIENTNET_VARIANT:", EFFICIENTNET_VARIANT)
        print("EFFICIENTNET_INPUT_SIZE:", EFFICIENTNET_INPUT_SIZE)
        print("USE_MASK_GUIDED_CROPS:", USE_MASK_GUIDED_CROPS)
        print("REQUIRE_GROUNDTRUTH:", REQUIRE_GROUNDTRUTH)
        print("REQUIRE_PRETRAINED_BACKBONE:", REQUIRE_PRETRAINED_BACKBONE)
        print("Local EfficientNet weights:", find_local_efficientnet_weights(EFFICIENTNET_VARIANT))
        """,
    ),
    cell(
        "code",
        """
        # Hücre 4 — GPU kontrolü
        print("TensorFlow:", tf.__version__)
        gpus = tf.config.list_physical_devices("GPU")
        print("GPU listesi:", gpus)
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("Memory growth etkin:", gpu)
                except RuntimeError as exc:
                    print("Memory growth ayarlanamadı:", exc)
        else:
            print("GPU bulunamadı; BATCH_SIZE 16 yapılıyor.")
            BATCH_SIZE = 16
        print("Aktif BATCH_SIZE:", BATCH_SIZE)
        """,
    ),
    cell(
        "code",
        """
        # Hücre 5 — Dataset keşfi ve doğrulama
        def _as_path(path_value) -> Path:
            if isinstance(path_value, bytes):
                return Path(path_value.decode("utf-8"))
            if hasattr(path_value, "numpy"):
                return Path(path_value.numpy().decode("utf-8"))
            return Path(str(path_value))

        def is_valid_image_file(path: str | Path) -> bool:
            try:
                with Image.open(path) as image:
                    image.verify()
                return True
            except Exception:
                return False

        def read_rgb_uint8(path: str | Path) -> np.ndarray:
            path = Path(path)
            try:
                with Image.open(path) as image:
                    return np.asarray(image.convert("RGB"), dtype=np.uint8)
            except Exception:
                image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise ValueError(f"Görüntü okunamadı: {path}")
                return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        def read_mask_uint8(mask_path: str | Path, image_shape: tuple[int, int]) -> np.ndarray | None:
            if not mask_path:
                return None
            path = Path(mask_path)
            if not path.exists():
                return None
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                try:
                    with Image.open(path) as image:
                        mask = np.asarray(image.convert("L"), dtype=np.uint8)
                except Exception:
                    return None
            height, width = image_shape
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            return (mask > 127).astype(np.uint8)

        def crop_with_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                return image
            height, width = image.shape[:2]
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            box_w, box_h = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
            margin = int(max(box_w, box_h) * CROP_MARGIN_RATIO)
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(width - 1, x2 + margin), min(height - 1, y2 + margin)
            crop = image[y1 : y2 + 1, x1 : x2 + 1]
            return crop if crop.size else image

        def random_crop(image: np.ndarray, min_scale: float = 0.70) -> np.ndarray:
            height, width = image.shape[:2]
            if height < 8 or width < 8:
                return image
            scale = float(np.random.uniform(min_scale, 1.0))
            crop_h = max(8, int(height * scale))
            crop_w = max(8, int(width * scale))
            y1 = int(np.random.randint(0, max(1, height - crop_h + 1)))
            x1 = int(np.random.randint(0, max(1, width - crop_w + 1)))
            return image[y1 : y1 + crop_h, x1 : x1 + crop_w]

        def load_image_numpy(path_value, target_size) -> np.ndarray:
            path = _as_path(path_value)
            if hasattr(target_size, "numpy"):
                target_size = tuple(int(x) for x in target_size.numpy().tolist())
            else:
                target_size = tuple(int(x) for x in target_size)
            image_rgb = read_rgb_uint8(path)
            image_rgb = cv2.resize(image_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
            array = image_rgb.astype(np.float32) / 255.0
            return array.astype(np.float32)

        def load_training_image_numpy(path_value, label_value, mask_path_value, target_size, augment_value) -> np.ndarray:
            path = _as_path(path_value)
            label = int(label_value.numpy()) if hasattr(label_value, "numpy") else int(label_value)
            mask_path = _as_path(mask_path_value)
            mask_path_str = "" if str(mask_path) == "." else str(mask_path)
            augment = bool(augment_value.numpy()) if hasattr(augment_value, "numpy") else bool(augment_value)
            if hasattr(target_size, "numpy"):
                target_size = tuple(int(x) for x in target_size.numpy().tolist())
            else:
                target_size = tuple(int(x) for x in target_size)

            image_rgb = read_rgb_uint8(path)
            if augment and USE_MASK_GUIDED_CROPS and label == CLASS_TAMPERED and mask_path_str and np.random.random() < MASK_CROP_PROBABILITY:
                mask = read_mask_uint8(mask_path_str, image_rgb.shape[:2])
                if mask is not None:
                    image_rgb = crop_with_mask(image_rgb, mask)
            elif augment and label == CLASS_AUTHENTIC and np.random.random() < AUTHENTIC_RANDOM_CROP_PROBABILITY:
                image_rgb = random_crop(image_rgb)

            image_rgb = cv2.resize(image_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
            array = image_rgb.astype(np.float32) / 255.0
            return array.astype(np.float32)

        @dataclass(frozen=True)
        class DatasetSplit:
            paths: list[str]
            labels: list[int]
            mask_paths: list[str]

        class ForensicDatasetBuilder:
            def __init__(self, raw_dir: str | Path = CASIA_DIR) -> None:
                self.raw_dir = Path(raw_dir)
                self.mask_index = self._build_mask_index()

            def load_image_paths(self) -> tuple[list[str], list[int]]:
                paths, labels, _ = self.load_image_records()
                return paths, labels

            def load_image_records(self) -> tuple[list[str], list[int], list[str]]:
                paths, labels, mask_paths = [], [], []
                self._append_casia(paths, labels, mask_paths)
                return paths, labels, mask_paths

            def split_paths(self) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
                paths, labels, mask_paths = self.load_image_records()
                if not paths:
                    raise ValueError(f"Veri seti görüntüsü bulunamadı: {self.raw_dir}")
                x_temp, x_test, y_temp, y_test, m_temp, m_test = train_test_split(
                    paths, labels, mask_paths, test_size=TEST_RATIO, stratify=labels, random_state=RANDOM_STATE
                )
                val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
                x_train, x_val, y_train, y_val, m_train, m_val = train_test_split(
                    x_temp, y_temp, m_temp, test_size=val_ratio_adjusted, stratify=y_temp, random_state=RANDOM_STATE
                )
                return (
                    DatasetSplit(list(x_train), list(y_train), list(m_train)),
                    DatasetSplit(list(x_val), list(y_val), list(m_val)),
                    DatasetSplit(list(x_test), list(y_test), list(m_test)),
                )

            def _append_casia(self, paths: list[str], labels: list[int], mask_paths: list[str]) -> None:
                au_dirs = self._find_label_dirs("Au")
                tp_dirs = self._find_label_dirs("Tp")
                print("Bulunan Au klasörleri:", [str(path) for path in au_dirs])
                print("Bulunan Tp klasörleri:", [str(path) for path in tp_dirs])
                print("Groundtruth mask sayısı:", len(self.mask_index))
                for directory in au_dirs:
                    self._append_directory(paths, labels, mask_paths, directory, CLASS_AUTHENTIC)
                for directory in tp_dirs:
                    self._append_directory(paths, labels, mask_paths, directory, CLASS_TAMPERED)

            def _find_label_dirs(self, dirname: str) -> list[Path]:
                direct = [self.raw_dir / dirname, self.raw_dir / "CASIA2" / dirname]
                discovered = [path for path in direct if path.exists() and path.is_dir()]
                discovered.extend(path for path in self.raw_dir.rglob("*") if path.is_dir() and path.name.lower() == dirname.lower())
                unique, seen = [], set()
                for path in discovered:
                    resolved = path.resolve()
                    if resolved not in seen:
                        unique.append(path)
                        seen.add(resolved)
                return unique

            def _find_mask_dirs(self) -> list[Path]:
                names = {"gt", "groundtruth", "ground_truth", "mask", "masks"}
                direct = []
                for dirname in GROUNDTRUTH_DIR_NAMES:
                    direct.extend(
                        [
                            self.raw_dir / dirname,
                            self.raw_dir / "CASIA2" / dirname,
                            self.raw_dir.parent / dirname,
                        ]
                    )
                discovered = [path for path in direct if path.exists() and path.is_dir()]
                discovered.extend(
                    path
                    for path in self.raw_dir.rglob("*")
                    if path.is_dir() and (path.name.lower() in names or "groundtruth" in path.name.lower())
                )
                unique, seen = [], set()
                for path in discovered:
                    resolved = path.resolve()
                    if resolved not in seen:
                        unique.append(path)
                        seen.add(resolved)
                return unique

            @staticmethod
            def _mask_key(path: str | Path) -> str:
                stem = Path(path).stem.lower()
                for suffix in ("_gt", "_mask", "_edgemask", "_groundtruth", "_gt_mask"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                return stem

            def _build_mask_index(self) -> dict[str, str]:
                mask_dirs = self._find_mask_dirs()
                print("Bulunan Groundtruth klasörleri:", [str(path) for path in mask_dirs])
                mask_index: dict[str, str] = {}
                for directory in mask_dirs:
                    for mask_path in tqdm(sorted(directory.rglob("*")), desc=f"{directory} mask taranıyor"):
                        if mask_path.is_file() and mask_path.suffix.lower() in SUPPORTED_FORMATS:
                            mask_index.setdefault(self._mask_key(mask_path), str(mask_path))
                return mask_index

            def _match_mask(self, image_path: str | Path, label: int) -> str:
                if label != CLASS_TAMPERED or not self.mask_index:
                    return ""
                return self.mask_index.get(self._mask_key(image_path), "")

            def _append_directory(self, paths: list[str], labels: list[int], mask_paths: list[str], directory: Path, label: int) -> None:
                if not directory.exists():
                    return
                existing = set(paths)
                for image_path in tqdm(sorted(directory.rglob("*")), desc=f"{directory} taranıyor"):
                    if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_FORMATS:
                        path_str = str(image_path)
                        if path_str not in existing and is_valid_image_file(image_path):
                            paths.append(path_str)
                            labels.append(label)
                            mask_paths.append(self._match_mask(image_path, label))
                            existing.add(path_str)

        builder = ForensicDatasetBuilder(CASIA_DIR)
        all_paths, all_labels, all_masks = builder.load_image_records()
        counts = pd.Series(all_labels).map({0: "Authentic", 1: "Tampered"}).value_counts().reindex(["Authentic", "Tampered"]).fillna(0).astype(int)
        matched_masks = sum(1 for label, mask in zip(all_labels, all_masks) if label == CLASS_TAMPERED and mask)
        print("Geçerli görüntü:", len(all_paths))
        print("Tampered mask eşleşmesi:", matched_masks, "/", int(counts["Tampered"]))
        display(counts.rename("count").to_frame())
        assert len(all_paths) > 0 and set(all_labels) == {0, 1}
        if REQUIRE_GROUNDTRUTH:
            assert len(builder.mask_index) > 0, "CASIA 2 Groundtruth(5123 files) klasörü bulunamadı. Kaggle input olarak divg07 datasetini eklediğini kontrol et."
            assert matched_masks > 0, "Groundtruth dosyaları bulundu ama Tp görselleriyle eşleşmedi. Dosya isimlerini kontrol et."

        sample_indices = np.random.default_rng(RANDOM_STATE).choice(len(all_paths), size=min(5, len(all_paths)), replace=False)
        fig, axes = plt.subplots(1, len(sample_indices), figsize=(4 * len(sample_indices), 4))
        if len(sample_indices) == 1:
            axes = [axes]
        for ax, idx in zip(axes, sample_indices):
            image = read_rgb_uint8(all_paths[idx])
            ax.imshow(image)
            ax.set_title(CLASS_NAMES[int(all_labels[idx])] + (" + GT" if all_masks[idx] else ""))
            ax.axis("off")
        plt.tight_layout(); plt.show()
        """,
    ),
    cell(
        "code",
        """
        # Hücre 6 — Split ve tf.data pipeline
        train_split, val_split, test_split = builder.split_paths()
        print("Train:", len(train_split.paths), "Val:", len(val_split.paths), "Test:", len(test_split.paths))
        print("Train mask eşleşmesi:", sum(1 for label, mask in zip(train_split.labels, train_split.mask_paths) if label == CLASS_TAMPERED and mask))

        def make_dataset(split: DatasetSplit, augment: bool) -> tf.data.Dataset:
            ds = tf.data.Dataset.from_tensor_slices((split.paths, split.labels, split.mask_paths))
            def load_and_preprocess(path, label, mask_path):
                image = tf.numpy_function(
                    load_training_image_numpy,
                    [path, label, mask_path, EFFICIENTNET_INPUT_SIZE, augment],
                    tf.float32,
                )
                image.set_shape([EFFICIENTNET_INPUT_SIZE[0], EFFICIENTNET_INPUT_SIZE[1], 3])
                if augment:
                    image = tf.image.random_flip_left_right(image)
                    image = tf.image.random_brightness(image, 0.12)
                    image = tf.image.random_contrast(image, 0.9, 1.1)
                    image = tf.image.random_saturation(image, 0.9, 1.1)
                    image = tf.clip_by_value(image, 0.0, 1.0)
                return image, tf.cast(label, tf.float32)
            ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            if augment:
                ds = ds.shuffle(1000, seed=RANDOM_STATE)
            return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        train_ds = make_dataset(train_split, True)
        val_ds = make_dataset(val_split, False)
        test_ds = make_dataset(test_split, False)
        sample_x, sample_y = next(iter(train_ds))
        print(sample_x.shape, sample_y.shape)
        assert sample_x.shape[1:] == (*EFFICIENTNET_INPUT_SIZE, 3)
        """,
    ),
    cell(
        "code",
        """
        # Hücre 7 — EfficientNet model build
        @dataclass
        class AIDetectionResult:
            model_name: str
            is_forged: bool
            confidence: float
            class_label: str
            processing_time: float
            heatmap: np.ndarray | None = None
            overlay_image: np.ndarray | None = None
            error_message: str | None = None

        class EfficientNetForensicModel:
            def __init__(self, model_path: str | Path = EFFICIENTNET_MODEL_PATH) -> None:
                self.model_path = Path(model_path)
                self.model: Any | None = None
                self.input_size = EFFICIENTNET_INPUT_SIZE
                self.pretrained_weights_loaded = False
                self.backbone_name = ""

            def build_model(self, imagenet_weights: bool = False) -> Any:
                tf = self._tensorflow()
                from tensorflow.keras import Model, layers, optimizers, regularizers
                from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4
                backbone_cls = EfficientNetB0 if EFFICIENTNET_VARIANT == "B0" else EfficientNetB4
                local_weights = find_local_efficientnet_weights(EFFICIENTNET_VARIANT) if imagenet_weights else None
                weight_candidates = []
                if local_weights:
                    weight_candidates.append(local_weights)
                if imagenet_weights:
                    weight_candidates.append("imagenet")
                if not weight_candidates:
                    weight_candidates.append(None)

                base_model = None
                last_error = None
                for weights in weight_candidates:
                    try:
                        base_model = backbone_cls(weights=weights, include_top=False, input_shape=(*self.input_size, 3))
                        self.pretrained_weights_loaded = weights is not None
                        if weights is not None:
                            print("EfficientNet pretrained weights kaynağı:", weights)
                        break
                    except Exception as exc:
                        last_error = exc
                        print(f"{EFFICIENTNET_VARIANT} pretrained ağırlık yüklenemedi ({weights}): {exc}")

                if base_model is None:
                    if REQUIRE_PRETRAINED_BACKBONE and imagenet_weights:
                        raise RuntimeError(
                            "EfficientNet ImageNet ağırlığı yüklenemedi. Eğitim başlatılmadı. "
                            "Kaggle Notebook Internet ayarını aç veya input'a efficientnetb0_notop.h5 ağırlık dosyasını ekle."
                        ) from last_error
                    print(f"{EFFICIENTNET_VARIANT} weights=None ile devam ediliyor.")
                    base_model = backbone_cls(weights=None, include_top=False, input_shape=(*self.input_size, 3))
                    self.pretrained_weights_loaded = False

                self.backbone_name = base_model.name
                base_model.trainable = not self.pretrained_weights_loaded
                inputs = tf.keras.Input(shape=(*self.input_size, 3), name="image")
                x = layers.Rescaling(255.0, name="efficientnet_input_rescale")(inputs)
                x = base_model(x, training=not self.pretrained_weights_loaded)
                x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
                x = layers.BatchNormalization(name="bn_head")(x)
                x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_128")(x)
                x = layers.Dropout(0.45, name="dropout_128")(x)
                outputs = layers.Dense(1, activation="sigmoid", name="forgery_output")(x)
                model = Model(inputs=inputs, outputs=outputs, name="efficientnet_forensic")
                model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
                self.model = model
                return model

            def load_weights(self, path: str | Path | None = None) -> None:
                model_path = Path(path) if path is not None else self.model_path
                if not model_path.exists():
                    raise FileNotFoundError(f"EfficientNet model ağırlığı bulunamadı: {model_path}")
                tf = self._tensorflow()
                try:
                    self.model = tf.keras.models.load_model(str(model_path), compile=False)
                except Exception:
                    if self.model is None:
                        self.build_model(imagenet_weights=False)
                    self.model.load_weights(str(model_path))

            def is_available(self, path: str | Path | None = None) -> bool:
                model_path = Path(path) if path is not None else self.model_path
                return model_path.exists()

            def predict(self, image: np.ndarray) -> AIDetectionResult:
                if self.model is None:
                    raise RuntimeError("EfficientNet modeli yüklenmedi. Önce load_weights() çağrılmalı.")
                start = time.perf_counter()
                pred = float(np.asarray(self.model.predict(np.expand_dims(image.astype(np.float32), 0), verbose=0)).reshape(-1)[0])
                forged = pred >= CONFIDENCE_THRESHOLD
                conf = pred if forged else 1.0 - pred
                return AIDetectionResult("EfficientNet CNN", bool(forged), float(conf), CLASS_NAMES[int(forged)], time.perf_counter() - start)

            @staticmethod
            def unavailable_result(message: str) -> AIDetectionResult:
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
                    raise ImportError("TensorFlow yüklü değil. AI inference için requirements.txt kurulmalı.") from exc
                return tf

        efficientnet_wrapper = EfficientNetForensicModel(EFFICIENTNET_MODEL_PATH)
        efficientnet_model = efficientnet_wrapper.build_model(imagenet_weights=True)
        print("ImageNet pretrained yüklendi mi:", efficientnet_wrapper.pretrained_weights_loaded)
        print("Backbone layer:", efficientnet_wrapper.backbone_name)
        if REQUIRE_PRETRAINED_BACKBONE and not efficientnet_wrapper.pretrained_weights_loaded:
            raise RuntimeError("Pretrained EfficientNet yüklenmediği için eğitim durduruldu.")
        efficientnet_model.summary()
        """,
    ),
    cell(
        "code",
        """
        # Hücre 8 — Head eğitimi
        class EpochPrinter(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                print(f"Epoch {epoch+1}: val_accuracy={logs.get('val_accuracy', 0):.4f} | val_auc={logs.get('val_auc', 0):.4f} | val_loss={logs.get('val_loss', 0):.4f}")

        classes = np.array(sorted(set(train_split.labels)))
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(train_split.labels))
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        print("class_weight:", class_weight)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(str(EFFICIENTNET_BEST_PATH), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
            EpochPrinter(),
        ]
        head_epochs = 5 if efficientnet_wrapper.pretrained_weights_loaded else 3
        if efficientnet_wrapper.pretrained_weights_loaded:
            print("Head eğitimi: pretrained backbone donuk, classifier head öğreniyor.")
        else:
            print("ImageNet yok: backbone rastgele olduğu için donuk bırakılmadı; kısa warmup tüm modeli eğitiyor.")
        history_head = efficientnet_model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=callbacks, class_weight=class_weight, verbose=1)
        """,
    ),
    cell(
        "code",
        """
        # Hücre 9 — Fine-tune
        base_model = efficientnet_model.get_layer(efficientnet_wrapper.backbone_name)
        if efficientnet_wrapper.pretrained_weights_loaded:
            base_model.trainable = True
            for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
                layer.trainable = False
            for layer in base_model.layers[-FINE_TUNE_LAYERS:]:
                layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
            fine_tune_lr = LEARNING_RATE / 10
            print("Fine-tune: pretrained backbone son", FINE_TUNE_LAYERS, "katman.")
        else:
            base_model.trainable = True
            fine_tune_lr = LEARNING_RATE
            print("Fine-tune: pretrained yok, tüm backbone eğitimde kalıyor.")
        efficientnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
        history_finetune = efficientnet_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_EFFICIENTNET, callbacks=callbacks, class_weight=class_weight, verbose=1)

        efficientnet_model = load_model_compat(EFFICIENTNET_BEST_PATH)
        efficientnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        merged = {}
        for history in [history_head, history_finetune]:
            for key, values in history.history.items():
                merged.setdefault(key, []).extend(values)
        for key in [k for k in merged if "loss" in k]:
            axes[0].plot(merged[key], label=key)
        axes[0].legend(); axes[0].set_title("Loss")
        for key in [k for k in merged if "accuracy" in k or "auc" in k]:
            axes[1].plot(merged[key], label=key)
        axes[1].legend(); axes[1].set_title("Metrics")
        plt.tight_layout(); plt.show()
        """,
    ),
    cell(
        "code",
        """
        # Hücre 10 — Değerlendirme ve threshold seçimi
        val_true = np.array(val_split.labels)
        val_scores = efficientnet_model.predict(val_ds, verbose=1).reshape(-1)
        threshold_grid = np.linspace(0.20, 0.80, 61)
        best_threshold, best_val_f1 = max(
            [(float(th), float(f1_score(val_true, (val_scores >= th).astype(int), zero_division=0))) for th in threshold_grid],
            key=lambda item: item[1],
        )
        print(f"Validation en iyi threshold: {best_threshold:.2f} | val_f1={best_val_f1:.4f}")

        y_true = np.array(test_split.labels)
        y_scores = efficientnet_model.predict(test_ds, verbose=1).reshape(-1)
        y_pred = (y_scores >= best_threshold).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auc": float(roc_auc_score(y_true, y_scores)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "threshold": float(best_threshold),
            "backbone_variant": EFFICIENTNET_VARIANT,
            "pretrained_weights_loaded": bool(efficientnet_wrapper.pretrained_weights_loaded),
            "mask_guided_crops": bool(USE_MASK_GUIDED_CROPS),
            "train_mask_matches": int(sum(1 for label, mask in zip(train_split.labels, train_split.mask_paths) if label == CLASS_TAMPERED and mask)),
        }
        training_results["efficientnet"].update(metrics)
        print(metrics)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.xticks([0, 1], CLASS_NAMES); plt.yticks([0, 1], CLASS_NAMES)
        for r in range(2):
            for c in range(2):
                plt.text(c, r, cm[r, c], ha="center", va="center")
        plt.title("EfficientNet CNN Confusion Matrix")
        plt.colorbar(); plt.tight_layout(); plt.show()
        """,
    ),
    cell(
        "code",
        """
        # Hücre 11 — Kaydet ve doğrula
        efficientnet_model.save_weights(str(EFFICIENTNET_WEIGHTS_PATH))
        EFFICIENTNET_MODEL_PATH.write_bytes(EFFICIENTNET_WEIGHTS_PATH.read_bytes())
        assert EFFICIENTNET_MODEL_PATH.exists()
        training_results["files"]["efficientnet_finetuned.h5"] = {"path": str(EFFICIENTNET_MODEL_PATH), "size_mb": file_size_mb(EFFICIENTNET_MODEL_PATH)}
        training_results["files"]["efficientnet_finetuned.weights.h5"] = {"path": str(EFFICIENTNET_WEIGHTS_PATH), "size_mb": file_size_mb(EFFICIENTNET_WEIGHTS_PATH)}

        del efficientnet_model
        tf.keras.backend.clear_session()
        validation_wrapper = EfficientNetForensicModel(EFFICIENTNET_MODEL_PATH)
        reloaded = validation_wrapper.build_model(imagenet_weights=False)
        reloaded.load_weights(str(EFFICIENTNET_MODEL_PATH))
        sample_batch, sample_labels = next(iter(test_ds.take(1)))
        sample_pred = float(reloaded.predict(sample_batch[:1], verbose=0).reshape(-1)[0])
        print("Reload tek örnek tahmini:", sample_pred, "| gerçek label:", int(sample_labels[0].numpy()))
        assert 0.0 <= sample_pred <= 1.0
        save_results()
        print("Model ağırlıkları başarıyla kaydedildi:", EFFICIENTNET_MODEL_PATH)
        display(pd.DataFrame([{"model": "EfficientNet CNN", **training_results["efficientnet"]}]))
        display(pd.DataFrame([{"file": name, **info} for name, info in training_results["files"].items()]))
        """,
    ),
    cell(
        "markdown",
        """
        # İndirme

        Kaggle Output sekmesinden şunları indir:

        - `efficientnet_finetuned.h5`
        - `efficientnet_training_results.json`

        Projede şu konuma koy:

        - `efficientnet_finetuned.h5` → `data/models/efficientnet_finetuned.h5`

        Not: `efficientnet_finetuned.h5` dosyası Keras full-model değil, weight-only HDF5 dosyasıdır.
        IFDS uygulaması önce full model yüklemeyi dener; olmazsa aynı mimariyi kurup bu ağırlıkları yükler.

        Böylece AI tarafında iki algoritma olur:

        - `xception_finetuned.h5`
        - `efficientnet_finetuned.h5`

        Bu notebook özellikle Kaggle datasetindeki `CASIA 2 Groundtruth(5123 files)` klasörünü kullanacak şekilde ayarlıdır.
        Hücre 5'te `Tampered mask eşleşmesi` değeri sıfırdan büyük görünmelidir.
        """,
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks") / "IFDS_Kaggle_EfficientNet_CNN_Egitimi.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
print("cells:", len(cells))
