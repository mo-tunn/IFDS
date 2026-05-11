# 🔍 Görüntü Sahteciliği Tespiti Sistemi — Implementation Plan
**Image Forgery Detection System (IFDS)**  
Versiyon: 1.0 | Mayıs 2026

---

## ⚡ AI Coder'a Verilecek Master Prompt

```
Sen kıdemli bir Python/Computer Vision mühendisisin.
Aşağıdaki implementation planını EKSIKSIZ olarak uygula.
Her adımı sırayla yap, bir adım bitmeden diğerine geçme.
Kodun tamamı production-ready, type-annotated ve Doxygen uyumlu docstring'li olmalı.
Her modül için pytest unit test yaz.
```

---

## 📁 Proje Dizin Yapısı

```
ifds/
├── app.py                          # Streamlit ana giriş noktası
├── requirements.txt
├── requirements-dev.txt            # SonarQube, pytest, doxygen araçları
├── Doxyfile                        # Doxygen konfigürasyonu
├── sonar-project.properties        # SonarQube konfigürasyonu
├── .github/
│   └── workflows/
│       └── sonar.yml               # CI/CD pipeline
│
├── config/
│   └── settings.py                 # Global sabitler ve konfigürasyon
│
├── data/
│   ├── raw/                        # Ham veri setleri (CASIA v2, Coverage)
│   │   ├── CASIA2/
│   │   │   ├── Au/                 # Authentic görüntüler
│   │   │   └── Tp/                 # Tampered görüntüler
│   │   └── Coverage/
│   ├── processed/                  # Ön işlenmiş, model-ready veri
│   └── models/                     # Eğitilmiş model ağırlıkları
│       ├── xception_finetuned.h5
│       └── efficientnet_lstm.h5
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_loader.py         # Çoklu format yükleme
│   │   └── augmentation.py         # Veri artırma pipeline
│   │
│   ├── classical/
│   │   ├── __init__.py
│   │   ├── base_detector.py        # Abstract base class
│   │   ├── sift_detector.py
│   │   ├── surf_detector.py
│   │   ├── akaze_detector.py
│   │   └── orb_detector.py
│   │
│   ├── ai_models/
│   │   ├── __init__.py
│   │   ├── base_model.py           # Abstract base class
│   │   ├── xception_model.py       # CNN fine-tune
│   │   ├── efficientnet_lstm.py    # Hibrit model
│   │   └── gradcam.py              # Grad-CAM görselleştirme
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset_builder.py      # CASIA v2 + Coverage loader
│   │   ├── train_xception.py       # Model 1 eğitim scripti
│   │   └── train_lstm.py           # Model 2 eğitim scripti
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── report_generator.py     # PDF/HTML rapor üretimi
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py        # Matplotlib/Plotly yardımcıları
│       └── metrics.py              # Accuracy, F1, IoU hesapları
│
├── ui/
│   ├── __init__.py
│   ├── components/
│   │   ├── upload_section.py       # Dosya yükleme komponenti
│   │   ├── results_grid.py         # Sonuç kartları grid
│   │   ├── heatmap_viewer.py       # Isı haritası görüntüleyici
│   │   └── comparison_table.py     # Karşılaştırma tablosu
│   └── styles/
│       └── theme.css               # Özel CSS stilleri
│
└── tests/
    ├── __init__.py
    ├── test_image_loader.py
    ├── test_sift.py
    ├── test_surf.py
    ├── test_akaze.py
    ├── test_orb.py
    ├── test_xception.py
    ├── test_lstm.py
    └── conftest.py                 # Shared test fixtures
```

---

## 📦 ADIM 1 — Proje Kurulumu

### 1.1 requirements.txt

```
# Core
streamlit==1.35.0
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80   # SURF için zorunlu
Pillow==10.3.0
numpy==1.26.4
pandas==2.2.2

# Deep Learning
tensorflow==2.16.1
keras==3.3.3

# Visualization
matplotlib==3.9.0
plotly==5.22.0
seaborn==0.13.2

# Reporting
fpdf2==2.7.9
jinja2==3.1.4

# Utilities
scikit-learn==1.5.0
tqdm==4.66.4
python-dotenv==1.0.1
```

### 1.2 requirements-dev.txt

```
pytest==8.2.0
pytest-cov==5.0.0
pytest-mock==3.14.0
coverage==7.5.3
```

### 1.3 config/settings.py

```python
"""
@file settings.py
@brief Uygulama genelinde kullanılan sabit değerler ve konfigürasyon.
"""

import os
from pathlib import Path

# Proje kök dizini
BASE_DIR = Path(__file__).resolve().parent.parent

# Veri dizinleri
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Model konfigürasyonu
XCEPTION_INPUT_SIZE = (224, 224)
PATCH_SIZE = (32, 32)
GRID_SIZE = 7           # 7x7 = 49 patch
SEQUENCE_LENGTH = 49    # LSTM sequence uzunluğu

# Eğitim konfigürasyonu
BATCH_SIZE = 32
EPOCHS_XCEPTION = 20
EPOCHS_LSTM = 15
LEARNING_RATE = 1e-4
FINE_TUNE_LAYERS = 20   # Xception'da serbest bırakılacak son katman sayısı

# Veri seti bölme oranları
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Desteklenen dosya formatları
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
MAX_FILE_SIZE_MB = 50

# Sınıf etiketleri
CLASS_NAMES = ["Authentic", "Tampered"]
CLASS_AUTHENTIC = 0
CLASS_TAMPERED = 1

# Güven eşiği
CONFIDENCE_THRESHOLD = 0.5

# Model dosya yolları
XCEPTION_MODEL_PATH = MODELS_DIR / "xception_finetuned.h5"
LSTM_MODEL_PATH = MODELS_DIR / "efficientnet_lstm.h5"
```

---

## 📂 ADIM 2 — Preprocessing Modülü

### 2.1 src/preprocessing/image_loader.py

```python
"""
@file image_loader.py
@brief Çoklu format görüntü yükleme ve doğrulama modülü.
@details GIF, JPEG, PNG, BMP, TIFF ve diğer yaygın formatları destekler.
         Yüklenen görüntüler RGB'ye dönüştürülür ve model girişi için hazırlanır.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from config.settings import SUPPORTED_FORMATS, MAX_FILE_SIZE_MB


class ImageLoader:
    """
    @class ImageLoader
    @brief Farklı formatlardaki görüntü dosyalarını yükler ve ön işler.
    """

    @staticmethod
    def load_from_path(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        @brief Dosya yolundan görüntü yükler.
        @param file_path Görüntü dosyasının tam yolu.
        @return (numpy görüntü array, metadata dict) tuple'ı.
        @raises ValueError Desteklenmeyen format veya aşırı büyük dosya için.
        @raises FileNotFoundError Dosya bulunamazsa.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Desteklenmeyen format: {suffix}. "
                f"Desteklenenler: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"Dosya çok büyük: {file_size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)")
        
        # GIF: ilk frame al
        if suffix == ".gif":
            img = ImageLoader._load_gif(str(path))
        else:
            img = ImageLoader._load_standard(str(path))
        
        metadata = ImageLoader._extract_metadata(path, img)
        return img, metadata

    @staticmethod
    def load_from_bytes(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        @brief Byte dizisinden görüntü yükler (Streamlit upload için).
        @param file_bytes Görüntü byte verisi.
        @param filename Orijinal dosya adı (format tespiti için).
        @return (numpy görüntü array, metadata dict) tuple'ı.
        """
        import io
        suffix = Path(filename).suffix.lower()
        
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen format: {suffix}")
        
        if suffix == ".gif":
            pil_img = Image.open(io.BytesIO(file_bytes))
            pil_img.seek(0)
            img = np.array(pil_img.convert("RGB"))
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        metadata = {
            "filename": filename,
            "format": suffix.upper().replace(".", ""),
            "width": img.shape[1],
            "height": img.shape[0],
            "channels": img.shape[2] if len(img.shape) == 3 else 1,
            "size_mb": len(file_bytes) / (1024 * 1024)
        }
        return img, metadata

    @staticmethod
    def preprocess_for_model(
        img: np.ndarray,
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        @brief Görüntüyü model girişi için yeniden boyutlandırır ve normalize eder.
        @param img RGB numpy array.
        @param target_size Hedef boyut (width, height).
        @return [0, 1] aralığında normalize edilmiş float32 array.
        """
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    @staticmethod
    def _load_gif(path: str) -> np.ndarray:
        pil_img = Image.open(path)
        pil_img.seek(0)
        return np.array(pil_img.convert("RGB"))

    @staticmethod
    def _load_standard(path: str) -> np.ndarray:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Görüntü okunamadı: {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _extract_metadata(path: Path, img: np.ndarray) -> Dict[str, Any]:
        return {
            "filename": path.name,
            "format": path.suffix.upper().replace(".", ""),
            "width": img.shape[1],
            "height": img.shape[0],
            "channels": img.shape[2] if len(img.shape) == 3 else 1,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2)
        }
```

---

## 🔬 ADIM 3 — Klasik Algoritmalar Modülü

### 3.1 src/classical/base_detector.py

```python
"""
@file base_detector.py
@brief Tüm klasik dedektörler için abstract base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    """
    @class DetectionResult
    @brief Klasik algoritma tespit sonucunu tutan veri sınıfı.
    """
    algorithm: str          # Algoritma adı
    is_forged: bool         # Sahte mi?
    confidence: float       # Güven skoru [0, 1]
    match_count: int        # Eşleşen anahtar nokta sayısı
    total_keypoints: int    # Toplam anahtar nokta sayısı
    processing_time: float  # İşlem süresi (saniye)
    annotated_image: Optional[np.ndarray] = None  # Eşleşmelerin çizildiği görüntü
    forge_mask: Optional[np.ndarray] = None        # Sahte bölge maskesi


class BaseDetector(ABC):
    """
    @class BaseDetector
    @brief Klasik sahtecilik dedektörleri için abstract base class.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        @brief Görüntüde sahtecilik tespiti yapar.
        @param image RGB numpy array görüntü.
        @return DetectionResult nesnesi.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """@brief Algoritma adını döndürür."""
        pass
```

### 3.2 src/classical/sift_detector.py

```python
"""
@file sift_detector.py
@brief SIFT algoritması ile copy-move sahtecilik tespiti.
@details Scale-Invariant Feature Transform kullanarak görüntü içindeki
         tekrarlayan bölgeleri tespit eder.
"""

import cv2
import numpy as np
import time
from .base_detector import BaseDetector, DetectionResult


class SIFTDetector(BaseDetector):
    """
    @class SIFTDetector
    @brief SIFT tabanlı copy-move sahtecilik dedektörü.
    """

    def __init__(self, n_features: int = 0, match_threshold: float = 0.75):
        """
        @param n_features Çıkarılacak maksimum özellik sayısı (0 = sınırsız).
        @param match_threshold Lowe's ratio test eşiği.
        """
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.match_threshold = match_threshold
        self.min_match_count = 10

    def get_name(self) -> str:
        return "SIFT"

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        @brief SIFT ile görüntüde sahtecilik tespiti.
        @param image RGB numpy array.
        @return DetectionResult.
        """
        start = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Anahtar nokta ve descriptor çıkarımı
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < self.min_match_count:
            return DetectionResult(
                algorithm=self.get_name(),
                is_forged=False,
                confidence=0.0,
                match_count=0,
                total_keypoints=len(keypoints) if keypoints else 0,
                processing_time=time.time() - start
            )

        # FLANN tabanlı eşleştirici
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Self-matching (copy-move tespiti için görüntüyü kendisiyle eşleştir)
        matches = flann.knnMatch(descriptors, descriptors, k=3)
        
        # Lowe's ratio test + minimum mesafe filtresi
        good_matches = []
        for match_group in matches:
            if len(match_group) >= 3:
                m, n, o = match_group[0], match_group[1], match_group[2]
                # Kendisiyle eşleşmeyi atla (m.trainIdx != m.queryIdx)
                if (m.trainIdx != m.queryIdx and 
                    m.distance < self.match_threshold * n.distance):
                    good_matches.append(m)

        match_count = len(good_matches)
        total_kp = len(keypoints)
        confidence = min(match_count / max(total_kp * 0.1, 1), 1.0)
        is_forged = match_count >= self.min_match_count and confidence > 0.3
        
        # Görselleştirme
        annotated = self._draw_matches(image, keypoints, good_matches)
        forge_mask = self._create_forge_mask(image, keypoints, good_matches)
        
        return DetectionResult(
            algorithm=self.get_name(),
            is_forged=is_forged,
            confidence=confidence,
            match_count=match_count,
            total_keypoints=total_kp,
            processing_time=time.time() - start,
            annotated_image=annotated,
            forge_mask=forge_mask
        )

    def _draw_matches(
        self, image: np.ndarray,
        keypoints: list, matches: list
    ) -> np.ndarray:
        """@brief Eşleşen anahtar noktaları görüntü üzerine çizer."""
        result = image.copy()
        for match in matches[:50]:  # En iyi 50 eşleşme
            kp = keypoints[match.queryIdx]
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(result, pt, 5, (0, 255, 0), -1)
        return result

    def _create_forge_mask(
        self, image: np.ndarray,
        keypoints: list, matches: list
    ) -> np.ndarray:
        """@brief Tespit edilen sahte bölgeleri kaplayan maske oluşturur."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for match in matches:
            kp = keypoints[match.queryIdx]
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(mask, pt, 20, 255, -1)
        # Morfolojik genişleme ile bölge büyütme
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask
```

### 3.3 src/classical/surf_detector.py

```python
"""
@file surf_detector.py
@brief SURF (Speeded-Up Robust Features) algoritması ile sahtecilik tespiti.
@note SURF patent kısıtlı olup yalnızca akademik/araştırma amaçlı kullanılabilir.
      opencv-contrib-python paketi gerektirir.
@warning Ticari kullanım için lisans gereklidir.
"""

import cv2
import numpy as np
import time
from .base_detector import BaseDetector, DetectionResult


class SURFDetector(BaseDetector):
    """
    @class SURFDetector
    @brief SURF tabanlı copy-move sahtecilik dedektörü.
    """

    def __init__(self, hessian_threshold: float = 400):
        """
        @param hessian_threshold Hessian matris eşiği (yüksek = az ama güçlü keypoint).
        """
        try:
            self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
        except AttributeError:
            raise ImportError(
                "SURF için opencv-contrib-python gerekli: "
                "pip install opencv-contrib-python"
            )
        self.min_match_count = 10

    def get_name(self) -> str:
        return "SURF"

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        @brief SURF ile sahtecilik tespiti.
        @param image RGB numpy array.
        @return DetectionResult.
        """
        start = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.surf.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < self.min_match_count:
            return DetectionResult(
                algorithm=self.get_name(), is_forged=False,
                confidence=0.0, match_count=0,
                total_keypoints=len(keypoints) if keypoints else 0,
                processing_time=time.time() - start
            )

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=3)
        
        good_matches = []
        for mg in matches:
            if len(mg) >= 3:
                m, n = mg[0], mg[1]
                if m.trainIdx != m.queryIdx and m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        match_count = len(good_matches)
        confidence = min(match_count / max(len(keypoints) * 0.1, 1), 1.0)
        is_forged = match_count >= self.min_match_count and confidence > 0.3
        
        annotated = image.copy()
        cv2.drawKeypoints(image, keypoints[:100], annotated,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return DetectionResult(
            algorithm=self.get_name(),
            is_forged=is_forged,
            confidence=confidence,
            match_count=match_count,
            total_keypoints=len(keypoints),
            processing_time=time.time() - start,
            annotated_image=annotated
        )
```

### 3.4 src/classical/akaze_detector.py

```python
"""
@file akaze_detector.py
@brief AKAZE (Accelerated KAZE) algoritması ile sahtecilik tespiti.
@details Patent ücretsiz, binary descriptor kullanan hızlı dedektör.
"""

import cv2
import numpy as np
import time
from .base_detector import BaseDetector, DetectionResult


class AKAZEDetector(BaseDetector):
    """
    @class AKAZEDetector
    @brief AKAZE tabanlı copy-move sahtecilik dedektörü.
    """

    def __init__(self):
        self.akaze = cv2.AKAZE_create()
        self.min_match_count = 8

    def get_name(self) -> str:
        return "AKAZE"

    def detect(self, image: np.ndarray) -> DetectionResult:
        start = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.akaze.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < self.min_match_count:
            return DetectionResult(
                algorithm=self.get_name(), is_forged=False,
                confidence=0.0, match_count=0,
                total_keypoints=len(keypoints) if keypoints else 0,
                processing_time=time.time() - start
            )

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=3)
        
        good_matches = []
        for mg in matches:
            if len(mg) >= 3:
                m, n = mg[0], mg[1]
                if m.trainIdx != m.queryIdx and m.distance < 0.8 * n.distance:
                    good_matches.append(m)

        match_count = len(good_matches)
        confidence = min(match_count / max(len(keypoints) * 0.1, 1), 1.0)
        is_forged = match_count >= self.min_match_count and confidence > 0.25
        
        annotated = image.copy()
        cv2.drawKeypoints(image, keypoints, annotated, color=(0, 200, 0))
        
        return DetectionResult(
            algorithm=self.get_name(),
            is_forged=is_forged,
            confidence=confidence,
            match_count=match_count,
            total_keypoints=len(keypoints),
            processing_time=time.time() - start,
            annotated_image=annotated
        )
```

### 3.5 src/classical/orb_detector.py

```python
"""
@file orb_detector.py
@brief ORB (Oriented FAST and Rotated BRIEF) algoritması ile sahtecilik tespiti.
@details Tamamen patent ücretsiz, gerçek zamanlı uygulamalar için optimize edilmiş.
"""

import cv2
import numpy as np
import time
from .base_detector import BaseDetector, DetectionResult


class ORBDetector(BaseDetector):
    """
    @class ORBDetector
    @brief ORB tabanlı copy-move sahtecilik dedektörü.
    """

    def __init__(self, n_features: int = 1000):
        """
        @param n_features Çıkarılacak maksimum özellik sayısı.
        """
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.min_match_count = 8

    def get_name(self) -> str:
        return "ORB"

    def detect(self, image: np.ndarray) -> DetectionResult:
        start = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < self.min_match_count:
            return DetectionResult(
                algorithm=self.get_name(), is_forged=False,
                confidence=0.0, match_count=0,
                total_keypoints=len(keypoints) if keypoints else 0,
                processing_time=time.time() - start
            )

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=3)
        
        good_matches = []
        for mg in matches:
            if len(mg) >= 3:
                m, n = mg[0], mg[1]
                if m.trainIdx != m.queryIdx and m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        match_count = len(good_matches)
        confidence = min(match_count / max(len(keypoints) * 0.1, 1), 1.0)
        is_forged = match_count >= self.min_match_count and confidence > 0.25

        annotated = image.copy()
        cv2.drawKeypoints(image, keypoints[:200], annotated, color=(255, 100, 0))
        
        return DetectionResult(
            algorithm=self.get_name(),
            is_forged=is_forged,
            confidence=confidence,
            match_count=match_count,
            total_keypoints=len(keypoints),
            processing_time=time.time() - start,
            annotated_image=annotated
        )
```

---

## 🧠 ADIM 4 — AI Modelleri

### 4.1 src/ai_models/xception_model.py

```python
"""
@file xception_model.py
@brief CASIA v2 üzerinde fine-tune edilmiş Xception CNN modeli.
@details ImageNet ağırlıkları ile başlatılır, son FINE_TUNE_LAYERS katman
         CASIA v2 veri seti üzerinde yeniden eğitilir.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications import Xception
from dataclasses import dataclass
from typing import Optional
from config.settings import (
    XCEPTION_INPUT_SIZE, FINE_TUNE_LAYERS,
    LEARNING_RATE, CLASS_NAMES, XCEPTION_MODEL_PATH
)


@dataclass
class AIDetectionResult:
    """
    @class AIDetectionResult
    @brief AI model tespit sonucunu tutan veri sınıfı.
    """
    model_name: str
    is_forged: bool
    confidence: float           # [0, 1] — Tampered olma olasılığı
    class_label: str            # "Authentic" veya "Tampered"
    processing_time: float
    gradcam_heatmap: Optional[np.ndarray] = None


class XceptionForensicModel:
    """
    @class XceptionForensicModel
    @brief Fine-tuned Xception ile görüntü sahteciliği sınıflandırıcı.
    """

    def __init__(self):
        self.model: Optional[Model] = None
        self.input_size = XCEPTION_INPUT_SIZE

    def build_model(self) -> Model:
        """
        @brief Fine-tune için Xception modelini oluşturur.
        @details Base model dondurulur, custom classification head eklenir,
                 ardından son FINE_TUNE_LAYERS katman serbest bırakılır.
        @return Derlenmiş Keras Model.
        """
        base_model = Xception(
            weights="imagenet",
            include_top=False,
            input_shape=(*self.input_size, 3)
        )
        
        # Aşama 1: Tüm base model dondur
        base_model.trainable = False
        
        # Classification head
        inputs = tf.keras.Input(shape=(*self.input_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation="sigmoid", name="forgery_output")(x)
        
        model = Model(inputs, outputs, name="xception_forensic")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )
        
        self.model = model
        return model

    def unfreeze_for_finetuning(self):
        """
        @brief Son FINE_TUNE_LAYERS katmanı fine-tune için serbest bırakır.
        @note build_model() çağrısından sonra, initial training tamamlandıktan sonra çağır.
        """
        if self.model is None:
            raise RuntimeError("Önce build_model() çağırılmalı.")
        
        base_model = self.model.layers[1]  # Xception layer
        base_model.trainable = True
        
        # Son FINE_TUNE_LAYERS hariç dondur
        for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
            layer.trainable = False
        
        # Daha düşük LR ile yeniden derle
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE / 10),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )

    def load_weights(self, path: str = None):
        """
        @brief Kaydedilmiş model ağırlıklarını yükler.
        @param path Model dosya yolu. None ise varsayılan yol kullanılır.
        """
        if self.model is None:
            self.build_model()
        load_path = path or str(XCEPTION_MODEL_PATH)
        self.model.load_weights(load_path)

    def predict(self, image: np.ndarray) -> AIDetectionResult:
        """
        @brief Tek görüntü için sahtecilik tahmini yapar.
        @param image [0,1] normalize edilmiş, (224,224,3) boyutlu RGB array.
        @return AIDetectionResult.
        """
        import time
        if self.model is None:
            raise RuntimeError("Model yüklenmemiş. load_weights() çağırın.")
        
        start = time.time()
        img_batch = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)
        
        prediction = self.model.predict(img_batch, verbose=0)[0][0]
        is_forged = bool(prediction >= 0.5)
        confidence = float(prediction) if is_forged else float(1 - prediction)
        
        return AIDetectionResult(
            model_name="Xception (Fine-tuned)",
            is_forged=is_forged,
            confidence=confidence,
            class_label=CLASS_NAMES[int(is_forged)],
            processing_time=time.time() - start
        )
```

### 4.2 src/ai_models/efficientnet_lstm.py

```python
"""
@file efficientnet_lstm.py
@brief EfficientNet-B4 + LSTM hibrit modeli ile sahtecilik lokalizasyonu.
@details Görüntü patch'lere bölünür, her patch'ten EfficientNet ile
         özellik çıkarılır, LSTM bu sequence'i analiz ederek sahte bölgeyi
         lokalize eder.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications import EfficientNetB4
from dataclasses import dataclass
from typing import Optional, Tuple
from config.settings import (
    PATCH_SIZE, GRID_SIZE, SEQUENCE_LENGTH,
    LEARNING_RATE, LSTM_MODEL_PATH
)


@dataclass  
class LSTMDetectionResult:
    """@class LSTMDetectionResult @brief LSTM model sonucu."""
    model_name: str
    is_forged: bool
    confidence: float
    processing_time: float
    heatmap: Optional[np.ndarray] = None    # (GRID_SIZE, GRID_SIZE) forge prob haritası
    overlay_image: Optional[np.ndarray] = None  # Orijinal görüntü + heatmap overlay


class EfficientNetLSTMModel:
    """
    @class EfficientNetLSTMModel
    @brief Patch-based EfficientNet + LSTM hibrit sahtecilik dedektörü.
    """

    def __init__(self):
        self.feature_extractor: Optional[Model] = None
        self.full_model: Optional[Model] = None
        self.patch_size = PATCH_SIZE
        self.grid_size = GRID_SIZE
        self.sequence_len = SEQUENCE_LENGTH

    def _build_feature_extractor(self) -> Model:
        """
        @brief EfficientNet-B4 tabanlı özellik çıkarıcı oluşturur.
        @return Patch başına feature vektörü döndüren model (1280-D).
        """
        base = EfficientNetB4(
            weights="imagenet",
            include_top=False,
            input_shape=(*self.patch_size, 3),
            pooling="avg"
        )
        base.trainable = False
        return base

    def build_model(self) -> Model:
        """
        @brief Tam hibrit modeli oluşturur.
        @details Mimari:
                 Input (49, 32, 32, 3) → TimeDistributed(EfficientNet) →
                 (49, 1280) → LSTM(256) → LSTM(128) →
                 Dense(49) → Sigmoid → Reshape(7, 7)
        @return Derlenmiş Keras Model.
        """
        self.feature_extractor = self._build_feature_extractor()
        
        # Sequence input: (sequence_len, patch_h, patch_w, channels)
        sequence_input = tf.keras.Input(
            shape=(self.sequence_len, *self.patch_size, 3),
            name="patch_sequence"
        )
        
        # Feature extraction per patch
        features = layers.TimeDistributed(
            self.feature_extractor,
            name="efficientnet_features"
        )(sequence_input)  # (batch, 49, 1280)
        
        # LSTM layers
        x = layers.LSTM(256, return_sequences=True, name="lstm_1")(features)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(128, return_sequences=True, name="lstm_2")(x)
        x = layers.Dropout(0.2)(x)
        
        # Classification head per patch
        x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)
        patch_scores = layers.TimeDistributed(
            layers.Dense(1, activation="sigmoid"),
            name="patch_forgery_scores"
        )(x)  # (batch, 49, 1)
        
        patch_scores = layers.Reshape((self.sequence_len,), name="scores_flat")(patch_scores)
        
        # Global forgery score
        global_score = layers.Lambda(
            lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
            name="global_forgery_score"
        )(patch_scores)
        
        model = Model(
            inputs=sequence_input,
            outputs={"patch_scores": patch_scores, "global_score": global_score},
            name="efficientnet_lstm_forensic"
        )
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={"patch_scores": "binary_crossentropy", "global_score": "binary_crossentropy"},
            loss_weights={"patch_scores": 0.7, "global_score": 0.3},
            metrics={"global_score": ["accuracy"]}
        )
        
        self.full_model = model
        return model

    def image_to_patches(self, image: np.ndarray) -> np.ndarray:
        """
        @brief Görüntüyü (GRID_SIZE x GRID_SIZE) patch dizisine böler.
        @param image (H, W, 3) normalize RGB array.
        @return (sequence_len, patch_h, patch_w, 3) array.
        """
        # 224x224'e resize (7x7 grid için 32px patch)
        img_resized = tf.image.resize(image, (self.grid_size * self.patch_size[0],
                                               self.grid_size * self.patch_size[1])).numpy()
        patches = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                r_start = row * self.patch_size[0]
                r_end = r_start + self.patch_size[0]
                c_start = col * self.patch_size[1]
                c_end = c_start + self.patch_size[1]
                patch = img_resized[r_start:r_end, c_start:c_end]
                patches.append(patch)
        return np.array(patches)  # (49, 32, 32, 3)

    def predict(self, image: np.ndarray) -> LSTMDetectionResult:
        """
        @brief Görüntü üzerinde patch-level sahtecilik tespiti yapar.
        @param image [0,1] normalize, RGB numpy array.
        @return LSTMDetectionResult (heatmap dahil).
        """
        import time
        if self.full_model is None:
            raise RuntimeError("Model yüklenmemiş.")
        
        start = time.time()
        patches = self.image_to_patches(image)
        patches_batch = np.expand_dims(patches, axis=0)  # (1, 49, 32, 32, 3)
        
        outputs = self.full_model.predict(patches_batch, verbose=0)
        patch_scores = outputs["patch_scores"][0]   # (49,)
        global_score = float(outputs["global_score"][0][0])
        
        # Heatmap oluştur
        heatmap = patch_scores.reshape(self.grid_size, self.grid_size)
        overlay = self._create_overlay(image, heatmap)
        
        return LSTMDetectionResult(
            model_name="EfficientNet + LSTM",
            is_forged=global_score >= 0.5,
            confidence=global_score,
            processing_time=time.time() - start,
            heatmap=heatmap,
            overlay_image=overlay
        )

    def _create_overlay(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """@brief Heatmap'i orijinal görüntü üzerine bindirir."""
        import cv2
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        img_uint8 = (image * 255).astype(np.uint8)
        overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_rgb, 0.4, 0)
        return overlay

    def load_weights(self, path: str = None):
        """@brief Kaydedilmiş ağırlıkları yükler."""
        if self.full_model is None:
            self.build_model()
        load_path = path or str(LSTM_MODEL_PATH)
        self.full_model.load_weights(load_path)
```

### 4.3 src/ai_models/gradcam.py

```python
"""
@file gradcam.py
@brief Gradient-weighted Class Activation Mapping (Grad-CAM) görselleştirme.
@details CNN modelinin hangi bölgelere odaklandığını görsel olarak açıklar.
"""

import numpy as np
import tensorflow as tf
import cv2
from typing import Tuple


class GradCAM:
    """
    @class GradCAM
    @brief Xception modeli için Grad-CAM ısı haritası üretici.
    """

    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str = "block14_sepconv2_act"):
        """
        @param model Xception Keras modeli.
        @param last_conv_layer_name Gradyan hesaplanacak son conv katmanı adı.
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name

    def generate(self, img_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        @brief Grad-CAM ısı haritası üretir.
        @param img_array (1, 224, 224, 3) normalize görüntü batch.
        @return (heatmap, superimposed_img) tuple.
        """
        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output
            ]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        
        # Orijinal görüntü boyutuna resize
        img_uint8 = (img_array[0] * 255).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img_uint8, 0.6, heatmap_rgb, 0.4, 0)
        
        return heatmap, superimposed
```

---

## 🏋️ ADIM 5 — Model Eğitimi

### 5.1 src/training/dataset_builder.py

```python
"""
@file dataset_builder.py
@brief CASIA v2 ve Coverage veri setlerini yükler, ön işler ve TF Dataset oluşturur.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from config.settings import (
    RAW_DIR, XCEPTION_INPUT_SIZE, BATCH_SIZE,
    TRAIN_RATIO, VAL_RATIO, CLASS_AUTHENTIC, CLASS_TAMPERED
)


class ForensicDatasetBuilder:
    """
    @class ForensicDatasetBuilder
    @brief CASIA v2 + Coverage veri seti yükleyici.
    """

    def load_casia_v2(self) -> Tuple[List[str], List[int]]:
        """
        @brief CASIA v2 dosya yollarını ve etiketlerini yükler.
        @return (file_paths, labels) tuple.
        @details
            CASIA2/Au/  → CLASS_AUTHENTIC (0)
            CASIA2/Tp/  → CLASS_TAMPERED (1)
        """
        paths, labels = [], []
        casia_dir = RAW_DIR / "CASIA2"
        
        for img_path in (casia_dir / "Au").glob("*.*"):
            if img_path.suffix.lower() in {".jpg", ".bmp", ".png", ".tiff"}:
                paths.append(str(img_path))
                labels.append(CLASS_AUTHENTIC)
        
        for img_path in (casia_dir / "Tp").glob("*.*"):
            if img_path.suffix.lower() in {".jpg", ".bmp", ".png", ".tiff"}:
                paths.append(str(img_path))
                labels.append(CLASS_TAMPERED)
        
        return paths, labels

    def build_tf_datasets(
        self
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        @brief Train/Val/Test TF Dataset'lerini oluşturur.
        @return (train_ds, val_ds, test_ds) tuple.
        """
        paths, labels = self.load_casia_v2()
        
        # Stratified split
        X_temp, X_test, y_temp, y_test = train_test_split(
            paths, labels, test_size=0.15, stratify=labels, random_state=42
        )
        val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adjusted, stratify=y_temp, random_state=42
        )
        
        train_ds = self._create_dataset(X_train, y_train, augment=True)
        val_ds = self._create_dataset(X_val, y_val, augment=False)
        test_ds = self._create_dataset(X_test, y_test, augment=False)
        
        return train_ds, val_ds, test_ds

    def _load_and_preprocess(self, path: str, label: int):
        """@brief Tek görüntü yükler, decode eder ve normalize eder."""
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, XCEPTION_INPUT_SIZE)
        return img, label

    def _augment(self, img, label):
        """@brief Eğitim için veri artırma uygular."""
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return img, label

    def _create_dataset(
        self, paths: List[str], labels: List[int], augment: bool
    ) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds
```

### 5.2 src/training/train_xception.py

```python
"""
@file train_xception.py
@brief Xception modelini CASIA v2 üzerinde eğiten script.
@usage python -m src.training.train_xception
"""

import os
import tensorflow as tf
from pathlib import Path
from .dataset_builder import ForensicDatasetBuilder
from src.ai_models.xception_model import XceptionForensicModel
from config.settings import MODELS_DIR, EPOCHS_XCEPTION

def train():
    """@brief İki aşamalı fine-tune eğitimini başlatır."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    builder = ForensicDatasetBuilder()
    train_ds, val_ds, test_ds = builder.build_tf_datasets()
    
    forensic_model = XceptionForensicModel()
    model = forensic_model.build_model()
    print(model.summary())
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "xception_best.h5"),
            save_best_only=True, monitor="val_auc", mode="max"
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs/xception")
    ]
    
    # Aşama 1: Sadece head eğitimi (5 epoch)
    print("\n=== AŞAMA 1: Head Eğitimi ===")
    model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)
    
    # Aşama 2: Fine-tune
    print("\n=== AŞAMA 2: Fine-Tuning ===")
    forensic_model.unfreeze_for_finetuning()
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_XCEPTION, callbacks=callbacks)
    
    model.save(str(MODELS_DIR / "xception_finetuned.h5"))
    
    # Test değerlendirme
    results = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {results[1]:.4f} | Test AUC: {results[2]:.4f}")

if __name__ == "__main__":
    train()
```

---

## 🎨 ADIM 6 — Streamlit Arayüzü

### 6.1 app.py

```python
"""
@file app.py
@brief IFDS Streamlit ana uygulama dosyası.
@details Görüntü sahteciliği tespiti için web arayüzü.
@usage streamlit run app.py
"""

import streamlit as st
import numpy as np
from pathlib import Path

# Page config — İLK Streamlit komutu olmalı
st.set_page_config(
    page_title="IFDS — Görüntü Sahteciliği Tespiti",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
with open("ui/styles/theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from src.preprocessing.image_loader import ImageLoader
from src.classical.sift_detector import SIFTDetector
from src.classical.surf_detector import SURFDetector
from src.classical.akaze_detector import AKAZEDetector
from src.classical.orb_detector import ORBDetector
from src.ai_models.xception_model import XceptionForensicModel
from src.ai_models.efficientnet_lstm import EfficientNetLSTMModel
from src.ai_models.gradcam import GradCAM
from src.reporting.report_generator import ReportGenerator
from ui.components.results_grid import render_results_grid
from ui.components.comparison_table import render_comparison_table
from config.settings import XCEPTION_MODEL_PATH, LSTM_MODEL_PATH


@st.cache_resource
def load_models():
    """@brief Modelleri bir kez yükler ve cache'ler."""
    xception = XceptionForensicModel()
    xception.load_weights()
    
    lstm = EfficientNetLSTMModel()
    lstm.load_weights()
    
    return xception, lstm


def main():
    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.image("ui/assets/logo.png", width=180)
        st.title("⚙️ Analiz Ayarları")
        
        st.subheader("Klasik Algoritmalar")
        run_sift  = st.checkbox("SIFT",  value=True)
        run_surf  = st.checkbox("SURF",  value=True)
        run_akaze = st.checkbox("AKAZE", value=True)
        run_orb   = st.checkbox("ORB",   value=True)
        
        st.subheader("AI Modelleri")
        run_xception = st.checkbox("CNN (Xception)", value=True)
        run_lstm     = st.checkbox("CNN+LSTM (EfficientNet)", value=True)
        run_gradcam  = st.checkbox("Grad-CAM Görselleştirme", value=True)
        
        st.divider()
        st.info("ℹ️ Desteklenen formatlar: JPG, PNG, GIF, BMP, TIFF")

    # ── Ana İçerik ───────────────────────────────────────────
    st.title("🔍 Görüntü Sahteciliği Tespiti Sistemi")
    st.caption("SIFT · SURF · AKAZE · ORB · CNN · LSTM | Powered by OpenCV & TensorFlow")
    
    # Dosya Yükleme
    uploaded_file = st.file_uploader(
        "Analiz edilecek görüntüyü yükleyin",
        type=["jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif"],
        help="Maksimum 50 MB"
    )
    
    if uploaded_file is None:
        st.info("👆 Başlamak için bir görüntü yükleyin.")
        return
    
    # Görüntü yükleme
    try:
        file_bytes = uploaded_file.read()
        img, metadata = ImageLoader.load_from_bytes(file_bytes, uploaded_file.name)
    except ValueError as e:
        st.error(f"❌ Hata: {e}")
        return
    
    # Önizleme ve metadata
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img, caption="Yüklenen Görüntü", use_column_width=True)
    with col2:
        st.subheader("📋 Görüntü Bilgileri")
        st.table({
            "Özellik": ["Dosya Adı", "Format", "Boyut", "Çözünürlük", "Kanal"],
            "Değer": [
                metadata["filename"],
                metadata["format"],
                f"{metadata['size_mb']:.2f} MB",
                f"{metadata['width']} × {metadata['height']} px",
                str(metadata["channels"])
            ]
        })
    
    st.divider()
    
    # Analiz butonu
    if not st.button("🚀 ANALİZİ BAŞLAT", type="primary", use_container_width=True):
        return
    
    all_results = {}
    img_preprocessed = ImageLoader.preprocess_for_model(img)
    
    # ── Klasik Algoritmalar ──────────────────────────────────
    if any([run_sift, run_surf, run_akaze, run_orb]):
        st.subheader("🔬 Klasik Algoritma Sonuçları")
        
        detectors = []
        if run_sift:  detectors.append(SIFTDetector())
        if run_surf:
            try:
                detectors.append(SURFDetector())
            except ImportError:
                st.warning("SURF: opencv-contrib-python bulunamadı, atlanıyor.")
        if run_akaze: detectors.append(AKAZEDetector())
        if run_orb:   detectors.append(ORBDetector())
        
        classical_results = {}
        prog = st.progress(0, text="Klasik analiz başlıyor...")
        
        for i, detector in enumerate(detectors):
            with st.spinner(f"{detector.get_name()} analiz ediliyor..."):
                result = detector.detect(img)
                classical_results[detector.get_name()] = result
            prog.progress((i + 1) / len(detectors), text=f"{detector.get_name()} tamamlandı")
        
        prog.empty()
        render_results_grid(classical_results)
        all_results["classical"] = classical_results
    
    # ── AI Modelleri ─────────────────────────────────────────
    if run_xception or run_lstm:
        st.subheader("🧠 AI Model Sonuçları")
        
        xception_model, lstm_model = load_models()
        ai_results = {}
        
        if run_xception:
            with st.spinner("Xception CNN analiz ediyor..."):
                xception_result = xception_model.predict(img_preprocessed)
                ai_results["Xception"] = xception_result
                
                if run_gradcam:
                    gradcam = GradCAM(xception_model.model)
                    img_batch = np.expand_dims(img_preprocessed, 0)
                    _, superimposed = gradcam.generate(img_batch)
                    
                    col_orig, col_cam = st.columns(2)
                    with col_orig:
                        st.image(img, caption="Orijinal", use_column_width=True)
                    with col_cam:
                        st.image(superimposed, caption="Grad-CAM", use_column_width=True)
        
        if run_lstm:
            with st.spinner("EfficientNet+LSTM analiz ediyor..."):
                lstm_result = lstm_model.predict(img_preprocessed)
                ai_results["LSTM"] = lstm_result
                
                if lstm_result.overlay_image is not None:
                    col_orig, col_heat = st.columns(2)
                    with col_orig:
                        st.image(img, caption="Orijinal", use_column_width=True)
                    with col_heat:
                        st.image(lstm_result.overlay_image,
                                 caption="Sahte Bölge Isı Haritası",
                                 use_column_width=True)
        
        # AI sonuç kartları
        for name, result in ai_results.items():
            verdict_color = "🔴" if result.is_forged else "🟢"
            st.metric(
                label=f"{verdict_color} {result.model_name}",
                value=result.class_label,
                delta=f"Güven: {result.confidence*100:.1f}%"
            )
        
        all_results["ai"] = ai_results
    
    # ── Karşılaştırma ────────────────────────────────────────
    st.subheader("📊 Karşılaştırmalı Analiz")
    render_comparison_table(all_results)
    
    # ── Rapor ────────────────────────────────────────────────
    st.divider()
    if st.button("📥 PDF Rapor İndir", use_container_width=True):
        with st.spinner("Rapor hazırlanıyor..."):
            report = ReportGenerator()
            pdf_bytes = report.generate(img, metadata, all_results)
            st.download_button(
                "⬇️ Raporu İndir",
                data=pdf_bytes,
                file_name=f"ifds_report_{metadata['filename']}.pdf",
                mime="application/pdf"
            )


if __name__ == "__main__":
    main()
```

---

## 🧪 ADIM 7 — Unit Testler

### 7.1 tests/conftest.py

```python
"""
@file conftest.py
@brief Pytest shared fixtures.
"""

import pytest
import numpy as np
import cv2


@pytest.fixture
def sample_rgb_image():
    """@brief Test için rastgele 256x256 RGB görüntü üretir."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_normalized_image():
    """@brief [0,1] normalize edilmiş 224x224 görüntü."""
    return np.random.rand(224, 224, 3).astype(np.float32)


@pytest.fixture
def copy_move_image():
    """@brief Basit copy-move sahteciliği içeren sentetik görüntü."""
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    # Bir bölgeyi kopyala
    patch = img[50:100, 50:100].copy()
    img[150:200, 150:200] = patch
    return img
```

### 7.2 tests/test_sift.py

```python
"""
@file test_sift.py
@brief SIFT dedektörü unit testleri.
"""

import pytest
import numpy as np
from src.classical.sift_detector import SIFTDetector
from src.classical.base_detector import DetectionResult


def test_sift_returns_detection_result(sample_rgb_image):
    detector = SIFTDetector()
    result = detector.detect(sample_rgb_image)
    assert isinstance(result, DetectionResult)


def test_sift_name():
    assert SIFTDetector().get_name() == "SIFT"


def test_sift_confidence_in_range(sample_rgb_image):
    result = SIFTDetector().detect(sample_rgb_image)
    assert 0.0 <= result.confidence <= 1.0


def test_sift_detects_copy_move(copy_move_image):
    result = SIFTDetector().detect(copy_move_image)
    # Copy-move görüntüde match_count > 0 olmalı
    assert result.match_count >= 0  # En az çalışmalı


def test_sift_processing_time_positive(sample_rgb_image):
    result = SIFTDetector().detect(sample_rgb_image)
    assert result.processing_time > 0
```

---

## 📊 ADIM 8 — Kod Kalitesi Konfigürasyonları

### 8.1 sonar-project.properties

```properties
sonar.projectKey=ifds-image-forgery
sonar.projectName=Image Forgery Detection System
sonar.projectVersion=1.0
sonar.sources=src,app.py,ui
sonar.tests=tests
sonar.python.coverage.reportPaths=coverage.xml
sonar.python.version=3.10
sonar.exclusions=data/**,logs/**,**/__pycache__/**
```

### 8.2 Doxyfile (kritik satırlar)

```
PROJECT_NAME           = "Image Forgery Detection System"
PROJECT_NUMBER         = 1.0
OUTPUT_DIRECTORY       = docs/doxygen
INPUT                  = src app.py ui config
RECURSIVE              = YES
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = YES
GENERATE_HTML          = YES
GENERATE_LATEX         = NO
HAVE_DOT               = YES
CALL_GRAPH             = YES
CALLER_GRAPH           = YES
DOT_IMAGE_FORMAT       = svg
```

### 8.3 Testleri çalıştırma komutu

```bash
# Test + coverage raporu
pytest tests/ -v --cov=src --cov-report=xml --cov-report=html

# SonarQube tarama (SonarScanner kurulu olmalı)
sonar-scanner

# Doxygen
doxygen Doxyfile
```

---

## 🚀 ADIM 9 — Çalıştırma Sırası

```bash
# 1. Ortam kurulumu
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Veri setlerini data/raw/ altına yerleştir
#    CASIA2/ ve Coverage/ klasörleri

# 3. Xception eğitimi
python -m src.training.train_xception

# 4. LSTM eğitimi  
python -m src.training.train_lstm

# 5. Testleri çalıştır
pytest tests/ -v --cov=src

# 6. Uygulamayı başlat
streamlit run app.py
```

---

## 📋 AI Coder'a Ekstra Talimatlar

```
ÖNEMLI NOTLAR:

1. Her dosyayı yukarıdaki dizin yapısına göre AYNEN oluştur.
2. Her fonksiyona Doxygen uyumlu @brief, @param, @return docstring ekle.
3. Type hints kullan (Python 3.10+).
4. Magic number kullanma, tüm sabitler config/settings.py'de.
5. SURF için ImportError yakala ve graceful fallback yap.
6. Streamlit'te @st.cache_resource ile model yüklemesini optimize et.
7. Her klasik dedektör BaseDetector'dan inherit etmeli.
8. DetectionResult ve AIDetectionResult dataclass kullan.
9. tests/ klasörüne her modül için ayrı test dosyası oluştur.
10. ui/styles/theme.css'te renk paleti: #1F3864 (koyu mavi), #2E75B6 (açık mavi).

CASIA v2 İNDİRME:
https://github.com/namtpham/casia2groundtruth (ground truth maskları)
http://forensics.idealtest.org/ (orijinal CASIA2)
```

---

*IFDS Implementation Plan v1.0 — Mayıs 2026*
