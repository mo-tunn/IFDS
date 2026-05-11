# IFDS - Image Forgery Detection System

IFDS, görüntü manipülasyonu tespiti için geliştirilen Streamlit tabanlı bir analiz uygulamasıdır. Sistem; klasik OpenCV tabanlı özellik eşleştirme algoritmalarını ve isteğe bağlı derin öğrenme modellerini birlikte çalıştırarak yüklenen görsel için karşılaştırmalı sonuç, nihai karar ve indirilebilir rapor üretir.

## Özellikler

- SIFT, SURF, AKAZE ve ORB ile klasik kopyala-yapıştır / sahtecilik sinyali analizi
- Xception CNN ve EfficientNet CNN ile isteğe bağlı AI tabanlı sınıflandırma
- Xception modeli için Grad-CAM ısı haritası desteği
- Ağırlıklı karar mekanizması ile `Authentic`, `Tampered` veya `Review needed` sonucu
- Görsel metadatası, algoritma karşılaştırması ve PDF/HTML rapor çıktısı
- GIF, JPG/JPEG, PNG, BMP ve TIFF format desteği

## Proje Yapısı

```text
.
├── app.py                  # Streamlit uygulama giriş noktası
├── config/                 # Uygulama ayarları ve model yolları
├── data/
│   ├── raw/                # Yerel ham veri setleri (git'e eklenmez)
│   ├── processed/          # İşlenmiş veri çıktıları (git'e eklenmez)
│   └── models/             # Yerel model ağırlıkları (git'e eklenmez)
├── docs/                   # Dokümantasyon ve model metrikleri
├── notebooks/              # Eğitim notebook'ları
├── src/
│   ├── ai_models/          # Xception, EfficientNet ve Grad-CAM bileşenleri
│   ├── classical/          # SIFT, SURF, AKAZE, ORB dedektörleri
│   ├── preprocessing/      # Görüntü yükleme ve ön işleme
│   ├── reporting/          # PDF/HTML rapor üretimi
│   └── verdict.py          # Nihai karar hesaplama servisi
├── tests/                  # Pytest testleri
└── ui/                     # Streamlit arayüz bileşenleri
```

## Kurulum

Python 3.10 veya üzeri bir sürüm önerilir.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux kullanıyorsanız sanal ortam aktivasyonu:

```bash
source .venv/bin/activate
```

## Uygulamayı Çalıştırma

```bash
streamlit run app.py
```

Uygulama açıldıktan sonra bir görüntü yükleyip çalıştırılacak algoritmaları sol menüden seçebilirsiniz.

## Model Dosyaları

AI modelleri opsiyoneldir. Model ağırlıkları yoksa uygulama klasik algoritmalar ve raporlama akışıyla çalışmaya devam eder.

AI analizi için aşağıdaki dosyaları `data/models/` klasörüne yerleştirin:

```text
data/models/xception_finetuned.h5
data/models/efficientnet_finetuned.h5
```

Model ağırlıkları ve veri setleri büyük dosyalar olduğu için `.gitignore` kapsamındadır. Eğitim metrikleri için `docs/model_metrics.md` ve `data/models/training_results.json` dosyalarına bakabilirsiniz.

## Test

Geliştirme bağımlılıklarını kurup testleri çalıştırmak için:

```bash
pip install -r requirements-dev.txt
pytest tests -v --cov=src --cov-report=xml
```

## Eğitim Notları

`notebooks/` klasöründe Kaggle/CASIA veri setiyle model eğitimi için hazırlanmış notebook'lar bulunur. Yerel veri setlerinizi `data/raw/` altında tutabilirsiniz; bu klasör GitHub'a gönderilmez.

## GitHub'a Göndermeden Önce

- `.env`, Streamlit secrets, sanal ortamlar, veri setleri ve model ağırlıkları repoya eklenmez.
- `data/raw/`, `data/processed/` ve `data/models/` klasörleri `.gitkeep` ile korunur.
- Büyük model dosyalarını paylaşmanız gerekiyorsa Git LFS veya harici bir model depolama alanı kullanmanız önerilir.

## Lisans

Bu projeyi yayınlamadan önce uygun lisans dosyasını (`LICENSE`) eklemeniz önerilir.
