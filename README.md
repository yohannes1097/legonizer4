# Legonizer4 🧱

Sistem identifikasi LEGO brick menggunakan deep learning dan computer vision.

## 🌟 Fitur Utama

- 🔍 Deteksi dan ekstraksi otomatis objek LEGO dari gambar
- 🧠 Model deep learning berbasis EfficientNetB4 untuk ekstraksi fitur
- 🔎 Similarity search menggunakan FAISS untuk pencarian cepat
- 📊 Sistem tracking dan reporting akurasi
- 🚀 REST API dengan FastAPI
- 🐳 Docker support dengan CUDA

## 📋 Persyaratan Sistem

- Python 3.8+
- CUDA 11.8+ (opsional, untuk GPU support)
- Docker (opsional, untuk containerization)

## 🚀 Quick Start

### Setup Project

```bash
# Clone repository
git clone https://github.com/yohannes1097/legonizer4.git
cd legonizer4

# Setup environment (Linux/Mac)
./setup.sh

# Atau untuk Windows
setup.bat
```

### Penggunaan Dasar

1. **Preprocessing Data**
```bash
python preprocess.py --input_dir data/raw --output_dir data/processed
```

2. **Training Model**
```bash
python train.py --data_dir data/processed --output_dir data/models
```

3. **Menjalankan API Server**
```bash
python -m src.api.main
```

4. **Menggunakan API Client**
```bash
# Identifikasi LEGO dari gambar
python examples/api_client.py identify path/to/image.jpg

# Dapatkan accuracy report
python examples/api_client.py accuracy
```

## 📁 Struktur Project

```
legonizer4/
├── data/                   # Data directory
│   ├── raw/               # Raw images
│   ├── processed/         # Preprocessed images
│   ├── models/           # Trained models
│   └── metrics/          # Accuracy reports
├── src/                   # Source code
│   ├── api/              # FastAPI server
│   ├── models/           # Model definitions
│   ├── preprocessing/    # Image preprocessing
│   ├── search/          # FAISS search
│   └── utils/           # Utilities
├── tests/                # Unit tests
├── examples/             # Example scripts
├── docker-compose.yml    # Docker compose config
└── Dockerfile           # Docker build file
```

## 🛠️ Komponen Utama

### 1. Preprocessing (`src/preprocessing/`)

- **detector.py**: Deteksi objek LEGO menggunakan computer vision
- **processor.py**: Preprocessing gambar untuk training

### 2. Model (`src/models/`)

- **efficient_net.py**: Model berbasis EfficientNetB4
- **trainer.py**: Training pipeline dengan early stopping

### 3. Search (`src/search/`)

- **faiss_index.py**: Implementasi similarity search dengan FAISS

### 4. API (`src/api/`)

- **main.py**: FastAPI server
- Endpoints:
  - `/identify`: Identifikasi LEGO dari gambar
  - `/train`: Training ulang model
  - `/accuracy`: Report akurasi

## 🔧 Konfigurasi

Konfigurasi dapat diatur melalui:

1. **File config.json**
2. **Environment variables**:
   - `LEGONIZER_API_HOST`
   - `LEGONIZER_API_PORT`
   - `LEGONIZER_DEVICE`
   - `LEGONIZER_BATCH_SIZE`
   - dll.

## 🐳 Docker

### Build dan Run dengan Docker

```bash
# Build image
docker-compose build

# Run services
docker-compose up -d

# Check logs
docker-compose logs -f
```

### GPU Support

Pastikan NVIDIA Container Toolkit terinstall untuk GPU support.

## 🧪 Testing

Menjalankan unit tests:

```bash
python run_tests.py
```

## 📊 Accuracy Reporting

Sistem menyediakan detailed accuracy reports:

- Training accuracy
- Validation accuracy
- Prediction accuracy per class
- Confusion matrix
- Visualisasi metrics

Reports dapat diakses melalui:
1. API endpoint `/accuracy`
2. HTML reports di `data/metrics/reports/`

## 🤝 Kontribusi

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 TODO

- [ ] Implementasi data augmentation tambahan
- [ ] Integrasi dengan MLflow untuk experiment tracking
- [ ] Support untuk model arsitektur lain
- [ ] Web interface untuk visualisasi
- [ ] Optimasi performa FAISS index

## 📄 Lisensi

Project ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 👥 Tim

- yohannes - [GitHub](https://github.com/yohannes1097)
- stefanus - [GitHub](https://github.com/fatbear2010)

## 🙏 Acknowledgments

- EfficientNet paper dan implementasi
- FAISS library dari Facebook Research
- FastAPI framework
- Komunitas open source
