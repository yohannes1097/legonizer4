"""
Contoh penggunaan Legonizer4 untuk identifikasi LEGO brick
"""

import asyncio
from pathlib import Path
from src.preprocessing.detector import LegoDetector
from src.preprocessing.processor import ImageProcessor
from src.models.efficient_net import EfficientNetFeatureExtractor
from src.models.trainer import LegoTrainer
from src.search.faiss_index import LegoSearchIndex
from src.utils.metrics import AccuracyReporter

async def contoh_preprocessing():
    """
    Contoh preprocessing gambar LEGO
    """
    print("=== Contoh Preprocessing ===")
    
    # Inisialisasi detector dan processor
    detector = LegoDetector()
    processor = ImageProcessor()
    
    # Contoh deteksi objek LEGO dari gambar
    image_path = "data/raw/example_lego.jpg"  # Ganti dengan path gambar Anda
    
    if Path(image_path).exists():
        # Deteksi dan ekstrak objek LEGO
        extracted_objects = detector.detect_and_extract(
            image_path,
            output_path="data/processed/extracted"
        )
        print(f"Berhasil mengekstrak {len(extracted_objects)} objek LEGO")
        
        # Preprocessing untuk training
        processed_images, output_paths = processor.process_single_image(
            image_path,
            output_path="data/processed/training_ready",
            save_visualization=True
        )
        print(f"Berhasil memproses {len(processed_images)} gambar untuk training")
    else:
        print(f"File gambar tidak ditemukan: {image_path}")

async def contoh_training():
    """
    Contoh training model
    """
    print("\n=== Contoh Training ===")
    
    # Inisialisasi model
    num_classes = 10  # Sesuaikan dengan jumlah model LEGO Anda
    model = EfficientNetFeatureExtractor(num_classes=num_classes)
    
    # Inisialisasi trainer
    trainer = LegoTrainer(model, num_epochs=5)  # Epoch kecil untuk contoh
    
    # Contoh data (dalam implementasi nyata, load dari direktori data)
    train_data = ([], [])  # (image_paths, labels)
    val_data = ([], [])    # (image_paths, labels)
    class_to_idx = {}      # mapping class name ke index
    
    print("Training model... (implementasi lengkap membutuhkan data)")
    # results = trainer.train(train_data, val_data, "data/models", class_to_idx)

async def contoh_similarity_search():
    """
    Contoh similarity search dengan FAISS
    """
    print("\n=== Contoh Similarity Search ===")
    
    # Inisialisasi FAISS index
    dimension = 1792  # Dimensi feature dari EfficientNetB4
    search_index = LegoSearchIndex(dimension=dimension)
    
    # Contoh menambahkan item ke index
    import numpy as np
    
    # Simulasi feature vectors
    features = np.random.random((5, dimension)).astype('float32')
    metadata = [
        {"model_id": "31313", "name": "Bugatti Chiron"},
        {"model_id": "42115", "name": "Liebherr Excavator"},
        {"model_id": "10264", "name": "Corner Garage"},
        {"model_id": "21318", "name": "Tree House"},
        {"model_id": "75192", "name": "Millennium Falcon"}
    ]
    
    search_index.add_items(features, metadata)
    print(f"Menambahkan {len(features)} item ke index")
    
    # Contoh pencarian
    query_vector = np.random.random((1, dimension)).astype('float32')
    results = search_index.search(query_vector, k=3)
    
    print("Hasil pencarian similarity:")
    for i, (meta, distance) in enumerate(results):
        similarity = 1.0 / (1.0 + distance)
        print(f"  {i+1}. {meta['name']} (ID: {meta['model_id']}) - Similarity: {similarity:.4f}")

async def contoh_accuracy_reporting():
    """
    Contoh accuracy reporting
    """
    print("\n=== Contoh Accuracy Reporting ===")
    
    # Inisialisasi reporter
    reporter = AccuracyReporter()
    
    # Log contoh training accuracy
    reporter.log_training_accuracy(
        model_path="data/models/model_example.pth",
        train_accuracy=0.95,
        val_accuracy=0.87,
        num_epochs=50,
        num_classes=10,
        training_time=3600.0
    )
    
    # Log contoh prediction accuracy
    reporter.log_prediction_accuracy(
        predicted_class="31313",
        actual_class="31313",
        confidence=0.92,
        similarity_score=0.89
    )
    
    reporter.log_prediction_accuracy(
        predicted_class="42115",
        actual_class="31313",
        confidence=0.78,
        similarity_score=0.65
    )
    
    # Dapatkan statistik
    training_stats = reporter.get_training_stats()
    prediction_stats = reporter.get_prediction_stats()
    
    print("Training Statistics:")
    for key, value in training_stats.items():
        print(f"  {key}: {value}")
    
    print("\nPrediction Statistics:")
    for key, value in prediction_stats.items():
        if key != "class_statistics":
            print(f"  {key}: {value}")
    
    # Generate report
    report_path = reporter.generate_accuracy_report()
    print(f"\nReport HTML disimpan ke: {report_path}")

async def main():
    """
    Fungsi utama untuk menjalankan semua contoh
    """
    print("Legonizer4 - Contoh Penggunaan")
    print("=" * 50)
    
    await contoh_preprocessing()
    await contoh_training()
    await contoh_similarity_search()
    await contoh_accuracy_reporting()
    
    print("\n" + "=" * 50)
    print("Contoh selesai! Lihat dokumentasi untuk penggunaan lengkap.")

if __name__ == "__main__":
    asyncio.run(main())
