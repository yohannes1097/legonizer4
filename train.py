"""
Script utama untuk training model Legonizer4
"""

import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from src.models.efficient_net import EfficientNetFeatureExtractor
from src.models.trainer import LegoTrainer
from src.utils.data_loader import LegoDataLoader, LegoDataset
from src.utils.metrics import AccuracyReporter
from src.search.faiss_index import LegoSearchIndex

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Training model Legonizer4 untuk identifikasi LEGO brick"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path ke direktori data LEGO (berisi subdirektori per model ID)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/models",
        help="Direktori untuk menyimpan model hasil training"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size untuk training"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Jumlah epoch maksimum"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate untuk optimizer"
    )
    
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Proporsi data untuk validation (0.0-1.0)"
    )
    
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Jumlah epoch untuk early stopping"
    )
    
    # Model arguments
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Gunakan pre-trained weights"
    )
    
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate untuk regularisasi"
    )
    
    # Data augmentation
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Gunakan data augmentation"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device untuk training"
    )
    
    parser.add_argument(
        "--build_index",
        action="store_true",
        default=True,
        help="Build FAISS index setelah training"
    )
    
    return parser.parse_args()

def setup_device(device_arg: str) -> str:
    """
    Setup device untuk training
    
    Args:
        device_arg: Device argument dari command line
        
    Returns:
        Device string yang akan digunakan
    """
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    
    logger.info(f"Menggunakan device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def load_and_prepare_data(args) -> tuple:
    """
    Load dan prepare data untuk training
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple (train_loader, val_loader, class_to_idx, data_stats)
    """
    logger.info("Loading dan preparing data...")
    
    # Inisialisasi data loader
    data_loader = LegoDataLoader(
        data_dir=args.data_dir,
        val_split=args.val_split,
        seed=args.seed
    )
    
    # Split data
    train_paths, train_labels, val_paths, val_labels = data_loader.get_data_splits()
    
    # Buat datasets
    train_dataset = LegoDataset(
        train_paths, train_labels, augment=args.augment
    )
    val_dataset = LegoDataset(
        val_paths, val_labels, augment=False
    )
    
    # Buat data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Statistik data
    data_stats = {
        "num_classes": len(data_loader.class_to_idx),
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "class_distribution": data_loader.get_class_distribution()
    }
    
    logger.info(f"Data loaded: {data_stats['num_classes']} kelas, "
                f"{data_stats['train_samples']} training, "
                f"{data_stats['val_samples']} validation samples")
    
    return train_loader, val_loader, data_loader.class_to_idx, data_stats

def create_model(num_classes: int, args) -> EfficientNetFeatureExtractor:
    """
    Buat model untuk training
    
    Args:
        num_classes: Jumlah kelas
        args: Command line arguments
        
    Returns:
        Model instance
    """
    logger.info(f"Membuat model dengan {num_classes} kelas...")
    
    model = EfficientNetFeatureExtractor(
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate
    )
    
    return model

def train_model(model, train_loader, val_loader, class_to_idx, args) -> dict:
    """
    Training model
    
    Args:
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        class_to_idx: Class mapping
        args: Command line arguments
        
    Returns:
        Training results
    """
    logger.info("Memulai training...")
    
    # Setup device
    device = setup_device(args.device)
    
    # Inisialisasi trainer
    trainer = LegoTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Prepare data untuk trainer
    train_paths = [train_loader.dataset.image_paths[i] for i in range(len(train_loader.dataset))]
    train_labels = [train_loader.dataset.labels[i] for i in range(len(train_loader.dataset))]
    val_paths = [val_loader.dataset.image_paths[i] for i in range(len(val_loader.dataset))]
    val_labels = [val_loader.dataset.labels[i] for i in range(len(val_loader.dataset))]
    
    # Training
    start_time = time.time()
    results = trainer.train(
        train_data=(train_paths, train_labels),
        val_data=(val_paths, val_labels),
        model_dir=args.output_dir,
        class_to_idx=class_to_idx
    )
    training_time = time.time() - start_time
    
    results["training_time"] = training_time
    logger.info(f"Training selesai dalam {training_time:.2f} detik")
    
    return results

def build_search_index(model, train_loader, class_to_idx, output_dir: str):
    """
    Build FAISS search index dari model yang sudah ditraining
    
    Args:
        model: Trained model
        train_loader: Training data loader
        class_to_idx: Class mapping
        output_dir: Output directory
    """
    logger.info("Building FAISS search index...")
    
    # Setup trainer untuk feature extraction
    trainer = LegoTrainer(model)
    
    # Extract features dari training data
    train_paths = [train_loader.dataset.image_paths[i] for i in range(len(train_loader.dataset))]
    features = trainer.extract_features_batch(train_paths)
    
    # Prepare metadata
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    metadata = [
        {
            "model_id": idx_to_class[train_loader.dataset.labels[i]],
            "image_path": train_paths[i]
        }
        for i in range(len(train_paths))
    ]
    
    # Buat dan populate index
    search_index = LegoSearchIndex(dimension=features.shape[1])
    search_index.add_items(features, metadata)
    
    # Simpan index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    index_path = Path(output_dir) / f"search_index_{timestamp}.index"
    search_index.save_index(str(index_path))
    
    logger.info(f"FAISS index disimpan ke: {index_path}")

def log_training_results(results: dict, data_stats: dict, args):
    """
    Log hasil training ke accuracy reporter
    
    Args:
        results: Training results
        data_stats: Data statistics
        args: Command line arguments
    """
    logger.info("Logging training results...")
    
    # Inisialisasi reporter
    reporter = AccuracyReporter()
    
    # Log training accuracy
    reporter.log_training_accuracy(
        model_path=results["best_model_path"],
        train_accuracy=max(results["history"]["train_acc"]),
        val_accuracy=results["best_val_accuracy"],
        num_epochs=len(results["history"]["train_acc"]),
        num_classes=data_stats["num_classes"],
        training_time=results["training_time"],
        additional_metrics={
            "final_train_loss": results["history"]["train_loss"][-1],
            "final_val_loss": results["history"]["val_loss"][-1],
            "data_augmentation": args.augment,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate
        }
    )
    
    # Generate report
    report_path = reporter.generate_accuracy_report()
    logger.info(f"Training report disimpan ke: {report_path}")

def main():
    """
    Fungsi utama
    """
    # Parse arguments
    args = parse_arguments()
    
    logger.info("=" * 60)
    logger.info("LEGONIZER4 TRAINING")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Validation split: {args.val_split}")
    logger.info("=" * 60)
    
    try:
        # Load dan prepare data
        train_loader, val_loader, class_to_idx, data_stats = load_and_prepare_data(args)
        
        # Buat model
        model = create_model(data_stats["num_classes"], args)
        
        # Training
        results = train_model(model, train_loader, val_loader, class_to_idx, args)
        
        # Build search index jika diminta
        if args.build_index:
            build_search_index(model, train_loader, class_to_idx, args.output_dir)
        
        # Log results
        log_training_results(results, data_stats, args)
        
        logger.info("=" * 60)
        logger.info("TRAINING SELESAI!")
        logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
        logger.info(f"Model disimpan di: {results['best_model_path']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
