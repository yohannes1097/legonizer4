"""
Script untuk preprocessing data LEGO sebelum training
"""

import argparse
import logging
from pathlib import Path
import shutil
from typing import Dict, List
import json

from src.preprocessing.detector import LegoDetector
from src.preprocessing.processor import ImageProcessor

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
        description="Preprocessing data LEGO untuk training"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path ke direktori input berisi foto LEGO raw"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Path ke direktori output untuk hasil preprocessing"
    )
    
    parser.add_argument(
        "--min_contour_area",
        type=int,
        default=1000,
        help="Area minimum kontur untuk deteksi objek LEGO"
    )
    
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Ukuran target gambar (width height)"
    )
    
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="Simpan visualisasi hasil deteksi"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Jumlah worker thread untuk processing parallel"
    )
    
    parser.add_argument(
        "--copy_structure",
        action="store_true",
        default=True,
        help="Copy struktur direktori dari input ke output"
    )
    
    return parser.parse_args()

def validate_input_structure(input_dir: Path) -> Dict[str, List[str]]:
    """
    Validasi struktur direktori input dan dapatkan informasi dataset
    
    Args:
        input_dir: Path ke direktori input
        
    Returns:
        Dictionary berisi informasi struktur dataset
    """
    if not input_dir.exists():
        raise ValueError(f"Directory input tidak ditemukan: {input_dir}")
    
    # Dapatkan semua subdirektori (kelas)
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"Tidak ada subdirektori ditemukan di {input_dir}")
    
    # Validasi setiap kelas dan hitung file gambar
    dataset_info = {}
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = []
        
        for ext in valid_extensions:
            image_files.extend(list(class_dir.glob(f"*{ext}")))
            image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        if image_files:
            dataset_info[class_name] = [str(f) for f in image_files]
        else:
            logger.warning(f"Tidak ada file gambar ditemukan di {class_dir}")
    
    if not dataset_info:
        raise ValueError("Tidak ada file gambar valid ditemukan dalam dataset")
    
    return dataset_info

def setup_output_structure(output_dir: Path, dataset_info: Dict[str, List[str]]):
    """
    Setup struktur direktori output
    
    Args:
        output_dir: Path ke direktori output
        dataset_info: Informasi struktur dataset
    """
    # Buat direktori utama
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Buat subdirektori untuk setiap kelas
    for class_name in dataset_info.keys():
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(exist_ok=True)
    
    # Buat direktori untuk visualisasi
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

def process_dataset(dataset_info: Dict[str, List[str]], 
                   output_dir: Path,
                   args) -> Dict:
    """
    Process seluruh dataset
    
    Args:
        dataset_info: Informasi struktur dataset
        output_dir: Path ke direktori output
        args: Command line arguments
        
    Returns:
        Dictionary berisi statistik processing
    """
    # Inisialisasi detector dan processor
    detector = LegoDetector(min_contour_area=args.min_contour_area)
    processor = ImageProcessor(target_size=tuple(args.target_size), detector=detector)
    
    # Statistik processing
    stats = {
        "total_classes": len(dataset_info),
        "total_input_images": sum(len(files) for files in dataset_info.values()),
        "processed_images": 0,
        "failed_images": 0,
        "output_images": 0,
        "class_stats": {}
    }
    
    logger.info(f"Memulai preprocessing {stats['total_input_images']} gambar "
                f"dari {stats['total_classes']} kelas...")
    
    # Process setiap kelas
    for class_name, image_files in dataset_info.items():
        logger.info(f"Processing kelas: {class_name} ({len(image_files)} gambar)")
        
        class_output_dir = output_dir / class_name
        class_stats = {
            "input_images": len(image_files),
            "processed_images": 0,
            "failed_images": 0,
            "output_images": 0
        }
        
        # Process setiap gambar dalam kelas
        for img_path in image_files:
            try:
                # Process single image
                processed_images, output_paths = processor.process_single_image(
                    img_path,
                    str(class_output_dir),
                    args.save_visualization
                )
                
                class_stats["processed_images"] += 1
                class_stats["output_images"] += len(output_paths)
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                class_stats["failed_images"] += 1
        
        # Update global stats
        stats["processed_images"] += class_stats["processed_images"]
        stats["failed_images"] += class_stats["failed_images"]
        stats["output_images"] += class_stats["output_images"]
        stats["class_stats"][class_name] = class_stats
        
        logger.info(
            f"Kelas {class_name} selesai: "
            f"{class_stats['processed_images']} berhasil, "
            f"{class_stats['failed_images']} gagal, "
            f"{class_stats['output_images']} output images"
        )
    
    return stats

def save_processing_report(stats: Dict, output_dir: Path, args):
    """
    Simpan laporan hasil preprocessing
    
    Args:
        stats: Statistik processing
        output_dir: Path ke direktori output
        args: Command line arguments
    """
    # Buat laporan
    report = {
        "preprocessing_config": {
            "min_contour_area": args.min_contour_area,
            "target_size": args.target_size,
            "save_visualization": args.save_visualization,
            "max_workers": args.max_workers
        },
        "statistics": stats,
        "success_rate": stats["processed_images"] / stats["total_input_images"] if stats["total_input_images"] > 0 else 0
    }
    
    # Simpan sebagai JSON
    report_path = output_dir / "preprocessing_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Laporan preprocessing disimpan ke: {report_path}")

def main():
    """
    Fungsi utama
    """
    # Parse arguments
    args = parse_arguments()
    
    logger.info("=" * 60)
    logger.info("LEGONIZER4 PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target size: {args.target_size}")
    logger.info(f"Min contour area: {args.min_contour_area}")
    logger.info(f"Save visualization: {args.save_visualization}")
    logger.info("=" * 60)
    
    try:
        # Validasi input
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        dataset_info = validate_input_structure(input_dir)
        logger.info(f"Dataset ditemukan: {len(dataset_info)} kelas")
        
        # Print dataset summary
        for class_name, files in dataset_info.items():
            logger.info(f"  {class_name}: {len(files)} gambar")
        
        # Setup output structure
        setup_output_structure(output_dir, dataset_info)
        
        # Process dataset
        stats = process_dataset(dataset_info, output_dir, args)
        
        # Save report
        save_processing_report(stats, output_dir, args)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PREPROCESSING SELESAI!")
        logger.info(f"Total input: {stats['total_input_images']} gambar")
        logger.info(f"Berhasil diproses: {stats['processed_images']} gambar")
        logger.info(f"Gagal diproses: {stats['failed_images']} gambar")
        logger.info(f"Total output: {stats['output_images']} gambar")
        logger.info(f"Success rate: {stats['processed_images']/stats['total_input_images']*100:.1f}%")
        logger.info(f"Output disimpan di: {output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()
