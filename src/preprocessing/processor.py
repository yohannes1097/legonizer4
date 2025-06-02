"""
Module untuk preprocessing gambar LEGO sebelum training
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from .detector import LegoDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Kelas untuk melakukan preprocessing pada gambar LEGO
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 detector: Optional[LegoDetector] = None):
        """
        Inisialisasi processor
        
        Args:
            target_size: Ukuran output gambar (width, height)
            detector: Instance LegoDetector untuk deteksi objek
        """
        self.target_size = target_size
        self.detector = detector or LegoDetector()
        logger.info(f"Inisialisasi ImageProcessor dengan target_size={target_size}")

    def resize_and_pad(self, image: np.ndarray) -> np.ndarray:
        """
        Resize gambar dengan mempertahankan aspect ratio dan padding
        
        Args:
            image: Gambar input dalam format numpy array
            
        Returns:
            Gambar yang sudah diresize dan dipadding
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Hitung scaling factor
        scale = min(target_w/w, target_h/h)
        
        # Hitung dimensi baru
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize gambar
        resized = cv2.resize(image, (new_w, new_h))
        
        # Buat canvas kosong dengan ukuran target
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        
        # Padding dengan warna putih
        padded = cv2.copyMakeBorder(
            resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        
        return padded

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalisasi gambar untuk input ke neural network
        
        Args:
            image: Gambar input dalam format numpy array
            
        Returns:
            Gambar yang sudah dinormalisasi
        """
        # Konversi ke float32 dan normalisasi ke range [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Standardisasi dengan mean dan std ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        normalized = (normalized - mean) / std
        return normalized

    def process_single_image(self, 
                           image_path: str, 
                           output_path: Optional[str] = None,
                           save_visualization: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Proses satu gambar: deteksi, crop, resize, dan normalisasi
        
        Args:
            image_path: Path ke file gambar input
            output_path: Path untuk menyimpan hasil (optional)
            save_visualization: Flag untuk menyimpan visualisasi deteksi
            
        Returns:
            Tuple (processed_images, output_paths)
        """
        logger.info(f"Memproses gambar: {image_path}")
        
        # Deteksi dan ekstrak objek LEGO
        extracted_images = self.detector.detect_and_extract(image_path)
        
        processed_images = []
        output_paths = []
        
        for i, img in enumerate(extracted_images):
            # Resize dan padding
            resized = self.resize_and_pad(img)
            
            # Normalisasi
            normalized = self.normalize_image(resized)
            processed_images.append(normalized)
            
            if output_path:
                # Generate output path
                out_dir = Path(output_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                
                base_name = Path(image_path).stem
                out_path = out_dir / f"{base_name}_processed_{i}.jpg"
                
                # Simpan gambar yang sudah diproses
                cv2.imwrite(str(out_path), resized)
                output_paths.append(str(out_path))
                
                if save_visualization:
                    # Simpan visualisasi deteksi
                    vis_path = out_dir / f"{base_name}_detection.jpg"
                    self.detector.visualize_detection(
                        image_path,
                        str(vis_path)
                    )
        
        return processed_images, output_paths

    def process_directory(self,
                        input_dir: str,
                        output_dir: str,
                        save_visualization: bool = False,
                        max_workers: int = 4) -> dict:
        """
        Proses semua gambar dalam direktori secara parallel
        
        Args:
            input_dir: Path ke direktori input
            output_dir: Path ke direktori output
            save_visualization: Flag untuk menyimpan visualisasi deteksi
            max_workers: Jumlah worker thread maksimum
            
        Returns:
            Dictionary statistik hasil processing
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Directory tidak ditemukan: {input_dir}")
        
        # Dapatkan semua file gambar
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(input_path.glob(f"**/*{ext}"))
        
        logger.info(f"Ditemukan {len(image_files)} file gambar di {input_dir}")
        
        stats = {
            'total_images': len(image_files),
            'processed_images': 0,
            'failed_images': 0,
            'output_files': []
        }
        
        # Proses gambar secara parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for img_path in image_files:
                # Preserve struktur subdirektori
                rel_path = img_path.relative_to(input_path)
                out_subdir = Path(output_dir) / rel_path.parent
                
                future = executor.submit(
                    self.process_single_image,
                    str(img_path),
                    str(out_subdir),
                    save_visualization
                )
                futures.append(future)
            
            # Collect hasil
            for future in futures:
                try:
                    _, output_paths = future.result()
                    stats['processed_images'] += 1
                    stats['output_files'].extend(output_paths)
                except Exception as e:
                    logger.error(f"Error saat memproses gambar: {e}")
                    stats['failed_images'] += 1
        
        logger.info(
            f"Selesai memproses {stats['processed_images']} gambar "
            f"({stats['failed_images']} gagal)"
        )
        return stats

    def augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Melakukan augmentasi data pada gambar
        
        Args:
            image: Gambar input dalam format numpy array
            
        Returns:
            List gambar hasil augmentasi
        """
        augmented = []
        
        # Flip horizontal
        augmented.append(cv2.flip(image, 1))
        
        # Rotasi
        for angle in [90, 180, 270]:
            matrix = cv2.getRotationMatrix2D(
                (image.shape[1]/2, image.shape[0]/2),
                angle,
                1.0
            )
            rotated = cv2.warpAffine(
                image,
                matrix,
                (image.shape[1], image.shape[0])
            )
            augmented.append(rotated)
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
        augmented.extend([bright, dark])
        
        return augmented
