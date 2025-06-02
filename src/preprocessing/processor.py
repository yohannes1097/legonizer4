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
        
        # Hitung scaling factor dengan mempertahankan aspect ratio
        scale = min(target_w/w, target_h/h)
        
        # Hitung dimensi baru
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize gambar dengan interpolasi yang lebih baik
        resized = cv2.resize(
            image, 
            (new_w, new_h), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Buat canvas kosong dengan ukuran target
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        
        # Padding dengan warna putih dan blending di edges
        padded = cv2.copyMakeBorder(
            resized,
            top, bottom, left, right,
            cv2.BORDER_REPLICATE
        )
        
        # Smooth transition at borders
        if top > 0:
            padded[:top, :] = 255
        if bottom > 0:
            padded[-bottom:, :] = 255
        if left > 0:
            padded[:, :left] = 255
        if right > 0:
            padded[:, -right:] = 255
            
        return padded

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalisasi gambar untuk input ke neural network dengan color correction
        
        Args:
            image: Gambar input dalam format numpy array (BGR format dari OpenCV)
            
        Returns:
            Gambar yang sudah dinormalisasi dalam format RGB
        """
        # Convert BGR to RGB (OpenCV loads as BGR, but model expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Color correction in RGB space
        # Convert to LAB for better color normalization
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Normalize L channel
        l_norm = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_norm = clahe.apply(l_norm)
        
        # Merge channels and convert back to RGB
        corrected = cv2.merge([l_norm, a, b])
        corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)
        
        # Konversi ke float32 dan normalisasi ke range [0, 1]
        normalized = corrected.astype(np.float32) / 255.0
        
        # Standardisasi dengan mean dan std ImageNet (RGB format)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # RGB mean
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # RGB std
        
        normalized = (normalized - mean) / std
        return normalized.astype(np.float32)

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
        Melakukan augmentasi data pada gambar dengan teknik yang lebih advanced
        
        Args:
            image: Gambar input dalam format numpy array
            
        Returns:
            List gambar hasil augmentasi
        """
        augmented = []
        
        # Basic augmentations
        # Flip horizontal
        augmented.append(cv2.flip(image, 1))
        
        # Rotasi dengan border handling yang lebih baik
        for angle in [90, 180, 270]:
            # Get rotation matrix
            matrix = cv2.getRotationMatrix2D(
                (image.shape[1]/2, image.shape[0]/2),
                angle,
                1.0
            )
            
            # Calculate new bounds
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = int((image.shape[0] * sin) + (image.shape[1] * cos))
            new_h = int((image.shape[0] * cos) + (image.shape[1] * sin))
            
            # Adjust translation
            matrix[0, 2] += (new_w / 2) - image.shape[1]/2
            matrix[1, 2] += (new_h / 2) - image.shape[0]/2
            
            # Apply rotation with border handling
            rotated = cv2.warpAffine(
                image,
                matrix,
                (new_w, new_h),
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Resize back to original size
            rotated = cv2.resize(rotated, (image.shape[1], image.shape[0]))
            augmented.append(rotated)
        
        # Advanced color augmentations
        # Brightness
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
        augmented.extend([bright, dark])
        
        # Contrast
        contrast_high = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
        contrast_low = cv2.convertScaleAbs(image, alpha=0.7, beta=0)
        augmented.extend([contrast_high, contrast_low])
        
        # Color temperature
        # Warm
        warm = image.copy()
        warm[:,:,0] = cv2.convertScaleAbs(warm[:,:,0], alpha=0.9)  # Reduce blue
        warm[:,:,2] = cv2.convertScaleAbs(warm[:,:,2], alpha=1.1)  # Increase red
        augmented.append(warm)
        
        # Cool
        cool = image.copy()
        cool[:,:,0] = cv2.convertScaleAbs(cool[:,:,0], alpha=1.1)  # Increase blue
        cool[:,:,2] = cv2.convertScaleAbs(cool[:,:,2], alpha=0.9)  # Reduce red
        augmented.append(cool)
        
        # Add noise
        noise = image.copy()
        noise_mask = np.random.normal(0, 5, image.shape).astype(np.uint8)
        noise = cv2.add(noise, noise_mask)
        augmented.append(noise)
        
        return augmented
