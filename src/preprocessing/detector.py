"""
Module untuk deteksi objek LEGO dari gambar menggunakan teknik computer vision
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegoDetector:
    """
    Kelas untuk mendeteksi dan mengekstrak objek LEGO dari gambar
    """
    
    def __init__(self, min_contour_area: int = 1000, blur_kernel: int = 5):
        """
        Inisialisasi detector dengan parameter default
        
        Args:
            min_contour_area: Area minimum kontur yang dianggap sebagai objek LEGO
            blur_kernel: Ukuran kernel untuk Gaussian blur
        """
        self.min_contour_area = min_contour_area
        self.blur_kernel = blur_kernel
        logger.info(f"Inisialisasi LegoDetector dengan min_contour_area={min_contour_area}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing gambar untuk memudahkan deteksi
        
        Args:
            image: Gambar input dalam format numpy array
            
        Returns:
            Gambar yang sudah dipreprocess
        """
        # Convert ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplikasikan Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Threshold untuk mendapatkan binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh

    def find_lego_contours(self, preprocessed_img: np.ndarray) -> list:
        """
        Mencari kontur objek LEGO dari gambar yang sudah dipreprocess
        
        Args:
            preprocessed_img: Gambar yang sudah dipreprocess
            
        Returns:
            List kontur yang terdeteksi sebagai objek LEGO
        """
        # Cari semua kontur
        contours, _ = cv2.findContours(
            preprocessed_img, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter kontur berdasarkan area minimum
        lego_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > self.min_contour_area
        ]
        
        logger.info(f"Ditemukan {len(lego_contours)} objek LEGO potensial")
        return lego_contours

    def extract_lego_object(self, 
                          image: np.ndarray, 
                          contour: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Mengekstrak objek LEGO dari kontur yang ditemukan
        
        Args:
            image: Gambar original
            contour: Kontur objek LEGO
            
        Returns:
            Tuple berisi (cropped_image, bounding_box)
        """
        # Dapatkan bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Tambahkan padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Crop gambar
        cropped = image[y1:y2, x1:x2]
        
        return cropped, (x1, y1, x2, y2)

    def detect_and_extract(self, 
                          image_path: str, 
                          output_path: Optional[str] = None) -> list:
        """
        Mendeteksi dan mengekstrak semua objek LEGO dari gambar
        
        Args:
            image_path: Path ke file gambar input
            output_path: Path untuk menyimpan hasil ekstraksi (optional)
            
        Returns:
            List path file hasil ekstraksi jika output_path diberikan,
            atau list gambar hasil ekstraksi jika output_path None
        """
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
        
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Cari kontur
        contours = self.find_lego_contours(preprocessed)
        
        results = []
        for i, contour in enumerate(contours):
            # Ekstrak objek
            cropped, bbox = self.extract_lego_object(image, contour)
            
            if output_path:
                # Buat output directory jika belum ada
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate nama file output
                base_name = Path(image_path).stem
                out_path = output_dir / f"{base_name}_object_{i}.jpg"
                
                # Simpan gambar
                cv2.imwrite(str(out_path), cropped)
                results.append(str(out_path))
                logger.info(f"Objek LEGO disimpan ke: {out_path}")
            else:
                results.append(cropped)
        
        return results

    def visualize_detection(self, 
                          image_path: str, 
                          output_path: Optional[str] = None) -> np.ndarray:
        """
        Memvisualisasikan hasil deteksi dengan bounding box
        
        Args:
            image_path: Path ke file gambar input
            output_path: Path untuk menyimpan hasil visualisasi (optional)
            
        Returns:
            Gambar dengan visualisasi deteksi
        """
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
        
        # Copy gambar untuk visualisasi
        vis_image = image.copy()
        
        # Preprocess dan deteksi
        preprocessed = self.preprocess_image(image)
        contours = self.find_lego_contours(preprocessed)
        
        # Gambar bounding box
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                vis_image, 
                (x, y), 
                (x + w, y + h), 
                (0, 255, 0), 
                2
            )
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Visualisasi deteksi disimpan ke: {output_path}")
        
        return vis_image
