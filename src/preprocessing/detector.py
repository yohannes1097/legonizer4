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

    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Menghilangkan bayangan dari gambar menggunakan morphological operations
        
        Args:
            image: Gambar input BGR
            
        Returns:
            Gambar tanpa bayangan
        """
        rgb_planes = cv2.split(image)
        result_planes = []
        
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            result_planes.append(diff_img)
            
        return cv2.merge(result_planes)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Meningkatkan contrast gambar menggunakan CLAHE
        
        Args:
            image: Gambar input BGR
            
        Returns:
            Gambar dengan contrast yang ditingkatkan
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def color_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Melakukan segmentasi berdasarkan warna menggunakan HSV
        
        Args:
            image: Gambar input BGR
            
        Returns:
            Mask hasil segmentasi warna
        """
        # Convert ke HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range warna LEGO dengan toleransi lebih tinggi
        color_ranges = [
            # Merah (2 ranges untuk merah)
            (np.array([0, 30, 30]), np.array([15, 255, 255])),
            (np.array([160, 30, 30]), np.array([180, 255, 255])),
            # Biru
            (np.array([85, 30, 30]), np.array([135, 255, 255])),
            # Hijau
            (np.array([30, 30, 30]), np.array([90, 255, 255])),
            # Kuning
            (np.array([15, 30, 30]), np.array([40, 255, 255])),
            # Orange
            (np.array([5, 30, 30]), np.array([20, 255, 255])),
            # Putih/Light Gray
            (np.array([0, 0, 150]), np.array([180, 50, 255])),
            # Hitam/Dark Gray
            (np.array([0, 0, 0]), np.array([180, 255, 80])),
            # Abu-abu Medium
            (np.array([0, 0, 40]), np.array([180, 50, 220]))
        ]
        
        # Combine masks
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations untuk membersihkan noise
        kernel = np.ones((7,7), np.uint8)  # Increased kernel size
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)  # Close first to connect components
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)   # Then open to remove noise
        
        return combined_mask

    def validate_shape(self, contour: np.ndarray) -> bool:
        """
        Memvalidasi bentuk kontur apakah sesuai dengan karakteristik LEGO
        dengan toleransi yang lebih longgar
        
        Args:
            contour: Kontur yang akan divalidasi
            
        Returns:
            Boolean indicating if shape is valid
        """
        # Hitung properti bentuk
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
            
        # Hitung circularity dengan toleransi lebih longgar
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Hitung rectangularity
        rect_area = cv2.minAreaRect(contour)
        rect_area = rect_area[1][0] * rect_area[1][1]
        if rect_area == 0:
            return False
        rectangularity = area / rect_area
        
        # Hitung aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h if h != 0 else 0
        
        # Validasi metrics dengan toleransi yang lebih tinggi dan handling rotasi
        return (0.2 < circularity < 1.0 and  # Even more permissive circularity
                0.5 < rectangularity < 1.0 and  # More permissive rectangularity
                0.15 < aspect_ratio < 6.0)  # Wider aspect ratio range for rotated pieces

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing gambar untuk memudahkan deteksi
        
        Args:
            image: Gambar input dalam format numpy array
            
        Returns:
            Gambar yang sudah dipreprocess
        """
        # Remove shadows
        no_shadow = self.remove_shadows(image)
        
        # Enhance contrast
        enhanced = self.enhance_contrast(no_shadow)
        
        # Color segmentation (primary detection method)
        color_mask = self.color_segmentation(enhanced)
        
        # Convert ke grayscale untuk edge detection
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Edge detection dengan threshold yang lebih rendah
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate edges untuk menghubungkan garis yang terputus
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine color mask dan edges
        # Gunakan color mask sebagai primary mask, edges sebagai support
        combined = cv2.bitwise_and(color_mask, color_mask, mask=edges)
        
        # Final cleanup dengan morphological operations
        kernel = np.ones((5,5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return combined

    def merge_nearby_contours(self, contours: list, merge_distance: int = 50) -> list:
        """
        Menggabungkan kontur yang berdekatan untuk mengatasi deteksi terpisah pada objek yang sama
        
        Args:
            contours: List kontur yang akan digabungkan
            merge_distance: Jarak maksimum untuk menggabungkan kontur
            
        Returns:
            List kontur yang sudah digabungkan
        """
        if len(contours) <= 1:
            return contours
        
        merged_contours = []
        used_indices = set()
        
        for i, cnt1 in enumerate(contours):
            if i in used_indices:
                continue
                
            # Dapatkan bounding box untuk kontur pertama
            x1, y1, w1, h1 = cv2.boundingRect(cnt1)
            center1 = (x1 + w1//2, y1 + h1//2)
            
            # Cari kontur yang berdekatan
            group = [cnt1]
            used_indices.add(i)
            
            for j, cnt2 in enumerate(contours):
                if j in used_indices:
                    continue
                    
                x2, y2, w2, h2 = cv2.boundingRect(cnt2)
                center2 = (x2 + w2//2, y2 + h2//2)
                
                # Hitung jarak antar center
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < merge_distance:
                    group.append(cnt2)
                    used_indices.add(j)
            
            # Jika ada kontur yang digabungkan, buat kontur baru
            if len(group) > 1:
                # Gabungkan semua points
                all_points = np.vstack(group)
                # Buat convex hull dari semua points
                merged_contour = cv2.convexHull(all_points)
                merged_contours.append(merged_contour)
            else:
                merged_contours.append(cnt1)
        
        return merged_contours

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
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_contour_area:
                filtered_contours.append(cnt)
        
        # Gabungkan kontur yang berdekatan
        merged_contours = self.merge_nearby_contours(filtered_contours)
        
        # Validasi bentuk setelah penggabungan
        lego_contours = []
        for cnt in merged_contours:
            if self.validate_shape(cnt):
                lego_contours.append(cnt)
        
        logger.info(f"Ditemukan {len(lego_contours)} objek LEGO tervalidasi")
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
