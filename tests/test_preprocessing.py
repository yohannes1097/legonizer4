"""
Unit tests untuk preprocessing module
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from src.preprocessing.detector import LegoDetector
from src.preprocessing.processor import ImageProcessor

class TestLegoDetector(unittest.TestCase):
    """
    Test cases untuk LegoDetector
    """
    
    def setUp(self):
        """
        Setup test environment
        """
        self.detector = LegoDetector()
        self.temp_dir = tempfile.mkdtemp()
        
        # Buat test image
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        # Tambahkan objek persegi putih (simulasi LEGO)
        cv2.rectangle(self.test_image, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(self.test_image, (200, 200), (280, 280), (255, 255, 255), -1)
        
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        cv2.imwrite(str(self.test_image_path), self.test_image)
    
    def tearDown(self):
        """
        Cleanup test environment
        """
        shutil.rmtree(self.temp_dir)
    
    def test_detect_objects(self):
        """
        Test deteksi objek LEGO
        """
        contours = self.detector.detect_objects(self.test_image)
        
        # Harus mendeteksi setidaknya 1 objek
        self.assertGreater(len(contours), 0)
        
        # Contour harus berupa list numpy arrays
        for contour in contours:
            self.assertIsInstance(contour, np.ndarray)
    
    def test_extract_objects(self):
        """
        Test ekstraksi objek dari gambar
        """
        contours = self.detector.detect_objects(self.test_image)
        extracted = self.detector.extract_objects(self.test_image, contours)
        
        # Harus mengekstrak objek
        self.assertGreater(len(extracted), 0)
        
        # Setiap objek harus berupa numpy array
        for obj in extracted:
            self.assertIsInstance(obj, np.ndarray)
            self.assertEqual(len(obj.shape), 3)  # Height, Width, Channels
    
    def test_detect_and_extract(self):
        """
        Test deteksi dan ekstraksi sekaligus
        """
        output_dir = Path(self.temp_dir) / "extracted"
        extracted = self.detector.detect_and_extract(
            str(self.test_image_path),
            str(output_dir)
        )
        
        # Harus mengekstrak objek
        self.assertGreater(len(extracted), 0)
        
        # Output directory harus dibuat
        self.assertTrue(output_dir.exists())

class TestImageProcessor(unittest.TestCase):
    """
    Test cases untuk ImageProcessor
    """
    
    def setUp(self):
        """
        Setup test environment
        """
        self.processor = ImageProcessor()
        self.temp_dir = tempfile.mkdtemp()
        
        # Buat test image dengan PIL
        test_image = Image.new('RGB', (300, 300), color='white')
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        test_image.save(self.test_image_path)
    
    def tearDown(self):
        """
        Cleanup test environment
        """
        shutil.rmtree(self.temp_dir)
    
    def test_resize_image(self):
        """
        Test resize gambar
        """
        # Load test image
        image = cv2.imread(str(self.test_image_path))
        
        # Resize
        resized = self.processor.resize_image(image, (224, 224))
        
        # Check dimensi
        self.assertEqual(resized.shape[:2], (224, 224))
    
    def test_normalize_image(self):
        """
        Test normalisasi gambar
        """
        # Buat test image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Normalize
        normalized = self.processor.normalize_image(image)
        
        # Check range [0, 1]
        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)
        
        # Check dtype
        self.assertEqual(normalized.dtype, np.float32)
    
    def test_process_single_image(self):
        """
        Test processing single image
        """
        output_dir = Path(self.temp_dir) / "processed"
        
        processed_images, output_paths = self.processor.process_single_image(
            str(self.test_image_path),
            str(output_dir)
        )
        
        # Harus ada hasil processing
        self.assertGreater(len(processed_images), 0)
        self.assertEqual(len(processed_images), len(output_paths))
        
        # Check output directory
        self.assertTrue(output_dir.exists())

if __name__ == '__main__':
    unittest.main()
