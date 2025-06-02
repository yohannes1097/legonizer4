"""
Unit tests untuk utils modules
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np

from src.utils.config import ConfigManager, LegonizerConfig
from src.utils.metrics import AccuracyReporter
from src.utils.data_loader import LegoDataLoader

class TestConfigManager(unittest.TestCase):
    """
    Test cases untuk ConfigManager
    """
    
    def setUp(self):
        """
        Setup test environment
        """
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        
    def tearDown(self):
        """
        Cleanup test environment
        """
        shutil.rmtree(self.temp_dir)
    
    def test_create_default_config(self):
        """
        Test pembuatan default config
        """
        config_manager = ConfigManager(str(self.config_path))
        config = config_manager.get_config()
        
        # Check config structure
        self.assertIsInstance(config, LegonizerConfig)
        self.assertEqual(config.model.num_classes, 10)
        self.assertEqual(config.training.batch_size, 32)
        self.assertEqual(config.api.port, 8000)
        
        # Config file harus dibuat
        self.assertTrue(self.config_path.exists())
    
    def test_save_load_config(self):
        """
        Test save dan load config
        """
        config_manager = ConfigManager(str(self.config_path))
        
        # Update config
        config_manager.update_config(
            training={'batch_size': 64, 'num_epochs': 100}
        )
        
        # Load config baru
        new_config_manager = ConfigManager(str(self.config_path))
        config = new_config_manager.get_config()
        
        # Check updated values
        self.assertEqual(config.training.batch_size, 64)
        self.assertEqual(config.training.num_epochs, 100)
    
    def test_validate_config(self):
        """
        Test validasi config
        """
        config_manager = ConfigManager(str(self.config_path))
        
        # Valid config
        self.assertTrue(config_manager.validate_config())
        
        # Invalid config
        config_manager.config.model.num_classes = -1
        self.assertFalse(config_manager.validate_config())
    
    def test_reset_to_default(self):
        """
        Test reset config ke default
        """
        config_manager = ConfigManager(str(self.config_path))
        
        # Update config
        config_manager.update_config(training={'batch_size': 128})
        
        # Reset
        config_manager.reset_to_default()
        
        # Check default values
        config = config_manager.get_config()
        self.assertEqual(config.training.batch_size, 32)

class TestAccuracyReporter(unittest.TestCase):
    """
    Test cases untuk AccuracyReporter
    """
    
    def setUp(self):
        """
        Setup test environment
        """
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test_accuracy.json"
        self.report_dir = Path(self.temp_dir) / "reports"
        
        self.reporter = AccuracyReporter(
            log_file=str(self.log_file),
            report_dir=str(self.report_dir)
        )
    
    def tearDown(self):
        """
        Cleanup test environment
        """
        shutil.rmtree(self.temp_dir)
    
    def test_log_training_accuracy(self):
        """
        Test logging training accuracy
        """
        self.reporter.log_training_accuracy(
            model_path="test_model.pth",
            train_accuracy=0.95,
            val_accuracy=0.87,
            num_epochs=50,
            num_classes=10,
            training_time=3600.0
        )
        
        # Check log file dibuat
        self.assertTrue(self.log_file.exists())
        
        # Check log content
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['type'], 'training')
        self.assertEqual(logs[0]['train_accuracy'], 0.95)
    
    def test_log_prediction_accuracy(self):
        """
        Test logging prediction accuracy
        """
        self.reporter.log_prediction_accuracy(
            predicted_class="31313",
            actual_class="31313",
            confidence=0.92
        )
        
        # Check log
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['type'], 'prediction')
        self.assertTrue(logs[0]['is_correct'])
    
    def test_get_training_stats(self):
        """
        Test mendapatkan training statistics
        """
        # Log beberapa training
        for i in range(3):
            self.reporter.log_training_accuracy(
                model_path=f"model_{i}.pth",
                train_accuracy=0.9 + i * 0.01,
                val_accuracy=0.8 + i * 0.01,
                num_epochs=50,
                num_classes=10,
                training_time=3600.0
            )
        
        stats = self.reporter.get_training_stats()
        
        self.assertEqual(stats['total_trainings'], 3)
        self.assertEqual(stats['best_train_accuracy'], 0.92)
        self.assertEqual(stats['best_val_accuracy'], 0.82)
    
    def test_get_prediction_stats(self):
        """
        Test mendapatkan prediction statistics
        """
        # Log beberapa predictions
        predictions = [
            ("31313", "31313", 0.9),  # Correct
            ("42115", "31313", 0.7),  # Incorrect
            ("31313", "31313", 0.8),  # Correct
        ]
        
        for pred, actual, conf in predictions:
            self.reporter.log_prediction_accuracy(pred, actual, conf)
        
        stats = self.reporter.get_prediction_stats()
        
        self.assertEqual(stats['total_predictions'], 3)
        self.assertEqual(stats['correct_predictions'], 2)
        self.assertAlmostEqual(stats['overall_accuracy'], 2/3, places=2)
    
    def test_generate_accuracy_report(self):
        """
        Test generate HTML report
        """
        # Log some data
        self.reporter.log_training_accuracy(
            model_path="test_model.pth",
            train_accuracy=0.95,
            val_accuracy=0.87,
            num_epochs=50,
            num_classes=10,
            training_time=3600.0
        )
        
        # Generate report
        report_path = self.reporter.generate_accuracy_report()
        
        # Check report file
        self.assertTrue(Path(report_path).exists())
        
        # Check HTML content
        with open(report_path, 'r') as f:
            content = f.read()
        
        self.assertIn("Legonizer4 Accuracy Report", content)
        self.assertIn("Training Statistics", content)

class TestLegoDataLoader(unittest.TestCase):
    """
    Test cases untuk LegoDataLoader
    """
    
    def setUp(self):
        """
        Setup test environment
        """
        self.temp_dir = tempfile.mkdtemp()
        
        # Buat struktur data dummy
        self.data_dir = Path(self.temp_dir) / "data"
        
        # Buat beberapa kelas
        classes = ["31313", "42115", "10264"]
        for class_name in classes:
            class_dir = self.data_dir / class_name
            class_dir.mkdir(parents=True)
            
            # Buat dummy image files
            for i in range(5):
                img_file = class_dir / f"img_{i}.jpg"
                img_file.write_text("dummy image content")
    
    def tearDown(self):
        """
        Cleanup test environment
        """
        shutil.rmtree(self.temp_dir)
    
    def test_create_class_mapping(self):
        """
        Test pembuatan class mapping
        """
        data_loader = LegoDataLoader(str(self.data_dir))
        
        # Check class mapping
        self.assertEqual(len(data_loader.class_to_idx), 3)
        self.assertIn("31313", data_loader.class_to_idx)
        self.assertIn("42115", data_loader.class_to_idx)
        self.assertIn("10264", data_loader.class_to_idx)
    
    def test_get_data_splits(self):
        """
        Test data splitting
        """
        data_loader = LegoDataLoader(str(self.data_dir), val_split=0.2)
        
        train_paths, train_labels, val_paths, val_labels = data_loader.get_data_splits()
        
        # Check split proportions
        total_samples = len(train_paths) + len(val_paths)
        val_ratio = len(val_paths) / total_samples
        
        self.assertAlmostEqual(val_ratio, 0.2, delta=0.1)
        
        # Check labels are valid
        for label in train_labels + val_labels:
            self.assertIn(label, range(len(data_loader.class_to_idx)))
    
    def test_get_class_distribution(self):
        """
        Test mendapatkan distribusi kelas
        """
        data_loader = LegoDataLoader(str(self.data_dir))
        distribution = data_loader.get_class_distribution()
        
        # Check distribusi
        self.assertEqual(len(distribution), 3)
        for class_name, count in distribution.items():
            self.assertEqual(count, 5)  # 5 images per class
    
    def test_invalid_data_dir(self):
        """
        Test dengan direktori data tidak valid
        """
        with self.assertRaises(ValueError):
            LegoDataLoader("/path/that/does/not/exist")

if __name__ == '__main__':
    unittest.main()
