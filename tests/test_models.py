"""
Unit tests untuk model dan training modules
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np

from src.models.efficient_net import EfficientNetFeatureExtractor
from src.models.trainer import LegoTrainer

class TestEfficientNetFeatureExtractor(unittest.TestCase):
    """
    Test cases untuk EfficientNetFeatureExtractor
    """
    
    def setUp(self):
        """
        Setup test environment
        """
        self.num_classes = 10
        self.model = EfficientNetFeatureExtractor(
            num_classes=self.num_classes,
            pretrained=False  # Untuk testing lebih cepat
        )
        self.batch_size = 4
        self.input_size = (224, 224)
    
    def test_model_structure(self):
        """
        Test struktur model
        """
        # Check output layer
        self.assertEqual(
            self.model.classifier[-1].out_features,
            self.num_classes
        )
        
        # Check feature extractor
        self.assertIsNotNone(self.model.features)
    
    def test_forward_pass(self):
        """
        Test forward pass
        """
        # Buat dummy input
        x = torch.randn(self.batch_size, 3, *self.input_size)
        
        # Forward pass
        outputs = self.model(x)
        
        # Check output shape
        self.assertEqual(
            outputs.shape,
            (self.batch_size, self.num_classes)
        )
    
    def test_extract_features(self):
        """
        Test ekstraksi features
        """
        # Buat dummy input
        x = torch.randn(self.batch_size, 3, *self.input_size)
        
        # Extract features
        features = self.model.extract_features(x)
        
        # Check feature dimensi
        self.assertEqual(features.shape[0], self.batch_size)
        self.assertEqual(features.shape[1], self.model.feature_dim)
    
    def test_save_load_model(self):
        """
        Test save dan load model
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pth"
            
            # Save model
            torch.save(self.model.state_dict(), model_path)
            
            # Load ke model baru
            new_model = EfficientNetFeatureExtractor(
                num_classes=self.num_classes,
                pretrained=False
            )
            new_model.load_state_dict(torch.load(model_path))
            
            # Test forward pass
            x = torch.randn(self.batch_size, 3, *self.input_size)
            with torch.no_grad():
                out1 = self.model(x)
                out2 = new_model(x)
            
            # Output harus sama
            torch.testing.assert_close(out1, out2)

class TestLegoTrainer(unittest.TestCase):
    """
    Test cases untuk LegoTrainer
    """
    
    def setUp(self):
        """
        Setup test environment
        """
        self.num_classes = 5
        self.model = EfficientNetFeatureExtractor(
            num_classes=self.num_classes,
            pretrained=False
        )
        self.trainer = LegoTrainer(
            model=self.model,
            num_epochs=2  # Untuk testing lebih cepat
        )
        self.temp_dir = tempfile.mkdtemp()
        
        # Buat dummy dataset
        self.num_samples = 20
        self.image_size = (224, 224)
        self.dummy_images = [
            str(Path(self.temp_dir) / f"img_{i}.jpg")
            for i in range(self.num_samples)
        ]
        self.dummy_labels = np.random.randint(
            0, self.num_classes, self.num_samples
        ).tolist()
        
        # Buat dummy image files
        for img_path in self.dummy_images:
            # Buat random image
            img = np.random.randint(
                0, 255,
                (*self.image_size, 3),
                dtype=np.uint8
            )
            # Simpan sebagai JPEG
            torch.save(
                torch.from_numpy(img).permute(2, 0, 1),
                img_path
            )
    
    def tearDown(self):
        """
        Cleanup test environment
        """
        shutil.rmtree(self.temp_dir)
    
    def test_train_step(self):
        """
        Test single training step
        """
        # Buat dummy batch
        batch_size = 4
        images = torch.randn(batch_size, 3, *self.image_size)
        labels = torch.randint(0, self.num_classes, (batch_size,))
        
        # Training step
        loss = self.trainer._train_step(images, labels)
        
        # Loss harus valid
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))
    
    def test_validation_step(self):
        """
        Test single validation step
        """
        # Buat dummy batch
        batch_size = 4
        images = torch.randn(batch_size, 3, *self.image_size)
        labels = torch.randint(0, self.num_classes, (batch_size,))
        
        # Validation step
        loss, acc = self.trainer._validation_step(images, labels)
        
        # Metrics harus valid
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isnan(acc))
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
    
    def test_train(self):
        """
        Test full training loop
        """
        # Split dummy data
        train_size = int(0.8 * self.num_samples)
        train_data = (
            self.dummy_images[:train_size],
            self.dummy_labels[:train_size]
        )
        val_data = (
            self.dummy_images[train_size:],
            self.dummy_labels[train_size:]
        )
        
        # Class mapping
        class_to_idx = {
            str(i): i for i in range(self.num_classes)
        }
        
        # Training
        results = self.trainer.train(
            train_data=train_data,
            val_data=val_data,
            model_dir=self.temp_dir,
            class_to_idx=class_to_idx
        )
        
        # Check results
        self.assertIn('best_model_path', results)
        self.assertIn('best_val_accuracy', results)
        self.assertIn('history', results)
        
        # History harus berisi metrics
        history = results['history']
        self.assertIn('train_loss', history)
        self.assertIn('train_acc', history)
        self.assertIn('val_loss', history)
        self.assertIn('val_acc', history)
        
        # Model file harus ada
        self.assertTrue(Path(results['best_model_path']).exists())
    
    def test_extract_features_batch(self):
        """
        Test batch feature extraction
        """
        # Extract features
        features = self.trainer.extract_features_batch(
            self.dummy_images[:5]
        )
        
        # Check hasil
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 5)
        self.assertEqual(features.shape[1], self.model.feature_dim)

if __name__ == '__main__':
    unittest.main()
