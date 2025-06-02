"""
Unit tests untuk search module
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import faiss

from src.search.faiss_index import LegoSearchIndex

class TestLegoSearchIndex(unittest.TestCase):
    """
    Test cases untuk LegoSearchIndex
    """
    
    def setUp(self):
        """
        Setup test environment
        """
        self.dimension = 1792  # EfficientNetB4 feature dimension
        self.num_vectors = 100
        self.index = LegoSearchIndex(dimension=self.dimension)
        self.temp_dir = tempfile.mkdtemp()
        
        # Buat dummy vectors dan metadata
        self.vectors = np.random.random((self.num_vectors, self.dimension)).astype('float32')
        self.metadata = [
            {
                'model_id': f'model_{i}',
                'name': f'Test Model {i}',
                'image_path': f'/path/to/image_{i}.jpg'
            }
            for i in range(self.num_vectors)
        ]
    
    def tearDown(self):
        """
        Cleanup test environment
        """
        shutil.rmtree(self.temp_dir)
    
    def test_add_items(self):
        """
        Test menambahkan items ke index
        """
        # Add items
        self.index.add_items(self.vectors, self.metadata)
        
        # Check jumlah items
        self.assertEqual(self.index.index.ntotal, self.num_vectors)
        self.assertEqual(len(self.index.metadata), self.num_vectors)
    
    def test_search(self):
        """
        Test pencarian similarity
        """
        # Add items
        self.index.add_items(self.vectors, self.metadata)
        
        # Buat query vector
        query = np.random.random((1, self.dimension)).astype('float32')
        
        # Search
        k = 5
        results = self.index.search(query, k=k)
        
        # Check hasil
        self.assertEqual(len(results), k)
        for meta, dist in results:
            self.assertIsInstance(meta, dict)
            self.assertIsInstance(dist, float)
            self.assertIn('model_id', meta)
    
    def test_save_load_index(self):
        """
        Test save dan load index
        """
        # Add items
        self.index.add_items(self.vectors, self.metadata)
        
        # Save index
        index_path = Path(self.temp_dir) / "test_index.faiss"
        self.index.save_index(str(index_path))
        
        # Load ke index baru
        new_index = LegoSearchIndex(dimension=self.dimension)
        new_index.load_index(str(index_path))
        
        # Check jumlah items sama
        self.assertEqual(
            self.index.index.ntotal,
            new_index.index.ntotal
        )
        self.assertEqual(
            len(self.index.metadata),
            len(new_index.metadata)
        )
        
        # Check hasil search sama
        query = np.random.random((1, self.dimension)).astype('float32')
        results1 = self.index.search(query, k=5)
        results2 = new_index.search(query, k=5)
        
        for r1, r2 in zip(results1, results2):
            self.assertEqual(r1[0]['model_id'], r2[0]['model_id'])
            self.assertAlmostEqual(r1[1], r2[1], places=5)
    
    def test_clear_index(self):
        """
        Test membersihkan index
        """
        # Add items
        self.index.add_items(self.vectors, self.metadata)
        
        # Clear index
        self.index.clear()
        
        # Check index kosong
        self.assertEqual(self.index.index.ntotal, 0)
        self.assertEqual(len(self.index.metadata), 0)
    
    def test_invalid_dimension(self):
        """
        Test input vector dengan dimensi salah
        """
        wrong_dim = self.dimension + 1
        invalid_vectors = np.random.random((10, wrong_dim)).astype('float32')
        
        with self.assertRaises(ValueError):
            self.index.add_items(invalid_vectors, self.metadata[:10])
    
    def test_metadata_mismatch(self):
        """
        Test jumlah metadata tidak sesuai vectors
        """
        with self.assertRaises(ValueError):
            self.index.add_items(
                self.vectors,
                self.metadata[:self.num_vectors-1]  # Kurang satu
            )
    
    def test_empty_search(self):
        """
        Test search pada index kosong
        """
        query = np.random.random((1, self.dimension)).astype('float32')
        results = self.index.search(query, k=5)
        
        self.assertEqual(len(results), 0)
    
    def test_large_k_search(self):
        """
        Test search dengan k lebih besar dari jumlah items
        """
        # Add items
        self.index.add_items(self.vectors, self.metadata)
        
        # Search dengan k besar
        query = np.random.random((1, self.dimension)).astype('float32')
        k = self.num_vectors + 10
        results = self.index.search(query, k=k)
        
        # Hasil tidak boleh lebih dari jumlah items
        self.assertEqual(len(results), self.num_vectors)

if __name__ == '__main__':
    unittest.main()
