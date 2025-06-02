"""
Module untuk similarity search menggunakan FAISS
"""

import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegoSearchIndex:
    """
    Kelas untuk melakukan similarity search menggunakan FAISS
    """
    
    def __init__(self, 
                 dimension: int,
                 index_type: str = 'L2',
                 nlist: int = 100,
                 nprobe: int = 10):
        """
        Inisialisasi FAISS index
        
        Args:
            dimension: Dimensi feature vector
            index_type: Tipe index FAISS ('L2' atau 'IVF')
            nlist: Jumlah cluster untuk IVF index
            nprobe: Jumlah cluster yang dicari saat query
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Inisialisasi index
        if index_type == 'L2':
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_L2
            )
            self.index.nprobe = nprobe
        else:
            raise ValueError(f"Tipe index tidak valid: {index_type}")
        
        # Untuk menyimpan metadata
        self.metadata = []
        
        logger.info(
            f"Index diinisialisasi dengan dimension={dimension}, "
            f"type={index_type}"
        )

    def add_items(self, 
                 features: np.ndarray,
                 metadata_list: List[Dict]):
        """
        Tambahkan item ke index
        
        Args:
            features: Matrix numpy berisi feature vectors
            metadata_list: List metadata untuk setiap feature vector
        """
        if len(features) != len(metadata_list):
            raise ValueError(
                "Jumlah features dan metadata harus sama"
            )
        
        # Training untuk IVF index
        if self.index_type == 'IVF' and not self.index.is_trained:
            self.index.train(features)
            logger.info("Index IVF telah ditraining")
        
        # Tambahkan ke index
        self.index.add(features)
        self.metadata.extend(metadata_list)
        
        logger.info(f"Menambahkan {len(features)} item ke index")

    def search(self, 
              query_vector: np.ndarray,
              k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Cari item yang similar
        
        Args:
            query_vector: Query feature vector
            k: Jumlah hasil yang diinginkan
            
        Returns:
            List tuple (metadata, distance)
        """
        # Reshape query vector jika perlu
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Gabungkan dengan metadata
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS returns -1 for not found
                results.append((self.metadata[idx], float(dist)))
        
        return results

    def save_index(self, 
                  path: str,
                  include_timestamp: bool = True):
        """
        Simpan index dan metadata ke file
        
        Args:
            path: Path dasar untuk menyimpan file
            include_timestamp: Tambahkan timestamp ke nama file
        """
        # Generate nama file
        base_path = Path(path)
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = base_path.parent / f"{base_path.stem}_{timestamp}{base_path.suffix}"
        
        # Buat direktori jika belum ada
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan index
        index_path = str(base_path)
        faiss.write_index(self.index, index_path)
        
        # Simpan metadata dan konfigurasi
        metadata_path = base_path.with_suffix('.meta')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'config': {
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'nlist': self.nlist,
                    'nprobe': self.nprobe
                }
            }, f)
        
        logger.info(
            f"Index disimpan ke {index_path}, "
            f"metadata disimpan ke {metadata_path}"
        )

    @classmethod
    def load_index(cls, path: str) -> 'LegoSearchIndex':
        """
        Load index dan metadata dari file
        
        Args:
            path: Path ke file index
            
        Returns:
            Instance LegoSearchIndex
        """
        # Load metadata dan config
        metadata_path = Path(path).with_suffix('.meta')
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        # Buat instance baru
        instance = cls(
            dimension=data['config']['dimension'],
            index_type=data['config']['index_type'],
            nlist=data['config']['nlist'],
            nprobe=data['config']['nprobe']
        )
        
        # Load index
        instance.index = faiss.read_index(path)
        instance.metadata = data['metadata']
        
        logger.info(f"Index diload dari: {path}")
        return instance

    def get_stats(self) -> Dict:
        """
        Dapatkan statistik index
        
        Returns:
            Dictionary berisi statistik
        """
        return {
            'total_items': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'nlist': self.nlist if self.index_type == 'IVF' else None,
            'nprobe': self.nprobe if self.index_type == 'IVF' else None,
            'is_trained': getattr(self.index, 'is_trained', True)
        }

    def clear(self):
        """
        Hapus semua item dari index
        """
        if self.index_type == 'L2':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, faiss.METRIC_L2
            )
            self.index.nprobe = self.nprobe
        
        self.metadata = []
        logger.info("Index telah dikosongkan")

    def update_metadata(self, 
                       index: int,
                       metadata: Dict):
        """
        Update metadata untuk item tertentu
        
        Args:
            index: Index item yang akan diupdate
            metadata: Metadata baru
        """
        if 0 <= index < len(self.metadata):
            self.metadata[index].update(metadata)
        else:
            raise ValueError(f"Index tidak valid: {index}")
