"""
Module untuk loading dan preprocessing dataset LEGO
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import random
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegoDataLoader:
    """
    Kelas untuk loading dan preprocessing dataset LEGO
    """
    
    def __init__(self, 
                 data_dir: str,
                 val_split: float = 0.2,
                 seed: int = 42):
        """
        Inisialisasi data loader
        
        Args:
            data_dir: Path ke direktori data
            val_split: Proporsi data untuk validation
            seed: Random seed untuk reproducibility
        """
        self.data_dir = Path(data_dir)
        self.val_split = val_split
        self.seed = seed
        
        # Validasi direktori data
        if not self.data_dir.exists():
            raise ValueError(f"Directory tidak ditemukan: {data_dir}")
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load class mapping
        self.class_to_idx = self._create_class_mapping()
        logger.info(f"Menemukan {len(self.class_to_idx)} kelas LEGO")

    def _create_class_mapping(self) -> Dict[str, int]:
        """
        Buat mapping dari nama kelas (model ID) ke index
        
        Returns:
            Dictionary mapping class name ke index
        """
        # Dapatkan semua subdirektori (setiap direktori adalah satu kelas)
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Buat mapping
        class_to_idx = {d.name: idx for idx, d in enumerate(sorted(class_dirs))}
        return class_to_idx

    def _get_image_files(self) -> List[Tuple[str, int]]:
        """
        Dapatkan semua file gambar dan labelnya
        
        Returns:
            List tuple (image_path, label_idx)
        """
        image_files = []
        
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        # Scan setiap direktori kelas
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            
            # Dapatkan semua file gambar
            for ext in valid_extensions:
                image_files.extend([
                    (str(f), class_idx)
                    for f in class_dir.glob(f"*{ext}")
                ])
        
        return image_files

    def get_data_splits(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Split data menjadi training dan validation sets
        
        Returns:
            Tuple (train_paths, train_labels, val_paths, val_labels)
        """
        # Dapatkan semua file dan label
        image_files = self._get_image_files()
        if not image_files:
            raise ValueError(f"Tidak ada file gambar ditemukan di {self.data_dir}")
        
        # Unzip paths dan labels
        paths, labels = zip(*image_files)
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels,
            test_size=self.val_split,
            stratify=labels,
            random_state=self.seed
        )
        
        logger.info(
            f"Dataset split: {len(train_paths)} training, "
            f"{len(val_paths)} validation images"
        )
        
        return train_paths, train_labels, val_paths, val_labels

    def get_class_weights(self) -> torch.Tensor:
        """
        Hitung class weights untuk imbalanced dataset
        
        Returns:
            Tensor berisi class weights
        """
        # Dapatkan semua label
        _, labels = zip(*self._get_image_files())
        
        # Hitung jumlah sampel per kelas
        class_counts = np.bincount(labels)
        
        # Hitung weights
        weights = 1.0 / class_counts
        weights = weights / weights.sum()  # Normalize
        
        return torch.FloatTensor(weights)

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Dapatkan distribusi kelas dalam dataset
        
        Returns:
            Dictionary berisi jumlah sampel per kelas
        """
        # Dapatkan semua label
        _, labels = zip(*self._get_image_files())
        
        # Hitung distribusi
        counts = np.bincount(labels)
        
        # Convert ke dictionary dengan nama kelas
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        distribution = {
            idx_to_class[idx]: count.item()
            for idx, count in enumerate(counts)
        }
        
        return distribution

    def preview_augmentations(self, 
                            image_path: str,
                            output_dir: str,
                            num_previews: int = 5):
        """
        Preview hasil augmentasi pada satu gambar
        
        Args:
            image_path: Path ke file gambar
            output_dir: Direktori untuk menyimpan preview
            num_previews: Jumlah preview yang diinginkan
        """
        # Load gambar
        image = Image.open(image_path).convert('RGB')
        
        # Buat transformasi augmentasi
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            )
        ])
        
        # Buat direktori output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate dan simpan preview
        for i in range(num_previews):
            augmented = augmentation(image)
            save_path = output_path / f"preview_{i+1}.jpg"
            augmented.save(save_path)
        
        logger.info(f"Preview augmentasi disimpan di: {output_dir}")

class LegoDataset(Dataset):
    """
    Dataset class untuk LEGO images
    """
    
    def __init__(self,
                 image_paths: List[str],
                 labels: List[int],
                 transform: Optional[transforms.Compose] = None,
                 augment: bool = False):
        """
        Inisialisasi dataset
        
        Args:
            image_paths: List path gambar
            labels: List label (class indices)
            transform: Transformasi tambahan
            augment: Flag untuk menggunakan augmentasi data
        """
        self.image_paths = image_paths
        self.labels = labels
        
        # Base transform
        base_transform = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        # Augmentation transform
        augment_transform = []
        if augment:
            augment_transform = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                )
            ]
        
        # Combine transforms
        all_transforms = augment_transform + base_transform
        if transform:
            all_transforms.append(transform)
        
        self.transform = transforms.Compose(all_transforms)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load gambar
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transformasi
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]
