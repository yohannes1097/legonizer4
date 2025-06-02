"""
Module untuk training model LEGO classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm

from .efficient_net import EfficientNetFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegoImageDataset(Dataset):
    """
    Dataset untuk gambar LEGO
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: List[int],
                 transform: Optional[transforms.Compose] = None):
        """
        Inisialisasi dataset
        
        Args:
            image_paths: List path gambar
            labels: List label (model ID index)
            transform: Transformasi yang akan diaplikasikan ke gambar
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load gambar
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Aplikasikan transformasi
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

class LegoTrainer:
    """
    Kelas untuk training dan evaluasi model
    """
    
    def __init__(self,
                 model: EfficientNetFeatureExtractor,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 50,
                 early_stopping_patience: int = 5):
        """
        Inisialisasi trainer
        
        Args:
            model: Instance EfficientNetFeatureExtractor
            device: Device untuk training ('cuda' atau 'cpu')
            learning_rate: Learning rate untuk optimizer
            batch_size: Batch size untuk training
            num_epochs: Jumlah epoch maksimum
            early_stopping_patience: Jumlah epoch untuk early stopping
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Setup loss function dan optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        logger.info(
            f"Trainer diinisialisasi dengan device={device}, "
            f"lr={learning_rate}, batch_size={batch_size}"
        )

    def train_epoch(self, 
                   train_loader: DataLoader) -> Tuple[float, float]:
        """
        Training untuk satu epoch
        
        Args:
            train_loader: DataLoader untuk training data
            
        Returns:
            Tuple (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

    def validate(self, 
                val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validasi model
        
        Args:
            val_loader: DataLoader untuk validation data
            
        Returns:
            Tuple (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

    def train(self,
             train_data: Tuple[List[str], List[int]],
             val_data: Tuple[List[str], List[int]],
             model_dir: str,
             class_to_idx: Dict[str, int]) -> dict:
        """
        Training model lengkap dengan early stopping
        
        Args:
            train_data: Tuple (image_paths, labels) untuk training
            val_data: Tuple (image_paths, labels) untuk validation
            model_dir: Direktori untuk menyimpan model
            class_to_idx: Mapping class name ke index
            
        Returns:
            Dictionary berisi history training dan path model terbaik
        """
        # Setup datasets dan dataloaders
        train_dataset = LegoImageDataset(train_data[0], train_data[1])
        val_dataset = LegoImageDataset(val_data[0], val_data[1])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Untuk tracking metrics
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0
        patience_counter = 0
        best_model_path = None
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Check untuk model terbaik
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Simpan model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = Path(model_dir) / f"model_{timestamp}.pth"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.model.save_model(str(model_path))
                
                # Simpan class mapping
                mapping_path = model_path.with_suffix('.json')
                with open(mapping_path, 'w') as f:
                    json.dump(class_to_idx, f)
                
                best_model_path = str(model_path)
                logger.info(f"Model terbaik disimpan: {best_model_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping pada epoch {epoch+1} "
                    f"(tidak ada improvement selama {self.early_stopping_patience} epoch)"
                )
                break
        
        return {
            'history': history,
            'best_model_path': best_model_path,
            'best_val_accuracy': best_val_acc
        }

    def predict(self, 
               image_path: str,
               class_to_idx: Dict[str, int],
               top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Prediksi kelas untuk satu gambar
        
        Args:
            image_path: Path ke file gambar
            class_to_idx: Mapping class name ke index
            top_k: Jumlah prediksi top-k yang diinginkan
            
        Returns:
            List tuple (class_name, probability)
        """
        # Prepare image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(image)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=top_k)
        
        # Convert indices to class names
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        predictions = [
            (idx_to_class[idx.item()], prob.item())
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]
        
        return predictions

    def extract_features_batch(self, 
                             image_paths: List[str]) -> np.ndarray:
        """
        Ekstrak features dari batch gambar
        
        Args:
            image_paths: List path gambar
            
        Returns:
            Array numpy berisi features
        """
        # Prepare dataset dan dataloader
        dummy_labels = [0] * len(image_paths)  # Labels tidak digunakan
        dataset = LegoImageDataset(image_paths, dummy_labels)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Extract features
        self.model.eval()
        features_list = []
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                _, features = self.model(images)
                features_list.append(features.cpu().numpy())
        
        # Gabungkan semua features
        features = np.concatenate(features_list, axis=0)
        return features
