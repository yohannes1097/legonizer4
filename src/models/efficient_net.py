"""
Module untuk model architecture menggunakan EfficientNetB4
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from typing import Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientNetFeatureExtractor(nn.Module):
    """
    Model untuk ekstraksi fitur dan klasifikasi menggunakan EfficientNetB4
    """
    
    def __init__(self, 
                 num_classes: int,
                 pretrained: bool = True,
                 feature_dim: int = 1792,
                 dropout_rate: float = 0.5):
        """
        Inisialisasi model
        
        Args:
            num_classes: Jumlah kelas (model ID) yang akan diklasifikasi
            pretrained: Menggunakan pre-trained weights atau tidak
            feature_dim: Dimensi feature vector dari EfficientNetB4
            dropout_rate: Dropout rate untuk regularisasi
        """
        super().__init__()
        
        # Load EfficientNetB4 pre-trained
        self.efficient_net = EfficientNet.from_pretrained(
            'efficientnet-b4'
        ) if pretrained else EfficientNet.from_name('efficientnet-b4')
        
        # Freeze beberapa layer awal
        if pretrained:
            for param in list(self.efficient_net.parameters())[:-20]:
                param.requires_grad = False
        
        # Ganti classifier head
        self.efficient_net._fc = nn.Identity()
        
        # Feature extraction dan klasifikasi
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        logger.info(
            f"Model diinisialisasi dengan {num_classes} kelas "
            f"(pretrained={pretrained})"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor dengan shape (batch_size, 3, H, W)
            
        Returns:
            Tuple (logits, features)
                - logits: Prediksi kelas
                - features: Feature vectors untuk similarity search
        """
        # Extract features
        features = self.efficient_net(x)
        
        # Klasifikasi
        logits = self.classifier(features)
        
        return logits, features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ekstrak features saja tanpa klasifikasi
        
        Args:
            x: Input tensor dengan shape (batch_size, 3, H, W)
            
        Returns:
            Feature vectors
        """
        with torch.no_grad():
            features = self.efficient_net(x)
        return features

    def save_model(self, path: str):
        """
        Simpan model ke file
        
        Args:
            path: Path untuk menyimpan model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'feature_dim': self.efficient_net._fc.in_features,
            'num_classes': self.classifier[-1].out_features
        }, path)
        logger.info(f"Model disimpan ke: {path}")

    @classmethod
    def load_model(cls, 
                  path: str, 
                  map_location: Optional[str] = None) -> 'EfficientNetFeatureExtractor':
        """
        Load model dari file
        
        Args:
            path: Path ke file model
            map_location: Device untuk load model
            
        Returns:
            Instance EfficientNetFeatureExtractor
        """
        checkpoint = torch.load(path, map_location=map_location)
        
        model = cls(
            num_classes=checkpoint['num_classes'],
            feature_dim=checkpoint['feature_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model diload dari: {path}")
        return model

    def get_embedding_dim(self) -> int:
        """
        Dapatkan dimensi feature vector
        
        Returns:
            Dimensi feature vector
        """
        return self.efficient_net._fc.in_features
