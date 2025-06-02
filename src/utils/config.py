"""
Module untuk configuration management
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """
    Konfigurasi untuk model
    """
    num_classes: int = 10
    pretrained: bool = True
    feature_dim: int = 1792
    dropout_rate: float = 0.5

@dataclass
class TrainingConfig:
    """
    Konfigurasi untuk training
    """
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    val_split: float = 0.2
    early_stopping_patience: int = 5
    seed: int = 42
    device: str = "auto"
    augment: bool = True

@dataclass
class PreprocessingConfig:
    """
    Konfigurasi untuk preprocessing
    """
    min_contour_area: int = 1000
    blur_kernel: int = 5
    target_size: tuple = (224, 224)
    save_visualization: bool = False
    max_workers: int = 4

@dataclass
class SearchConfig:
    """
    Konfigurasi untuk similarity search
    """
    index_type: str = "L2"  # L2 atau IVF
    nlist: int = 100
    nprobe: int = 10
    dimension: int = 1792

@dataclass
class APIConfig:
    """
    Konfigurasi untuk API
    """
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"

@dataclass
class LegonizerConfig:
    """
    Konfigurasi utama untuk Legonizer4
    """
    model: ModelConfig
    training: TrainingConfig
    preprocessing: PreprocessingConfig
    search: SearchConfig
    api: APIConfig
    
    # Paths
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    models_dir: str = "data/models"
    metrics_dir: str = "data/metrics"
    
    def __post_init__(self):
        """
        Post-initialization untuk validasi dan setup
        """
        # Pastikan semua path ada
        for path_attr in ['data_dir', 'raw_data_dir', 'processed_data_dir', 
                         'models_dir', 'metrics_dir']:
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)

class ConfigManager:
    """
    Manager untuk konfigurasi Legonizer4
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inisialisasi config manager
        
        Args:
            config_path: Path ke file konfigurasi
        """
        self.config_path = config_path or "config.json"
        self.config = self._load_or_create_config()

    def _load_or_create_config(self) -> LegonizerConfig:
        """
        Load konfigurasi dari file atau buat default
        
        Returns:
            Instance LegonizerConfig
        """
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                
                # Parse nested configs
                config = LegonizerConfig(
                    model=ModelConfig(**config_dict.get('model', {})),
                    training=TrainingConfig(**config_dict.get('training', {})),
                    preprocessing=PreprocessingConfig(**config_dict.get('preprocessing', {})),
                    search=SearchConfig(**config_dict.get('search', {})),
                    api=APIConfig(**config_dict.get('api', {})),
                    **{k: v for k, v in config_dict.items() 
                       if k not in ['model', 'training', 'preprocessing', 'search', 'api']}
                )
                
                logger.info(f"Konfigurasi diload dari: {config_file}")
                return config
                
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Menggunakan default config.")
        
        # Buat default config
        config = LegonizerConfig(
            model=ModelConfig(),
            training=TrainingConfig(),
            preprocessing=PreprocessingConfig(),
            search=SearchConfig(),
            api=APIConfig()
        )
        
        # Simpan default config
        self.save_config(config)
        logger.info(f"Default config dibuat dan disimpan ke: {config_file}")
        
        return config

    def save_config(self, config: Optional[LegonizerConfig] = None):
        """
        Simpan konfigurasi ke file
        
        Args:
            config: Instance LegonizerConfig (optional, default menggunakan self.config)
        """
        if config is None:
            config = self.config
        
        # Convert ke dictionary
        config_dict = asdict(config)
        
        # Simpan ke file
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Konfigurasi disimpan ke: {self.config_path}")

    def update_config(self, **kwargs):
        """
        Update konfigurasi dengan nilai baru
        
        Args:
            **kwargs: Key-value pairs untuk update
        """
        # Update nested configs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                if isinstance(value, dict):
                    # Update nested config
                    nested_config = getattr(self.config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    # Update top-level config
                    setattr(self.config, key, value)
        
        # Simpan perubahan
        self.save_config()

    def get_config(self) -> LegonizerConfig:
        """
        Dapatkan konfigurasi saat ini
        
        Returns:
            Instance LegonizerConfig
        """
        return self.config

    def reset_to_default(self):
        """
        Reset konfigurasi ke default
        """
        self.config = LegonizerConfig(
            model=ModelConfig(),
            training=TrainingConfig(),
            preprocessing=PreprocessingConfig(),
            search=SearchConfig(),
            api=APIConfig()
        )
        self.save_config()
        logger.info("Konfigurasi direset ke default")

    def validate_config(self) -> bool:
        """
        Validasi konfigurasi
        
        Returns:
            True jika valid, False jika tidak
        """
        try:
            # Validasi model config
            assert self.config.model.num_classes > 0, "num_classes harus > 0"
            assert 0 <= self.config.model.dropout_rate <= 1, "dropout_rate harus antara 0-1"
            
            # Validasi training config
            assert self.config.training.batch_size > 0, "batch_size harus > 0"
            assert self.config.training.num_epochs > 0, "num_epochs harus > 0"
            assert self.config.training.learning_rate > 0, "learning_rate harus > 0"
            assert 0 < self.config.training.val_split < 1, "val_split harus antara 0-1"
            
            # Validasi preprocessing config
            assert self.config.preprocessing.min_contour_area > 0, "min_contour_area harus > 0"
            assert len(self.config.preprocessing.target_size) == 2, "target_size harus tuple (width, height)"
            
            # Validasi search config
            assert self.config.search.index_type in ["L2", "IVF"], "index_type harus L2 atau IVF"
            assert self.config.search.dimension > 0, "dimension harus > 0"
            
            # Validasi API config
            assert 1 <= self.config.api.port <= 65535, "port harus antara 1-65535"
            
            logger.info("Konfigurasi valid")
            return True
            
        except AssertionError as e:
            logger.error(f"Konfigurasi tidak valid: {e}")
            return False

    def get_env_overrides(self) -> Dict[str, Any]:
        """
        Dapatkan override dari environment variables
        
        Returns:
            Dictionary berisi override values
        """
        overrides = {}
        
        # Environment variable mappings
        env_mappings = {
            'LEGONIZER_BATCH_SIZE': ('training', 'batch_size', int),
            'LEGONIZER_LEARNING_RATE': ('training', 'learning_rate', float),
            'LEGONIZER_NUM_EPOCHS': ('training', 'num_epochs', int),
            'LEGONIZER_API_PORT': ('api', 'port', int),
            'LEGONIZER_API_HOST': ('api', 'host', str),
            'LEGONIZER_DEVICE': ('training', 'device', str),
            'LEGONIZER_DATA_DIR': (None, 'data_dir', str),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = type_func(os.environ[env_var])
                    if section:
                        if section not in overrides:
                            overrides[section] = {}
                        overrides[section][key] = value
                    else:
                        overrides[key] = value
                except ValueError as e:
                    logger.warning(f"Invalid environment variable {env_var}: {e}")
        
        return overrides

    def apply_env_overrides(self):
        """
        Aplikasikan environment variable overrides
        """
        overrides = self.get_env_overrides()
        if overrides:
            logger.info(f"Applying environment overrides: {overrides}")
            self.update_config(**overrides)

# Global config manager instance
_config_manager = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Dapatkan global config manager instance
    
    Args:
        config_path: Path ke file konfigurasi
        
    Returns:
        Instance ConfigManager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config() -> LegonizerConfig:
    """
    Shortcut untuk mendapatkan konfigurasi
    
    Returns:
        Instance LegonizerConfig
    """
    return get_config_manager().get_config()
