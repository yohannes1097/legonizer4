"""
Contoh penggunaan API Legonizer4
"""

import argparse
import json
import logging
from pathlib import Path
import requests
from typing import Optional, Dict, List
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Legonizer4Client:
    """
    Client untuk Legonizer4 API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Inisialisasi client
        
        Args:
            base_url: URL base API
        """
        self.base_url = base_url.rstrip('/')
        
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     **kwargs) -> Dict:
        """
        Buat HTTP request ke API
        
        Args:
            method: HTTP method (GET, POST, etc)
            endpoint: API endpoint
            **kwargs: Arguments untuk requests
            
        Returns:
            Response JSON
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {e}")
            raise
    
    def health_check(self) -> Dict:
        """
        Check status API
        
        Returns:
            Status API
        """
        return self._make_request('GET', '/health')
    
    def identify_lego(self, 
                     image_path: str,
                     top_n: int = 5) -> Dict:
        """
        Identifikasi LEGO dari gambar
        
        Args:
            image_path: Path ke file gambar
            top_n: Jumlah prediksi top-n yang diinginkan
            
        Returns:
            Hasil prediksi
        """
        # Prepare file
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'top_n': str(top_n)}
            
            return self._make_request(
                'POST',
                '/identify',
                files=files,
                data=data
            )
    
    def list_models(self) -> Dict:
        """
        Dapatkan daftar model yang tersedia
        
        Returns:
            List model
        """
        return self._make_request('GET', '/models')
    
    def get_accuracy(self) -> Dict:
        """
        Dapatkan report akurasi model
        
        Returns:
            Accuracy report
        """
        return self._make_request('GET', '/accuracy')
    
    def train_model(self,
                   data_dir: str,
                   val_split: float = 0.2,
                   num_epochs: int = 50) -> Dict:
        """
        Training ulang model
        
        Args:
            data_dir: Path ke direktori data training
            val_split: Proporsi data validation
            num_epochs: Jumlah epoch
            
        Returns:
            Training results
        """
        data = {
            'data_dir': data_dir,
            'val_split': str(val_split),
            'num_epochs': str(num_epochs)
        }
        
        return self._make_request('POST', '/train', data=data)

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Client untuk Legonizer4 API"
    )
    
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="URL base API"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Health check command
    subparsers.add_parser(
        "health",
        help="Check status API"
    )
    
    # Identify command
    identify_parser = subparsers.add_parser(
        "identify",
        help="Identifikasi LEGO dari gambar"
    )
    identify_parser.add_argument(
        "image",
        type=str,
        help="Path ke file gambar"
    )
    identify_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Jumlah prediksi top-n"
    )
    
    # List models command
    subparsers.add_parser(
        "list-models",
        help="List model yang tersedia"
    )
    
    # Get accuracy command
    subparsers.add_parser(
        "accuracy",
        help="Dapatkan report akurasi"
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Training ulang model"
    )
    train_parser.add_argument(
        "data_dir",
        type=str,
        help="Path ke direktori data training"
    )
    train_parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Proporsi data validation"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Jumlah epoch"
    )
    
    return parser.parse_args()

def format_predictions(predictions: List[Dict]) -> str:
    """
    Format hasil prediksi untuk display
    
    Args:
        predictions: List hasil prediksi
        
    Returns:
        Formatted string
    """
    output = []
    
    for i, pred_group in enumerate(predictions):
        output.append(f"\nObjek #{i+1}:")
        
        for j, pred in enumerate(pred_group):
            confidence = pred['confidence'] * 100
            similarity = pred.get('similarity', 0) * 100
            
            output.append(
                f"  {j+1}. Model ID: {pred['model_id']}"
                f" (Confidence: {confidence:.1f}%"
                f", Similarity: {similarity:.1f}%)"
            )
    
    return '\n'.join(output)

def main():
    """
    Fungsi utama
    """
    args = parse_arguments()
    
    try:
        # Inisialisasi client
        client = Legonizer4Client(args.url)
        
        if args.command == "health":
            # Health check
            result = client.health_check()
            print(json.dumps(result, indent=2))
            
        elif args.command == "identify":
            # Identifikasi gambar
            start_time = time.time()
            result = client.identify_lego(args.image, args.top_n)
            elapsed = time.time() - start_time
            
            print("\nHasil Identifikasi:")
            print(f"Waktu: {elapsed:.2f} detik")
            print(f"Jumlah objek terdeteksi: {result['num_objects_detected']}")
            print(format_predictions(result['predictions']))
            
        elif args.command == "list-models":
            # List models
            result = client.list_models()
            
            print("\nModel yang tersedia:")
            for model in result['models']:
                print(f"\nPath: {model['path']}")
                print(f"Created: {model['created']}")
                print(f"Jumlah kelas: {model['num_classes']}")
                print("Classes:")
                for class_name in model['classes']:
                    print(f"  - {class_name}")
            
        elif args.command == "accuracy":
            # Get accuracy
            result = client.get_accuracy()
            print(json.dumps(result, indent=2))
            
        elif args.command == "train":
            # Training
            print("Memulai training...")
            result = client.train_model(
                args.data_dir,
                args.val_split,
                args.epochs
            )
            
            print("\nTraining selesai!")
            print(f"Model path: {result['model_path']}")
            print(f"Validation accuracy: {result['val_accuracy']:.4f}")
            
        else:
            print("Command tidak valid")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
