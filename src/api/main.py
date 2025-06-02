"""
Module untuk FastAPI endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import logging
import json
import shutil
import tempfile
from typing import List, Optional
import torch
from datetime import datetime

from ..models.efficient_net import EfficientNetFeatureExtractor
from ..models.trainer import LegoTrainer
from ..preprocessing.processor import ImageProcessor
from ..search.faiss_index import LegoSearchIndex

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi FastAPI
app = FastAPI(
    title="Legonizer4 API",
    description="API untuk identifikasi LEGO brick menggunakan machine learning",
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables untuk model dan index
model = None
search_index = None
processor = None
class_mapping = None

def load_model_and_index():
    """
    Load model dan search index
    """
    global model, search_index, processor, class_mapping
    
    try:
        # Load model terakhir dari direktori models
        models_dir = Path("data/models")
        model_files = list(models_dir.glob("*.pth"))
        if not model_files:
            raise FileNotFoundError("Tidak ada model yang ditemukan")
        
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        mapping_file = latest_model.with_suffix('.json')
        
        # Load class mapping
        with open(mapping_file) as f:
            class_mapping = json.load(f)
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = EfficientNetFeatureExtractor.load_model(
            str(latest_model),
            map_location=device
        )
        
        # Load search index
        index_files = list(models_dir.glob("*.index"))
        if index_files:
            latest_index = max(index_files, key=lambda x: x.stat().st_mtime)
            search_index = LegoSearchIndex.load_index(str(latest_index))
        
        # Inisialisasi image processor
        processor = ImageProcessor()
        
        logger.info("Model dan index berhasil diload")
        
    except Exception as e:
        logger.error(f"Error saat loading model dan index: {e}")
        raise HTTPException(
            status_code=500,
            detail="Gagal menginisialisasi model dan index"
        )

@app.on_event("startup")
async def startup_event():
    """
    Load model dan index saat startup
    """
    load_model_and_index()

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Selamat datang di Legonizer4 API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "index_loaded": search_index is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/identify")
async def identify_lego(
    file: UploadFile = File(...),
    top_n: int = Form(5)
):
    """
    Identifikasi LEGO dari gambar yang diupload
    """
    if not model or not processor:
        raise HTTPException(
            status_code=500,
            detail="Model belum diinisialisasi"
        )
    
    try:
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Deteksi dan preprocessing
        processed_images = processor.process_single_image(temp_path)[0]
        
        if not processed_images:
            raise HTTPException(
                status_code=400,
                detail="Tidak ada objek LEGO yang terdeteksi"
            )
        
        # Prediksi dengan model
        predictions = []
        for img in processed_images:
            # Convert ke tensor
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            # Prediksi
            with torch.no_grad():
                logits, features = model(img_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Dapatkan top-n predictions
            top_probs, top_indices = torch.topk(probs, k=top_n)
            
            # Convert ke class names
            idx_to_class = {v: k for k, v in class_mapping.items()}
            class_predictions = [
                {
                    "model_id": idx_to_class[idx.item()],
                    "confidence": prob.item()
                }
                for idx, prob in zip(top_indices[0], top_probs[0])
            ]
            
            # Similarity search jika index tersedia
            if search_index:
                similar_items = search_index.search(
                    features.cpu().numpy(),
                    k=top_n
                )
                
                # Tambahkan similarity scores
                for pred, (meta, dist) in zip(class_predictions, similar_items):
                    pred["similarity"] = 1.0 / (1.0 + dist)  # Convert distance ke similarity
                    pred.update(meta)
            
            predictions.append(class_predictions)
        
        # Cleanup
        Path(temp_path).unlink()
        
        return JSONResponse({
            "predictions": predictions,
            "num_objects_detected": len(processed_images),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error saat identifikasi: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/train")
async def train_model(
    data_dir: str = Form(...),
    val_split: float = Form(0.2),
    num_epochs: int = Form(50)
):
    """
    Training ulang model
    """
    if not model:
        raise HTTPException(
            status_code=500,
            detail="Model belum diinisialisasi"
        )
    
    try:
        # Setup trainer
        trainer = LegoTrainer(
            model,
            num_epochs=num_epochs
        )
        
        # Load dan split data
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Directory tidak ditemukan: {data_dir}")
        
        # TODO: Implementasi loading dan splitting data
        
        # Training
        results = trainer.train(
            train_data=([], []),  # TODO: Implement
            val_data=([], []),    # TODO: Implement
            model_dir="data/models",
            class_to_idx=class_mapping
        )
        
        return JSONResponse({
            "status": "success",
            "model_path": results["best_model_path"],
            "val_accuracy": results["best_val_accuracy"],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error saat training: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/models")
async def list_models():
    """
    List semua model yang tersedia
    """
    try:
        models_dir = Path("data/models")
        model_files = list(models_dir.glob("*.pth"))
        
        models = []
        for model_file in model_files:
            mapping_file = model_file.with_suffix('.json')
            if mapping_file.exists():
                with open(mapping_file) as f:
                    class_mapping = json.load(f)
                
                models.append({
                    "path": str(model_file),
                    "created": datetime.fromtimestamp(
                        model_file.stat().st_mtime
                    ).isoformat(),
                    "num_classes": len(class_mapping),
                    "classes": list(class_mapping.keys())
                })
        
        return JSONResponse({
            "models": models,
            "total": len(models)
        })
        
    except Exception as e:
        logger.error(f"Error saat listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/accuracy")
async def get_accuracy():
    """
    Dapatkan report akurasi model terkini
    """
    if not model:
        raise HTTPException(
            status_code=500,
            detail="Model belum diinisialisasi"
        )
    
    try:
        # TODO: Implement accuracy reporting
        return JSONResponse({
            "model_accuracy": 0.0,  # TODO: Calculate
            "total_predictions": 0,  # TODO: Calculate
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error saat mengambil accuracy: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
