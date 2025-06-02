"""
Module untuk tracking dan reporting accuracy metrics
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccuracyReporter:
    """
    Kelas untuk tracking dan reporting accuracy metrics
    """
    
    def __init__(self, 
                 log_file: str = "data/metrics/accuracy_log.json",
                 report_dir: str = "data/metrics/reports"):
        """
        Inisialisasi accuracy reporter
        
        Args:
            log_file: Path ke file log accuracy
            report_dir: Direktori untuk menyimpan report
        """
        self.log_file = Path(log_file)
        self.report_dir = Path(report_dir)
        
        # Buat direktori jika belum ada
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing logs
        self.logs = self._load_logs()
        
        logger.info(f"AccuracyReporter diinisialisasi dengan log_file={log_file}")

    def _load_logs(self) -> List[Dict]:
        """
        Load logs yang sudah ada
        
        Returns:
            List logs
        """
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading logs: {e}")
                return []
        return []

    def _save_logs(self):
        """
        Simpan logs ke file
        """
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving logs: {e}")

    def log_training_accuracy(self,
                            model_path: str,
                            train_accuracy: float,
                            val_accuracy: float,
                            num_epochs: int,
                            num_classes: int,
                            training_time: float,
                            additional_metrics: Optional[Dict] = None):
        """
        Log accuracy dari training
        
        Args:
            model_path: Path ke model yang ditraining
            train_accuracy: Accuracy training
            val_accuracy: Accuracy validation
            num_epochs: Jumlah epoch
            num_classes: Jumlah kelas
            training_time: Waktu training dalam detik
            additional_metrics: Metrics tambahan
        """
        log_entry = {
            "type": "training",
            "timestamp": datetime.now().isoformat(),
            "model_path": str(model_path),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "num_epochs": num_epochs,
            "num_classes": num_classes,
            "training_time_seconds": training_time,
            "additional_metrics": additional_metrics or {}
        }
        
        self.logs.append(log_entry)
        self._save_logs()
        
        logger.info(
            f"Training accuracy logged: train={train_accuracy:.4f}, "
            f"val={val_accuracy:.4f}"
        )

    def log_prediction_accuracy(self,
                              predicted_class: str,
                              actual_class: str,
                              confidence: float,
                              similarity_score: Optional[float] = None,
                              processing_time: Optional[float] = None):
        """
        Log accuracy dari prediksi
        
        Args:
            predicted_class: Kelas yang diprediksi
            actual_class: Kelas sebenarnya
            confidence: Confidence score
            similarity_score: Similarity score dari FAISS
            processing_time: Waktu processing dalam detik
        """
        is_correct = predicted_class == actual_class
        
        log_entry = {
            "type": "prediction",
            "timestamp": datetime.now().isoformat(),
            "predicted_class": predicted_class,
            "actual_class": actual_class,
            "is_correct": is_correct,
            "confidence": confidence,
            "similarity_score": similarity_score,
            "processing_time_seconds": processing_time
        }
        
        self.logs.append(log_entry)
        self._save_logs()
        
        logger.info(
            f"Prediction logged: {predicted_class} "
            f"({'correct' if is_correct else 'incorrect'})"
        )

    def get_training_stats(self) -> Dict:
        """
        Dapatkan statistik training
        
        Returns:
            Dictionary berisi statistik training
        """
        training_logs = [log for log in self.logs if log["type"] == "training"]
        
        if not training_logs:
            return {"message": "Tidak ada data training"}
        
        # Hitung statistik
        train_accuracies = [log["train_accuracy"] for log in training_logs]
        val_accuracies = [log["val_accuracy"] for log in training_logs]
        
        stats = {
            "total_trainings": len(training_logs),
            "latest_training": training_logs[-1]["timestamp"],
            "best_train_accuracy": max(train_accuracies),
            "best_val_accuracy": max(val_accuracies),
            "avg_train_accuracy": np.mean(train_accuracies),
            "avg_val_accuracy": np.mean(val_accuracies),
            "latest_model": training_logs[-1]["model_path"]
        }
        
        return stats

    def get_prediction_stats(self, 
                           last_n_days: Optional[int] = None) -> Dict:
        """
        Dapatkan statistik prediksi
        
        Args:
            last_n_days: Hanya ambil data n hari terakhir
            
        Returns:
            Dictionary berisi statistik prediksi
        """
        prediction_logs = [log for log in self.logs if log["type"] == "prediction"]
        
        # Filter berdasarkan tanggal jika diminta
        if last_n_days:
            cutoff_date = datetime.now().timestamp() - (last_n_days * 24 * 3600)
            prediction_logs = [
                log for log in prediction_logs
                if datetime.fromisoformat(log["timestamp"]).timestamp() > cutoff_date
            ]
        
        if not prediction_logs:
            return {"message": "Tidak ada data prediksi"}
        
        # Hitung statistik
        correct_predictions = [log for log in prediction_logs if log["is_correct"]]
        total_predictions = len(prediction_logs)
        accuracy = len(correct_predictions) / total_predictions if total_predictions > 0 else 0
        
        # Statistik per kelas
        class_stats = {}
        for log in prediction_logs:
            actual_class = log["actual_class"]
            if actual_class not in class_stats:
                class_stats[actual_class] = {"total": 0, "correct": 0}
            
            class_stats[actual_class]["total"] += 1
            if log["is_correct"]:
                class_stats[actual_class]["correct"] += 1
        
        # Hitung accuracy per kelas
        for class_name, stats in class_stats.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        # Confidence statistics
        confidences = [log["confidence"] for log in prediction_logs]
        
        stats = {
            "total_predictions": total_predictions,
            "correct_predictions": len(correct_predictions),
            "overall_accuracy": accuracy,
            "avg_confidence": np.mean(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "class_statistics": class_stats,
            "period_days": last_n_days or "all_time"
        }
        
        return stats

    def generate_accuracy_report(self, 
                               output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive accuracy report
        
        Args:
            output_path: Path untuk menyimpan report
            
        Returns:
            Path ke file report yang dihasilkan
        """
        # Generate report path jika tidak diberikan
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.report_dir / f"accuracy_report_{timestamp}.html"
        
        # Dapatkan statistik
        training_stats = self.get_training_stats()
        prediction_stats = self.get_prediction_stats()
        
        # Generate HTML report
        html_content = self._generate_html_report(training_stats, prediction_stats)
        
        # Simpan report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Accuracy report disimpan ke: {output_path}")
        return str(output_path)

    def _generate_html_report(self, 
                            training_stats: Dict,
                            prediction_stats: Dict) -> str:
        """
        Generate HTML report content
        
        Args:
            training_stats: Statistik training
            prediction_stats: Statistik prediksi
            
        Returns:
            HTML content string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Legonizer4 Accuracy Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .metric-value {{ font-weight: bold; color: #2196F3; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Legonizer4 Accuracy Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Training Statistics</h2>
                {self._format_training_stats_html(training_stats)}
            </div>
            
            <div class="section">
                <h2>Prediction Statistics</h2>
                {self._format_prediction_stats_html(prediction_stats)}
            </div>
        </body>
        </html>
        """
        return html

    def _format_training_stats_html(self, stats: Dict) -> str:
        """Format training statistics untuk HTML"""
        if "message" in stats:
            return f"<p>{stats['message']}</p>"
        
        return f"""
        <div class="metric">Total Trainings: <span class="metric-value">{stats['total_trainings']}</span></div>
        <div class="metric">Latest Training: <span class="metric-value">{stats['latest_training']}</span></div>
        <div class="metric">Best Train Accuracy: <span class="metric-value">{stats['best_train_accuracy']:.4f}</span></div>
        <div class="metric">Best Val Accuracy: <span class="metric-value">{stats['best_val_accuracy']:.4f}</span></div>
        <div class="metric">Avg Train Accuracy: <span class="metric-value">{stats['avg_train_accuracy']:.4f}</span></div>
        <div class="metric">Avg Val Accuracy: <span class="metric-value">{stats['avg_val_accuracy']:.4f}</span></div>
        <div class="metric">Latest Model: <span class="metric-value">{stats['latest_model']}</span></div>
        """

    def _format_prediction_stats_html(self, stats: Dict) -> str:
        """Format prediction statistics untuk HTML"""
        if "message" in stats:
            return f"<p>{stats['message']}</p>"
        
        # Format class statistics table
        class_table = "<table><tr><th>Class</th><th>Total</th><th>Correct</th><th>Accuracy</th></tr>"
        for class_name, class_stats in stats['class_statistics'].items():
            class_table += f"""
            <tr>
                <td>{class_name}</td>
                <td>{class_stats['total']}</td>
                <td>{class_stats['correct']}</td>
                <td>{class_stats['accuracy']:.4f}</td>
            </tr>
            """
        class_table += "</table>"
        
        return f"""
        <div class="metric">Total Predictions: <span class="metric-value">{stats['total_predictions']}</span></div>
        <div class="metric">Correct Predictions: <span class="metric-value">{stats['correct_predictions']}</span></div>
        <div class="metric">Overall Accuracy: <span class="metric-value">{stats['overall_accuracy']:.4f}</span></div>
        <div class="metric">Average Confidence: <span class="metric-value">{stats['avg_confidence']:.4f}</span></div>
        <div class="metric">Min Confidence: <span class="metric-value">{stats['min_confidence']:.4f}</span></div>
        <div class="metric">Max Confidence: <span class="metric-value">{stats['max_confidence']:.4f}</span></div>
        <h3>Per-Class Statistics</h3>
        {class_table}
        """

    def plot_accuracy_trends(self, 
                           output_path: Optional[str] = None) -> str:
        """
        Plot accuracy trends over time
        
        Args:
            output_path: Path untuk menyimpan plot
            
        Returns:
            Path ke file plot yang dihasilkan
        """
        # Generate output path jika tidak diberikan
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.report_dir / f"accuracy_trends_{timestamp}.png"
        
        # Dapatkan data training
        training_logs = [log for log in self.logs if log["type"] == "training"]
        
        if not training_logs:
            logger.warning("Tidak ada data training untuk diplot")
            return ""
        
        # Extract data
        timestamps = [datetime.fromisoformat(log["timestamp"]) for log in training_logs]
        train_accs = [log["train_accuracy"] for log in training_logs]
        val_accs = [log["val_accuracy"] for log in training_logs]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, train_accs, label='Training Accuracy', marker='o')
        plt.plot(timestamps, val_accs, label='Validation Accuracy', marker='s')
        plt.xlabel('Date')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Trends')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Simpan plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Accuracy trends plot disimpan ke: {output_path}")
        return str(output_path)
