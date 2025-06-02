"""
Script monitoring untuk Legonizer4
"""

import argparse
import time
import requests
import json
import logging
from datetime import datetime
from pathlib import Path
import psutil
import docker
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Legonizer4Monitor:
    """
    Monitor untuk sistem Legonizer4
    """
    
    def __init__(self, 
                 api_url: str = "http://localhost:8000",
                 container_name: str = "legonizer4-api",
                 log_file: str = "monitoring.log"):
        """
        Inisialisasi monitor
        
        Args:
            api_url: URL API Legonizer4
            container_name: Nama Docker container
            log_file: File untuk menyimpan log monitoring
        """
        self.api_url = api_url.rstrip('/')
        self.container_name = container_name
        self.log_file = log_file
        
        # Setup Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker client tidak tersedia: {e}")
            self.docker_client = None
    
    def check_api_health(self) -> Dict:
        """
        Check kesehatan API
        
        Returns:
            Dictionary berisi status API
        """
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds(),
                    "data": response.json()
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_container_status(self) -> Dict:
        """
        Check status Docker container
        
        Returns:
            Dictionary berisi status container
        """
        if not self.docker_client:
            return {"status": "docker_unavailable"}
        
        try:
            container = self.docker_client.containers.get(self.container_name)
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            return {
                "status": container.status,
                "cpu_percent": cpu_percent,
                "memory_usage_mb": memory_usage / (1024 * 1024),
                "memory_percent": memory_percent,
                "created": container.attrs['Created'],
                "started": container.attrs['State']['StartedAt']
            }
            
        except docker.errors.NotFound:
            return {"status": "not_found"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_system_resources(self) -> Dict:
        """
        Check system resources
        
        Returns:
            Dictionary berisi status system resources
        """
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    
    def get_api_metrics(self) -> Dict:
        """
        Dapatkan metrics dari API
        
        Returns:
            Dictionary berisi API metrics
        """
        try:
            response = requests.get(
                f"{self.api_url}/accuracy",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def run_health_check(self) -> Dict:
        """
        Jalankan comprehensive health check
        
        Returns:
            Dictionary berisi semua status checks
        """
        timestamp = datetime.now().isoformat()
        
        health_report = {
            "timestamp": timestamp,
            "api_health": self.check_api_health(),
            "container_status": self.check_container_status(),
            "system_resources": self.check_system_resources(),
            "api_metrics": self.get_api_metrics()
        }
        
        return health_report
    
    def log_health_report(self, report: Dict):
        """
        Log health report ke file
        
        Args:
            report: Health report dictionary
        """
        log_entry = {
            "timestamp": report["timestamp"],
            "api_status": report["api_health"]["status"],
            "container_status": report["container_status"].get("status", "unknown"),
            "cpu_percent": report["system_resources"]["cpu_percent"],
            "memory_percent": report["system_resources"]["memory_percent"]
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def print_health_report(self, report: Dict):
        """
        Print health report ke console
        
        Args:
            report: Health report dictionary
        """
        print(f"\n🔍 Legonizer4 Health Check - {report['timestamp']}")
        print("=" * 60)
        
        # API Health
        api_health = report["api_health"]
        status_emoji = "✅" if api_health["status"] == "healthy" else "❌"
        print(f"{status_emoji} API Status: {api_health['status']}")
        if "response_time" in api_health:
            print(f"   Response Time: {api_health['response_time']:.3f}s")
        
        # Container Status
        container_status = report["container_status"]
        if "status" in container_status:
            status_emoji = "✅" if container_status["status"] == "running" else "❌"
            print(f"{status_emoji} Container: {container_status['status']}")
            
            if "cpu_percent" in container_status:
                print(f"   CPU: {container_status['cpu_percent']:.1f}%")
                print(f"   Memory: {container_status['memory_usage_mb']:.1f}MB "
                      f"({container_status['memory_percent']:.1f}%)")
        
        # System Resources
        sys_resources = report["system_resources"]
        print(f"🖥️  System CPU: {sys_resources['cpu_percent']:.1f}%")
        print(f"🖥️  System Memory: {sys_resources['memory_percent']:.1f}%")
        print(f"💾 Disk Usage: {sys_resources['disk_percent']:.1f}%")
        
        # API Metrics
        api_metrics = report["api_metrics"]
        if "error" not in api_metrics:
            print("📊 API Metrics: Available")
        else:
            print(f"📊 API Metrics: {api_metrics['error']}")
    
    def continuous_monitoring(self, interval: int = 60):
        """
        Jalankan monitoring secara kontinyu
        
        Args:
            interval: Interval monitoring dalam detik
        """
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                report = self.run_health_check()
                self.print_health_report(report)
                self.log_health_report(report)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Monitor Legonizer4 system"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="URL API Legonizer4"
    )
    
    parser.add_argument(
        "--container-name",
        type=str,
        default="legonizer4-api",
        help="Nama Docker container"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval monitoring dalam detik"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="monitoring.log",
        help="File untuk menyimpan log monitoring"
    )
    
    parser.add_argument(
        "--once",
        action="store_true",
        help="Jalankan health check sekali saja"
    )
    
    return parser.parse_args()

def main():
    """
    Fungsi utama
    """
    args = parse_arguments()
    
    # Inisialisasi monitor
    monitor = Legonizer4Monitor(
        api_url=args.api_url,
        container_name=args.container_name,
        log_file=args.log_file
    )
    
    if args.once:
        # Single health check
        report = monitor.run_health_check()
        monitor.print_health_report(report)
        monitor.log_health_report(report)
    else:
        # Continuous monitoring
        monitor.continuous_monitoring(args.interval)

if __name__ == "__main__":
    main()
