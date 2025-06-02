"""
Script backup untuk data dan model Legonizer4
"""

import argparse
import shutil
import tarfile
import logging
from datetime import datetime
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Legonizer4Backup:
    """
    Backup manager untuk Legonizer4
    """
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Inisialisasi backup manager
        
        Args:
            backup_dir: Direktori untuk menyimpan backup
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Timestamp untuk backup
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def backup_data(self, data_dir: str = "data") -> str:
        """
        Backup data directory
        
        Args:
            data_dir: Path ke data directory
            
        Returns:
            Path ke backup file
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory tidak ditemukan: {data_dir}")
        
        backup_name = f"data_backup_{self.timestamp}.tar.gz"
        backup_path = self.backup_dir / backup_name
        
        logger.info(f"Creating data backup: {backup_path}")
        
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(data_path, arcname="data")
        
        logger.info(f"Data backup completed: {backup_path}")
        return str(backup_path)
    
    def backup_models(self, models_dir: str = "data/models") -> str:
        """
        Backup models directory
        
        Args:
            models_dir: Path ke models directory
            
        Returns:
            Path ke backup file
        """
        models_path = Path(models_dir)
        if not models_path.exists():
            raise ValueError(f"Models directory tidak ditemukan: {models_dir}")
        
        backup_name = f"models_backup_{self.timestamp}.tar.gz"
        backup_path = self.backup_dir / backup_name
        
        logger.info(f"Creating models backup: {backup_path}")
        
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(models_path, arcname="models")
        
        logger.info(f"Models backup completed: {backup_path}")
        return str(backup_path)
    
    def backup_config(self, config_file: str = "config.json") -> str:
        """
        Backup configuration file
        
        Args:
            config_file: Path ke config file
            
        Returns:
            Path ke backup file
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise ValueError(f"Config file tidak ditemukan: {config_file}")
        
        backup_name = f"config_backup_{self.timestamp}.json"
        backup_path = self.backup_dir / backup_name
        
        logger.info(f"Creating config backup: {backup_path}")
        shutil.copy2(config_path, backup_path)
        
        logger.info(f"Config backup completed: {backup_path}")
        return str(backup_path)
    
    def full_backup(self) -> Dict[str, str]:
        """
        Backup lengkap semua komponen
        
        Returns:
            Dictionary berisi path backup files
        """
        backup_files = {}
        
        try:
            # Backup data
            backup_files["data"] = self.backup_data()
        except Exception as e:
            logger.warning(f"Data backup failed: {e}")
        
        try:
            # Backup models
            backup_files["models"] = self.backup_models()
        except Exception as e:
            logger.warning(f"Models backup failed: {e}")
        
        try:
            # Backup config
            backup_files["config"] = self.backup_config()
        except Exception as e:
            logger.warning(f"Config backup failed: {e}")
        
        # Create backup manifest
        manifest = {
            "timestamp": self.timestamp,
            "backup_files": backup_files,
            "created_at": datetime.now().isoformat()
        }
        
        manifest_path = self.backup_dir / f"backup_manifest_{self.timestamp}.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        backup_files["manifest"] = str(manifest_path)
        
        return backup_files
    
    def restore_from_backup(self, backup_file: str, target_dir: str = "."):
        """
        Restore dari backup file
        
        Args:
            backup_file: Path ke backup file
            target_dir: Target directory untuk restore
        """
        backup_path = Path(backup_file)
        if not backup_path.exists():
            raise ValueError(f"Backup file tidak ditemukan: {backup_file}")
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Restoring from backup: {backup_file}")
        
        if backup_file.endswith('.tar.gz'):
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(target_path)
        elif backup_file.endswith('.json'):
            shutil.copy2(backup_path, target_path)
        else:
            raise ValueError(f"Unsupported backup file format: {backup_file}")
        
        logger.info(f"Restore completed to: {target_path}")
    
    def list_backups(self) -> List[Dict]:
        """
        List semua backup yang tersedia
        
        Returns:
            List backup information
        """
        backups = []
        
        # Cari manifest files
        for manifest_file in self.backup_dir.glob("backup_manifest_*.json"):
            try:
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                backups.append(manifest)
            except Exception as e:
                logger.warning(f"Error reading manifest {manifest_file}: {e}")
        
        # Sort by timestamp
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return backups
    
    def cleanup_old_backups(self, keep_count: int = 5):
        """
        Cleanup backup lama, hanya simpan yang terbaru
        
        Args:
            keep_count: Jumlah backup yang akan disimpan
        """
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            logger.info(f"Only {len(backups)} backups found, no cleanup needed")
            return
        
        # Hapus backup lama
        for backup in backups[keep_count:]:
            timestamp = backup['timestamp']
            logger.info(f"Cleaning up backup: {timestamp}")
            
            # Hapus backup files
            for backup_type, backup_file in backup['backup_files'].items():
                backup_path = Path(backup_file)
                if backup_path.exists():
                    backup_path.unlink()
                    logger.info(f"Deleted: {backup_file}")
        
        logger.info(f"Cleanup completed, kept {keep_count} most recent backups")

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Backup dan restore untuk Legonizer4"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create backup")
    backup_parser.add_argument(
        "--type",
        choices=["data", "models", "config", "full"],
        default="full",
        help="Type of backup"
    )
    backup_parser.add_argument(
        "--backup-dir",
        default="backups",
        help="Backup directory"
    )
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument(
        "backup_file",
        help="Path to backup file"
    )
    restore_parser.add_argument(
        "--target-dir",
        default=".",
        help="Target directory for restore"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available backups")
    list_parser.add_argument(
        "--backup-dir",
        default="backups",
        help="Backup directory"
    )
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup old backups")
    cleanup_parser.add_argument(
        "--keep",
        type=int,
        default=5,
        help="Number of backups to keep"
    )
    cleanup_parser.add_argument(
        "--backup-dir",
        default="backups",
        help="Backup directory"
    )
    
    return parser.parse_args()

def main():
    """
    Fungsi utama
    """
    args = parse_arguments()
    
    if not args.command:
        print("Please specify a command. Use --help for more information.")
        return
    
    try:
        if args.command == "backup":
            backup_manager = Legonizer4Backup(args.backup_dir)
            
            if args.type == "data":
                backup_file = backup_manager.backup_data()
                print(f"Data backup created: {backup_file}")
            elif args.type == "models":
                backup_file = backup_manager.backup_models()
                print(f"Models backup created: {backup_file}")
            elif args.type == "config":
                backup_file = backup_manager.backup_config()
                print(f"Config backup created: {backup_file}")
            elif args.type == "full":
                backup_files = backup_manager.full_backup()
                print("Full backup created:")
                for backup_type, backup_file in backup_files.items():
                    print(f"  {backup_type}: {backup_file}")
        
        elif args.command == "restore":
            backup_manager = Legonizer4Backup()
            backup_manager.restore_from_backup(args.backup_file, args.target_dir)
            print(f"Restore completed to: {args.target_dir}")
        
        elif args.command == "list":
            backup_manager = Legonizer4Backup(args.backup_dir)
            backups = backup_manager.list_backups()
            
            if not backups:
                print("No backups found")
            else:
                print("Available backups:")
                for backup in backups:
                    print(f"\nTimestamp: {backup['timestamp']}")
                    print(f"Created: {backup['created_at']}")
                    print("Files:")
                    for backup_type, backup_file in backup['backup_files'].items():
                        print(f"  {backup_type}: {backup_file}")
        
        elif args.command == "cleanup":
            backup_manager = Legonizer4Backup(args.backup_dir)
            backup_manager.cleanup_old_backups(args.keep)
            print(f"Cleanup completed, kept {args.keep} most recent backups")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
