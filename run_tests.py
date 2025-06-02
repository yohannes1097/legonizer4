"""
Script untuk menjalankan semua unit tests
"""

import unittest
import sys
import os
from pathlib import Path

def discover_and_run_tests():
    """
    Discover dan jalankan semua unit tests
    """
    # Set PYTHONPATH
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = discover_and_run_tests()
    sys.exit(exit_code)
