#!/usr/bin/env python3
"""
Convenience script to run the data scraping pipeline
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.main import main

if __name__ == "__main__":
    main()

