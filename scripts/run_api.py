#!/usr/bin/env python3
"""
Run the FastAPI server
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run
from src.api.main import main

if __name__ == "__main__":
    main()
