#!/usr/bin/env python3
"""
Run the Streamlit frontend
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run Streamlit app"""
    
    # Get frontend directory
    frontend_dir = Path(__file__).parent.parent / "frontend"
    app_file = frontend_dir / "app.py"
    
    if not app_file.exists():
        print(f"Error: Frontend app not found at {app_file}")
        return 1
    
    # Get configuration from environment
    port = os.getenv("STREAMLIT_SERVER_PORT", "8501")
    
    print("=" * 80)
    print("Starting SHL Assessment Recommender Frontend")
    print("=" * 80)
    print(f"Frontend URL: http://localhost:{port}")
    print(f"API URL: {os.getenv('API_BASE_URL', 'http://localhost:8000')}")
    print("=" * 80)
    print("\nMake sure the API is running:")
    print("  python scripts/run_api.py")
    print("\nPress Ctrl+C to stop the frontend")
    print("=" * 80)
    
    # Run streamlit
    try:
        subprocess.run([
            "streamlit", "run",
            str(app_file),
            "--server.port", port,
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down frontend...")
    except Exception as e:
        print(f"Error running frontend: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
