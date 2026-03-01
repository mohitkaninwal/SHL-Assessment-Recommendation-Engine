#!/usr/bin/env python3
"""
Run the React frontend (Vite)
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run React app"""
    
    # Get frontend directory
    frontend_dir = Path(__file__).parent.parent / "frontend"
    package_file = frontend_dir / "package.json"

    if not package_file.exists():
        print(f"Error: Frontend package.json not found at {package_file}")
        return 1
    
    # Get configuration from environment
    port = os.getenv("FRONTEND_PORT", "8501")
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    frontend_env = os.environ.copy()
    frontend_env["VITE_API_BASE_URL"] = api_base_url
    
    print("=" * 80)
    print("Starting SHL Assessment Recommender Frontend (React)")
    print("=" * 80)
    print(f"Frontend URL: http://localhost:{port}")
    print(f"API URL: {api_base_url}")
    print("=" * 80)
    print("\nMake sure the API is running:")
    print("  python scripts/run_api.py")
    print("\nPress Ctrl+C to stop the frontend")
    print("=" * 80)
    
    # Install dependencies if needed, then run Vite dev server.
    try:
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, env=frontend_env)
        subprocess.run([
            "npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", port
        ], cwd=frontend_dir, check=True, env=frontend_env)
    except KeyboardInterrupt:
        print("\n\nShutting down frontend...")
    except Exception as e:
        print(f"Error running frontend: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
