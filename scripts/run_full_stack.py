#!/usr/bin/env python3
"""
Run both API and React frontend together
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

# Process tracking
processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nShutting down...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    sys.exit(0)

def main():
    """Run full stack"""
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 80)
    print("Starting SHL Assessment Recommendation System - Full Stack")
    print("=" * 80)
    
    # Get configuration
    api_port = os.getenv("API_PORT", "8000")
    frontend_port = os.getenv("FRONTEND_PORT", "8501")
    api_base_url = os.getenv("API_BASE_URL", f"http://localhost:{api_port}")
    
    print(f"\nAPI URL: http://localhost:{api_port}")
    print(f"Frontend URL: http://localhost:{frontend_port}")
    print(f"API Docs: http://localhost:{api_port}/docs")
    print("\n" + "=" * 80)
    
    # Start API
    print("\n[1/2] Starting API server...")
    api_process = subprocess.Popen(
        [sys.executable, "scripts/run_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    processes.append(api_process)
    
    # Wait for API to start
    print("Waiting for API to start...")
    time.sleep(5)
    
    # Check if API is running
    try:
        import requests
        response = requests.get(f"http://localhost:{api_port}/health", timeout=5)
        if response.status_code == 200:
            print("✓ API is running")
        else:
            print("⚠ API may not be fully ready")
    except:
        print("⚠ Could not verify API status")
    
    # Start Frontend
    print("\n[2/2] Starting frontend...")
    frontend_dir = Path(__file__).parent.parent / "frontend"
    frontend_env = os.environ.copy()
    frontend_env["VITE_API_BASE_URL"] = api_base_url

    # Ensure frontend dependencies are installed
    try:
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, env=frontend_env)
    except Exception:
        print("⚠ Could not install frontend dependencies")

    frontend_process = subprocess.Popen(
        [
            "npm", "run", "dev", "--",
            "--host", "0.0.0.0",
            "--port", frontend_port
        ],
        cwd=str(frontend_dir),
        env=frontend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    processes.append(frontend_process)
    
    print("✓ Frontend is starting...")
    
    print("\n" + "=" * 80)
    print("Full stack is running!")
    print("=" * 80)
    print(f"\n🌐 Open your browser and go to: http://localhost:{frontend_port}")
    print("\nPress Ctrl+C to stop all services")
    print("=" * 80)
    
    # Keep running and show logs
    try:
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if api_process.poll() is not None:
                print("\n⚠ API process stopped unexpectedly")
                break
            
            if frontend_process.poll() is not None:
                print("\n⚠ Frontend process stopped unexpectedly")
                break
    
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    signal_handler(None, None)


if __name__ == "__main__":
    main()
