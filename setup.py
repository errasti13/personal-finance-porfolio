#!/usr/bin/env python3
"""
Setup script for Comprehensive Finance Tool
Installs required dependencies and initializes the project.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    packages = [
        "streamlit>=1.29.0",
        "pandas>=2.1.0",
        "plotly>=5.17.0", 
        "numpy>=1.26.0",
        "python-dateutil>=2.8.2",
        "yfinance>=0.2.20",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pdfplumber>=0.7.0",
        "openpyxl>=3.1.0",
        "xlsxwriter>=3.0.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def create_directories():
    """Create necessary directories."""
    dirs = ["data", "modules", "exports", "uploads"]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def main():
    print("🏦 Setting up Comprehensive Finance Tool...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\nTo run the application:")
    print("   streamlit run app.py")
    print("\nThe application will open in your default web browser.")

if __name__ == "__main__":
    main()
