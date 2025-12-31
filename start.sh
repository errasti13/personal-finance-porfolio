#!/bin/bash

# Comprehensive Finance Tool - Quick Start Script
# This script sets up and runs the finance tool

echo "🏦 Comprehensive Finance Tool - Quick Start"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "finance-venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv finance-venv
    chmod +x finance-venv/bin/activate
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source finance-venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p modules
mkdir -p exports
mkdir -p uploads

echo "✅ Directories created"

# Check if Streamlit is working
echo "🔍 Testing Streamlit installation..."
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Streamlit is working"
else
    echo "❌ Streamlit installation failed"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application, run:"
echo "   streamlit run app.py"
echo ""
echo "The application will open in your web browser at http://localhost:8501"
echo ""

# Ask if user wants to start the app immediately
read -p "Would you like to start the application now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting Comprehensive Finance Tool..."
    streamlit run app.py
fi
