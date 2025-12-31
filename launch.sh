#!/bin/bash

# Activate virtual environment and launch the Comprehensive Finance Tool

echo "🏦 Launching Comprehensive Finance Tool..."

# Change to the project directory
cd "$(dirname "$0")"

# Activate the virtual environment
source finance-venv/bin/activate

# Check if streamlit is available
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Please run setup.py first."
    exit 1
fi

echo "✅ Virtual environment activated"
echo "🚀 Starting Streamlit application..."
echo ""
echo "The application will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the application."
echo ""

# Launch the Streamlit app
streamlit run app.py

echo "👋 Application stopped."
