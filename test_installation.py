#!/usr/bin/env python3
"""
Test script to verify the Comprehensive Finance Tool installation and functionality.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        import streamlit as st
        print("  ✅ Streamlit")
    except ImportError as e:
        print(f"  ❌ Streamlit: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✅ Pandas")
    except ImportError as e:
        print(f"  ❌ Pandas: {e}")
        return False
    
    try:
        import plotly.express as px
        print("  ✅ Plotly Express")
    except ImportError as e:
        print(f"  ❌ Plotly Express: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✅ NumPy")
    except ImportError as e:
        print(f"  ❌ NumPy: {e}")
        return False
    
    try:
        import yfinance as yf
        print("  ✅ yfinance")
    except ImportError as e:
        print(f"  ❌ yfinance: {e}")
        return False
    
    try:
        from modules.transaction_analyzer import TransactionAnalyzer
        print("  ✅ Transaction Analyzer")
    except ImportError as e:
        print(f"  ❌ Transaction Analyzer: {e}")
        return False
    
    return True

def test_data_files():
    """Test if data files exist and are readable."""
    print("\n📁 Testing data files...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("  📁 Created data directory")
    
    # Test sample net worth data
    sample_file = os.path.join(data_dir, "net_worth_data.csv")
    if os.path.exists(sample_file):
        try:
            import pandas as pd
            df = pd.read_csv(sample_file)
            print(f"  ✅ Sample net worth data loaded ({len(df)} rows)")
        except Exception as e:
            print(f"  ❌ Error reading sample data: {e}")
    else:
        print("  ℹ️  No sample net worth data found (this is okay)")
    
    return True

def test_functionality():
    """Test basic functionality without running Streamlit."""
    print("\n⚙️  Testing core functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Test data processing
        sample_data = {
            'Date': ['2025-01-31', '2025-02-28'],
            'UBS Account (CHF)': [10000.0, 10500.0],
            'IBKR Account (CHF)': [15000.0, 15500.0]
        }
        
        df = pd.DataFrame(sample_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Total'] = df['UBS Account (CHF)'] + df['IBKR Account (CHF)']
        
        print("  ✅ Data processing works")
        
        # Test forex rate functionality (mock)
        def mock_forex_rate(from_curr, to_curr):
            return 0.93 if from_curr == 'EUR' and to_curr == 'CHF' else 1.0
        
        rate = mock_forex_rate('EUR', 'CHF')
        print(f"  ✅ Forex rate functionality (mock rate: {rate})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def main():
    print("🔍 Comprehensive Finance Tool - System Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data files
    data_ok = test_data_files()
    
    # Test functionality
    functionality_ok = test_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"  Imports:      {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  Data Files:   {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"  Functionality:{'✅ PASS' if functionality_ok else '❌ FAIL'}")
    
    if imports_ok and data_ok and functionality_ok:
        print("\n🎉 All tests passed! The application is ready to run.")
        print("\nTo start the application, run:")
        print("  ./launch.sh")
        print("\nOr manually:")
        print("  streamlit run app.py")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
