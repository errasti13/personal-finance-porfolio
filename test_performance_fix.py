#!/usr/bin/env python3
"""
Performance Calculation Test
Test the corrected portfolio performance calculation logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from modules.portfolio_simulator import PortfolioSimulator

def test_performance_calculations():
    """Test basic performance calculations with known scenarios."""
    
    simulator = PortfolioSimulator()
    
    print("🧪 Testing Performance Calculations...")
    print("=" * 60)
    
    # Test 1: Simple scenario with no contributions
    print("\n📊 Test 1: Simple Buy & Hold (No Contributions)")
    print("-" * 50)
    
    # Create synthetic data for a simple test
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create mock price data: 10% annual growth, some volatility
    np.random.seed(42)  # For reproducible results
    daily_returns = np.random.normal(0.1/252, 0.15/np.sqrt(252), len(dates))  # 10% annual, 15% vol
    prices = [100.0]  # Start at $100
    
    for ret in daily_returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Mock data structure
    mock_data = {
        'TEST': pd.DataFrame({
            'Close': prices
        }, index=dates)
    }
    
    # Test simple allocation (100% in TEST)
    allocations = {'TEST': 100.0}
    initial_investment = 10000
    
    # Calculate portfolio returns
    results = simulator.calculate_portfolio_returns(
        allocations=allocations,
        data=mock_data,
        initial_investment=initial_investment,
        rebalance_frequency='none',
        periodic_contribution=0,
        periodic_withdrawal=0
    )
    
    if not results.empty:
        metrics = simulator.calculate_metrics(results)
        
        print(f"📈 Initial Investment: ${initial_investment:,.2f}")
        print(f"📈 Final Value: ${metrics.get('Final Value', 0):,.2f}")
        print(f"📈 Total Return: {metrics.get('Total Return (%)', 0):.2f}%")
        print(f"📈 Time-Weighted Return: {metrics.get('Time-Weighted Return (%)', 0):.2f}%")
        print(f"📈 Annualized Return: {metrics.get('Annualized Return (%)', 0):.2f}%")
        print(f"📈 Volatility: {metrics.get('Volatility (%)', 0):.2f}%")
        print(f"📈 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
        print(f"📈 Max Drawdown: {metrics.get('Maximum Drawdown (%)', 0):.2f}%")
        
        # Validation: Check if returns make sense
        expected_return = (prices[-1] / prices[0] - 1) * 100
        actual_return = metrics.get('Time-Weighted Return (%)', 0)
        print(f"\n✅ Validation:")
        print(f"   Expected Return: {expected_return:.2f}%")
        print(f"   Calculated Return: {actual_return:.2f}%")
        print(f"   Difference: {abs(expected_return - actual_return):.4f}%")
        
        if abs(expected_return - actual_return) < 0.01:
            print("   ✅ PASS: Returns match expected values")
        else:
            print("   ❌ FAIL: Returns don't match expected values")
    
    # Test 2: With monthly contributions
    print("\n📊 Test 2: With Monthly Contributions")
    print("-" * 50)
    
    results_with_contributions = simulator.calculate_portfolio_returns(
        allocations=allocations,
        data=mock_data,
        initial_investment=initial_investment,
        rebalance_frequency='none',
        periodic_contribution=1000,  # $1000/month
        contribution_frequency='monthly',
        periodic_withdrawal=0
    )
    
    if not results_with_contributions.empty:
        metrics_contrib = simulator.calculate_metrics(results_with_contributions)
        
        print(f"📈 Initial Investment: ${initial_investment:,.2f}")
        print(f"📈 Monthly Contributions: $1,000")
        print(f"📈 Total Contributions: ${metrics_contrib.get('Total Contributions', 0):,.2f}")
        print(f"📈 Final Value: ${metrics_contrib.get('Final Value', 0):,.2f}")
        print(f"📈 Money-Weighted Return: {metrics_contrib.get('Total Return (%)', 0):.2f}%")
        print(f"📈 Time-Weighted Return: {metrics_contrib.get('Time-Weighted Return (%)', 0):.2f}%")
        print(f"📈 Annualized Return: {metrics_contrib.get('Annualized Return (%)', 0):.2f}%")
        
        # Validation: Time-weighted return should be similar to buy & hold
        tw_return_simple = metrics.get('Time-Weighted Return (%)', 0)
        tw_return_contrib = metrics_contrib.get('Time-Weighted Return (%)', 0)
        print(f"\n✅ Validation:")
        print(f"   Time-Weighted (No Contrib): {tw_return_simple:.2f}%")
        print(f"   Time-Weighted (With Contrib): {tw_return_contrib:.2f}%")
        print(f"   Difference: {abs(tw_return_simple - tw_return_contrib):.4f}%")
        
        if abs(tw_return_simple - tw_return_contrib) < 1.0:  # Within 1%
            print("   ✅ PASS: Time-weighted returns are consistent")
        else:
            print("   ❌ FAIL: Time-weighted returns vary too much")
    
    print("\n" + "=" * 60)
    print("🎯 Performance Calculation Tests Completed!")

if __name__ == "__main__":
    test_performance_calculations()
