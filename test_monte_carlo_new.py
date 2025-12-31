#!/usr/bin/env python3
"""
Monte Carlo Simulation Test - Portfolio Tester Logic
Test the new Monte Carlo simulation using the exact logic from portfolio-tester
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from modules.portfolio_simulator import PortfolioSimulator

def test_monte_carlo_simulation():
    """Test the new Monte Carlo simulation logic."""
    
    simulator = PortfolioSimulator()
    
    print("🎰 Testing Monte Carlo Simulation (Portfolio-Tester Logic)")
    print("=" * 70)
    
    # Create synthetic data for testing - multiple years of daily data
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"📅 Test Period: {start_date} to {end_date} ({len(dates)} days)")
    
    # Create mock price data for multiple assets
    np.random.seed(42)  # For reproducible results
    
    # Asset 1: US Stocks (higher return, higher volatility)
    us_returns = np.random.normal(0.10/252, 0.16/np.sqrt(252), len(dates))  # 10% annual, 16% vol
    us_prices = [100.0]
    for ret in us_returns[1:]:
        us_prices.append(us_prices[-1] * (1 + ret))
    
    # Asset 2: International Stocks (moderate return, moderate volatility)  
    intl_returns = np.random.normal(0.08/252, 0.18/np.sqrt(252), len(dates))  # 8% annual, 18% vol
    intl_prices = [100.0]
    for ret in intl_returns[1:]:
        intl_prices.append(intl_prices[-1] * (1 + ret))
    
    # Asset 3: Bonds (lower return, lower volatility)
    bond_returns = np.random.normal(0.04/252, 0.04/np.sqrt(252), len(dates))  # 4% annual, 4% vol
    bond_prices = [100.0]
    for ret in bond_returns[1:]:
        bond_prices.append(bond_prices[-1] * (1 + ret))
    
    # Mock data structure
    mock_data = {
        'US_STOCKS': pd.DataFrame({'Close': us_prices}, index=dates),
        'INTL_STOCKS': pd.DataFrame({'Close': intl_prices}, index=dates),  
        'BONDS': pd.DataFrame({'Close': bond_prices}, index=dates)
    }
    
    # Test diversified portfolio
    allocations = {
        'US_STOCKS': 60.0,    # 60% US stocks
        'INTL_STOCKS': 30.0,  # 30% International stocks
        'BONDS': 10.0         # 10% Bonds
    }
    
    initial_investment = 10000
    years_to_project = 3  # Reduced from 10 to 3 years since we only have 4 years of data
    num_simulations = 1000
    
    print(f"\n📊 Portfolio Allocation:")
    for asset, allocation in allocations.items():
        print(f"   {asset}: {allocation}%")
    
    print(f"\n🎯 Simulation Parameters:")
    print(f"   Initial Investment: ${initial_investment:,}")
    print(f"   Projection Period: {years_to_project} years")
    print(f"   Number of Simulations: {num_simulations:,}")
    
    # Test 1: No contributions
    print(f"\n📈 Test 1: Buy & Hold (No Contributions)")
    print("-" * 50)
    
    results = simulator.monte_carlo_simulation(
        allocations=allocations,
        data=mock_data,
        initial_investment=initial_investment,
        years=years_to_project,
        simulations=num_simulations,
        periodic_contribution=0,
        periodic_withdrawal=0
    )
    
    if results:
        print(f"✅ Simulation completed successfully!")
        print(f"   📊 Total Simulations: {results.get('total_simulations', 0):,}")
        print(f"   📈 Success Rate: {results.get('success_rate', 0):.1f}%")
        print(f"")
        print(f"   💰 Final Values:")
        print(f"      Best Case:    ${results.get('max', 0):,.0f}")
        print(f"      Median Case:  ${results.get('median', 0):,.0f}")  
        print(f"      Worst Case:   ${results.get('min', 0):,.0f}")
        print(f"      Mean:         ${results.get('mean', 0):,.0f}")
        print(f"")
        print(f"   📊 Distribution:")
        print(f"      95th Percentile: ${results.get('percentile_95', 0):,.0f}")
        print(f"      75th Percentile: ${results.get('percentile_75', 0):,.0f}")
        print(f"      25th Percentile: ${results.get('percentile_25', 0):,.0f}")
        print(f"      5th Percentile:  ${results.get('percentile_5', 0):,.0f}")
        
        if 'best_case' in results:
            best = results['best_case']
            print(f"")
            print(f"   🏆 Best Case Details:")
            print(f"      Total Return:      {best.get('total_return', 0):.1f}%")
            print(f"      Annualized Return: {best.get('annualized_return', 0):.1f}%")
            print(f"      Max Drawdown:      {best.get('max_drawdown', 0):.1f}%")
        
        if 'worst_case' in results:
            worst = results['worst_case']  
            print(f"")
            print(f"   📉 Worst Case Details:")
            print(f"      Total Return:      {worst.get('total_return', 0):.1f}%")
            print(f"      Annualized Return: {worst.get('annualized_return', 0):.1f}%")
            print(f"      Max Drawdown:      {worst.get('max_drawdown', 0):.1f}%")
    else:
        print("❌ Simulation failed!")
    
    # Test 2: With monthly contributions
    print(f"\n📈 Test 2: With Monthly Contributions")
    print("-" * 50)
    
    monthly_contribution = 1000
    
    results_contrib = simulator.monte_carlo_simulation(
        allocations=allocations,
        data=mock_data,
        initial_investment=initial_investment,
        years=years_to_project,
        simulations=num_simulations,
        periodic_contribution=monthly_contribution,
        contribution_frequency='monthly',
        periodic_withdrawal=0
    )
    
    if results_contrib:
        print(f"✅ Simulation with contributions completed!")
        print(f"   💰 Monthly Contributions: ${monthly_contribution:,}")
        print(f"   💰 Total Contributions: ${initial_investment + monthly_contribution * 12 * years_to_project:,}")
        print(f"   📊 Success Rate: {results_contrib.get('success_rate', 0):.1f}%")
        print(f"")
        print(f"   💰 Final Values:")
        print(f"      Best Case:    ${results_contrib.get('max', 0):,.0f}")
        print(f"      Median Case:  ${results_contrib.get('median', 0):,.0f}")
        print(f"      Worst Case:   ${results_contrib.get('min', 0):,.0f}")
        print(f"      Mean:         ${results_contrib.get('mean', 0):,.0f}")
        
        if 'median_case' in results_contrib:
            median = results_contrib['median_case']
            print(f"")
            print(f"   🎯 Median Case Details:")
            print(f"      Total Return:      {median.get('total_return', 0):.1f}%")
            print(f"      Annualized Return: {median.get('annualized_return', 0):.1f}%")
            print(f"      Max Drawdown:      {median.get('max_drawdown', 0):.1f}%")
    else:
        print("❌ Simulation with contributions failed!")
    
    # Test 3: Performance comparison
    if results and results_contrib:
        print(f"\n📊 Performance Comparison")
        print("-" * 50)
        
        # Compare median outcomes
        no_contrib_median = results.get('median', 0)
        with_contrib_median = results_contrib.get('median', 0)
        
        # Expected total with contributions (rough estimate)
        total_contributions = initial_investment + (monthly_contribution * 12 * years_to_project)
        
        print(f"   Median Final Value (No Contrib):    ${no_contrib_median:,.0f}")
        print(f"   Median Final Value (With Contrib):  ${with_contrib_median:,.0f}")
        print(f"   Total Money Invested (With Contrib): ${total_contributions:,.0f}")
        print(f"")
        print(f"   💡 Value of Dollar Cost Averaging:")
        print(f"      Additional Value: ${with_contrib_median - no_contrib_median:,.0f}")
        print(f"      Return on Additional Investment: {((with_contrib_median - no_contrib_median) / (total_contributions - initial_investment) - 1) * 100:.1f}%")
    
    print("\n" + "=" * 70)
    print("🎯 Monte Carlo Simulation Tests Completed!")

if __name__ == "__main__":
    test_monte_carlo_simulation()
