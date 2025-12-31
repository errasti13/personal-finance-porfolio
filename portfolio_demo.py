#!/usr/bin/env python3
"""
Portfolio Simulator Demo Script

This script demonstrates how to use the PortfolioSimulator class
programmatically for testing and automation.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from portfolio_simulator import PortfolioSimulator

def demo_portfolio_simulation():
    """Demonstrate portfolio simulation functionality."""
    print("🚀 Portfolio Simulator Demo")
    print("="*50)
    
    # Initialize the simulator
    simulator = PortfolioSimulator()
    
    # Define a sample portfolio
    allocations = {
        '^GSPC': 60,    # S&P 500 - 60%
        'URTH': 30,     # MSCI World - 30%
        'GC=F': 10      # Gold - 10%
    }
    
    print("📊 Portfolio Allocation:")
    for ticker, allocation in allocations.items():
        asset_name = next((name for name, t in simulator.AVAILABLE_ASSETS.items() if t == ticker), ticker)
        print(f"  • {asset_name}: {allocation}%")
    
    # Set date range (last 5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print(f"\n📅 Testing Period: {start_date} to {end_date}")
    print("\n⏳ Fetching historical data...")
    
    try:
        # Fetch historical data
        tickers = list(allocations.keys())
        historical_data = simulator.get_historical_data(tickers, start_date, end_date)
        
        if not all(ticker in historical_data and not historical_data[ticker].empty for ticker in tickers):
            print("❌ Error: Could not fetch complete data for all assets")
            return
        
        print("✅ Data fetched successfully!")
        
        # Run backtest
        print("\n🔄 Running backtest...")
        results = simulator.calculate_portfolio_returns(
            allocations, 
            historical_data, 
            initial_investment=10000,
            rebalance_frequency='monthly',
            periodic_contribution=500,  # $500 monthly contributions
            contribution_frequency='monthly'
        )
        
        print("💰 Testing with $500 monthly contributions...")
        
        if results.empty:
            print("❌ Error: Backtest failed")
            return
        
        # Calculate metrics
        metrics = simulator.calculate_metrics(results)
        
        print("✅ Backtest completed!")
        print("\n📈 Portfolio Performance:")
        print("-" * 40)
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'Return' in metric or 'Volatility' in metric or 'Drawdown' in metric or 'Day' in metric or 'Rate' in metric:
                    print(f"  {metric}: {value}%")
                else:
                    print(f"  {metric}: {value}")
            else:
                print(f"  {metric}: {value}")
        
        # Show some sample data points
        print(f"\n📊 Sample Data Points:")
        print("-" * 40)
        sample_dates = results['Date'].iloc[::len(results)//5]  # Show 5 data points
        for date in sample_dates:
            idx = results[results['Date'] == date].index[0]
            value = results.loc[idx, 'Portfolio_Value']
            cumulative_return = results.loc[idx, 'Cumulative_Return']
            print(f"  {date.strftime('%Y-%m-%d')}: ${value:,.2f} (Return: {cumulative_return:.1f}%)")
        
        # Run correlation analysis
        print(f"\n🔗 Asset Correlation Analysis:")
        print("-" * 40)
        correlation_matrix = simulator.get_asset_correlation(tickers, historical_data)
        
        if not correlation_matrix.empty:
            for i, ticker1 in enumerate(correlation_matrix.columns):
                for ticker2 in correlation_matrix.columns[i+1:]:
                    corr_value = correlation_matrix.loc[ticker1, ticker2]
                    name1 = next((name for name, t in simulator.AVAILABLE_ASSETS.items() if t == ticker1), ticker1)
                    name2 = next((name for name, t in simulator.AVAILABLE_ASSETS.items() if t == ticker2), ticker2)
                    print(f"  {name1} ↔ {name2}: {corr_value:.3f}")
        
        # Portfolio optimization suggestion
        print(f"\n⚡ Portfolio Optimization:")
        print("-" * 40)
        optimized_allocations = simulator.optimize_portfolio(tickers, historical_data)
        
        if optimized_allocations:
            print("  Suggested allocations (equal risk contribution):")
            for ticker, allocation in optimized_allocations.items():
                asset_name = next((name for name, t in simulator.AVAILABLE_ASSETS.items() if t == ticker), ticker)
                current = allocations[ticker]
                difference = allocation - current
                change_icon = "📈" if difference > 0 else "📉" if difference < 0 else "➡️"
                print(f"    {change_icon} {asset_name}: {allocation:.1f}% (current: {current}%, change: {difference:+.1f}%)")
        
        print(f"\n✨ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()

def list_available_assets():
    """List all available assets for portfolio construction."""
    simulator = PortfolioSimulator()
    
    print("\n📋 Available Assets for Portfolio Construction:")
    print("="*50)
    
    # Group assets by category
    categories = {
        "📈 Stock Market Indices": ["S&P 500", "NASDAQ", "Dow Jones", "Russell 2000", "EuroStoxx 50", "FTSE 100", "Nikkei 225"],
        "🌍 Global & Sector ETFs": ["MSCI World", "MSCI Emerging Markets", "VTI (Total Stock Market)", "VOO (S&P 500 ETF)", "VEA (Developed Markets)", "VWO (Emerging Markets)"],
        "🏅 Commodities": ["Gold", "Silver", "Oil (Crude)"],
        "💰 Cryptocurrencies": ["Bitcoin", "Ethereum"],
        "🏦 Bonds & Fixed Income": ["10-Year Treasury", "BND (Total Bond Market)"]
    }
    
    for category, assets in categories.items():
        print(f"\n{category}:")
        for asset in assets:
            if asset in simulator.AVAILABLE_ASSETS:
                ticker = simulator.AVAILABLE_ASSETS[asset]
                print(f"  • {asset} ({ticker})")
    
    print(f"\nTotal: {len(simulator.AVAILABLE_ASSETS)} assets available")

if __name__ == "__main__":
    print("🏦 Comprehensive Finance Tool - Portfolio Demo")
    print("=" * 60)
    
    # List available assets
    list_available_assets()
    
    # Run the demo
    print("\n" + "="*60)
    demo_portfolio_simulation()
    
    print("\n" + "="*60)
    print("💡 To use the full interactive interface, run:")
    print("   streamlit run app.py")
    print("\n📖 Check out the Portfolio tab for:")
    print("   • Interactive portfolio builder")
    print("   • Real-time backtesting")
    print("   • Monte Carlo simulations")
    print("   • Correlation analysis")
    print("   • Portfolio optimization suggestions")
