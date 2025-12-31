#!/usr/bin/env python3
"""
Performance test for the optimized Monte Carlo simulation.
"""

import numpy as np
from modules.portfolio_simulator import PortfolioSimulator
import time

def test_monte_carlo_performance():
    """Test the optimized Monte Carlo simulation performance."""
    print("🚀 Testing Optimized Monte Carlo Performance")
    print("=" * 50)
    
    # Initialize simulator
    simulator = PortfolioSimulator()
    
    # Test configuration
    allocations = {'^GSPC': 60, 'URTH': 30, 'GC=F': 10}
    
    print("📊 Fetching test data...")
    historical_data = simulator.get_historical_data(
        list(allocations.keys()), 
        '2020-01-01', 
        '2024-12-31'
    )
    
    # Test different simulation sizes
    test_configs = [
        (100, "Small test"),
        (1000, "Medium test"), 
        (5000, "Large test"),
        (10000, "Extra large test")
    ]
    
    for num_sims, desc in test_configs:
        print(f"\n🎲 {desc} ({num_sims:,} simulations)...")
        
        start_time = time.time()
        results = simulator.monte_carlo_simulation(
            allocations, 
            historical_data, 
            10000,  # Initial investment
            10,     # Years
            num_sims,
            periodic_contribution=500,
            contribution_frequency='monthly'
        )
        end_time = time.time()
        
        duration = end_time - start_time
        sims_per_second = num_sims / duration
        
        print(f"⏱️  Duration: {duration:.3f} seconds")
        print(f"🏃 Speed: {sims_per_second:,.0f} simulations/second")
        print(f"💰 Expected value: ${results['mean']:,.0f}")
        print(f"📈 Median: ${results['median']:,.0f}")
        print(f"🎯 95th percentile: ${results['percentile_95']:,.0f}")
    
    print(f"\n✅ Performance test completed successfully!")
    print("🚀 The vectorized implementation should be 10-100x faster than the original!")

if __name__ == "__main__":
    test_monte_carlo_performance()
