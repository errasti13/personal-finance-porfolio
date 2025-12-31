#!/usr/bin/env python3
"""
Portfolio Simulator Module
Integrates portfolio backtesting and simulation functionality using Yahoo Finance data.
Adapted from the portfolio-tester project for use with Streamlit.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class PortfolioSimulator:
    """
    Portfolio simulation and backtesting class using Yahoo Finance data.
    """
    
    # Yahoo Finance ticker symbols for major assets
    AVAILABLE_ASSETS = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'Dow Jones': '^DJI',
        'MSCI World': 'URTH',  # ETF tracking MSCI World
        'MSCI Emerging Markets': 'EEM',  # ETF tracking EM
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'EuroStoxx 50': '^STOXX50E',
        'FTSE 100': '^FTSE',
        'Nikkei 225': '^N225',
        'Russell 2000': '^RUT',
        '10-Year Treasury': '^TNX',
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Oil (Crude)': 'CL=F',
        'VTI (Total Stock Market)': 'VTI',
        'VOO (S&P 500 ETF)': 'VOO',
        'VEA (Developed Markets)': 'VEA',
        'VWO (Emerging Markets)': 'VWO',
        'BND (Total Bond Market)': 'BND'
    }
    
    def __init__(self):
        """Initialize the Portfolio Simulator."""
        self._data_cache = {}
        
    def get_historical_data(self, tickers: List[str], start_date: str, 
                          end_date: str, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers from Yahoo Finance.
        
        Args:
            tickers: List of Yahoo Finance ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with ticker as key and DataFrame with OHLCV data as value
        """
        data = {}
        
        def fetch_ticker_data(ticker):
            try:
                # Check cache first
                cache_key = f"{ticker}_{start_date}_{end_date}"
                if cache_key in self._data_cache:
                    return ticker, self._data_cache[cache_key]
                
                # Fetch data from Yahoo Finance
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date, auto_adjust=True)
                
                if not df.empty:
                    # Cache the data
                    self._data_cache[cache_key] = df
                    return ticker, df
                else:
                    st.warning(f"No data available for {ticker}")
                    return ticker, pd.DataFrame()
                    
            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {str(e)}")
                return ticker, pd.DataFrame()
        
        # Use ThreadPoolExecutor for concurrent downloads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_ticker_data, ticker) for ticker in tickers]
            
            for i, future in enumerate(as_completed(futures)):
                ticker, df = future.result()
                data[ticker] = df
                
                if progress_callback:
                    progress_callback(i + 1, len(tickers))
                    
        return data
    
    def calculate_portfolio_returns(self, allocations: Dict[str, float], 
                                  data: Dict[str, pd.DataFrame],
                                  initial_investment: float = 10000,
                                  rebalance_frequency: str = 'monthly',
                                  periodic_contribution: float = 0,
                                  contribution_frequency: str = 'monthly',
                                  periodic_withdrawal: float = 0,
                                  withdrawal_frequency: str = 'monthly') -> pd.DataFrame:
        """
        Calculate portfolio returns based on allocations and historical data.
        
        Args:
            allocations: Dictionary with ticker as key and allocation percentage as value
            data: Historical data for each ticker
            initial_investment: Starting portfolio value
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly', or 'none'
            periodic_contribution: Amount to add to portfolio periodically (positive number)
            contribution_frequency: How often to add money ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
            periodic_withdrawal: Amount to withdraw from portfolio periodically (positive number)
            withdrawal_frequency: How often to withdraw money ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
            
        Returns:
            DataFrame with portfolio performance metrics
        """
        if not allocations or not data:
            return pd.DataFrame()
        
        # Find common date range
        all_dates = None
        valid_tickers = []
        
        for ticker, allocation in allocations.items():
            if ticker in data and not data[ticker].empty and allocation > 0:
                if all_dates is None:
                    all_dates = data[ticker].index
                else:
                    all_dates = all_dates.intersection(data[ticker].index)
                valid_tickers.append(ticker)
        
        if not valid_tickers or all_dates.empty:
            st.warning("No overlapping data found for selected assets")
            return pd.DataFrame()
        
        # Create price matrix
        prices = pd.DataFrame()
        for ticker in valid_tickers:
            prices[ticker] = data[ticker].loc[all_dates]['Close']
        
        # Remove any remaining NaN values
        prices = prices.dropna()
        
        if prices.empty:
            st.warning("No valid price data after cleaning")
            return pd.DataFrame()
        
        # Calculate returns
        returns = prices.pct_change().fillna(0)
        
        # Initialize portfolio tracking
        portfolio_value = pd.Series(index=prices.index, dtype=float)
        portfolio_value.iloc[0] = initial_investment
        total_contributions = pd.Series(index=prices.index, dtype=float)
        total_contributions.iloc[0] = initial_investment
        total_withdrawals = pd.Series(index=prices.index, dtype=float)
        total_withdrawals.iloc[0] = 0
        
        # Calculate shares based on initial allocation
        shares = {}
        for ticker in valid_tickers:
            allocation_amount = initial_investment * allocations[ticker] / 100
            shares[ticker] = allocation_amount / prices[ticker].iloc[0]
        
        # Calculate portfolio value over time
        for i in range(1, len(prices)):
            date = prices.index[i]
            previous_date = prices.index[i-1]
            
            # Check for periodic contributions
            contribution_amount = 0
            if periodic_contribution > 0:
                if self._should_add_money(date, previous_date, contribution_frequency):
                    contribution_amount = periodic_contribution
            
            # Check for periodic withdrawals
            withdrawal_amount = 0
            if periodic_withdrawal > 0:
                if self._should_add_money(date, previous_date, withdrawal_frequency):
                    withdrawal_amount = periodic_withdrawal
            
            # Apply contribution/withdrawal before market movements
            previous_value = portfolio_value.iloc[i-1]
            adjusted_value = previous_value + contribution_amount - withdrawal_amount
            
            # Update shares if money was added or withdrawn
            if contribution_amount > 0 or withdrawal_amount > 0:
                # Redistribute the adjusted portfolio value according to allocations
                for ticker in valid_tickers:
                    allocation_amount = adjusted_value * allocations[ticker] / 100
                    shares[ticker] = allocation_amount / prices[ticker].iloc[i-1]
            
            # Check if rebalancing is needed
            should_rebalance = self._should_rebalance(date, previous_date, rebalance_frequency)
            
            if should_rebalance and rebalance_frequency != 'none':
                # Rebalance: recalculate shares based on current adjusted portfolio value
                for ticker in valid_tickers:
                    allocation_amount = adjusted_value * allocations[ticker] / 100
                    shares[ticker] = allocation_amount / prices[ticker].iloc[i-1]
            
            # Calculate current portfolio value after market movements
            current_value = sum(shares[ticker] * prices[ticker].iloc[i] for ticker in valid_tickers)
            portfolio_value.iloc[i] = current_value
            
            # Track cumulative contributions and withdrawals
            total_contributions.iloc[i] = total_contributions.iloc[i-1] + contribution_amount
            total_withdrawals.iloc[i] = total_withdrawals.iloc[i-1] + withdrawal_amount
        
        # Calculate net contributions (total money invested)
        net_contributions = total_contributions - total_withdrawals
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': prices.index,
            'Portfolio_Value': portfolio_value.values,
            'Daily_Return': portfolio_value.pct_change().fillna(0),
            'Cumulative_Return': ((portfolio_value / net_contributions - 1) * 100),
            'Total_Contributions': total_contributions.values,
            'Total_Withdrawals': total_withdrawals.values,
            'Net_Contributions': net_contributions.values
        })
        
        # Add individual asset values for comparison (buy and hold with same contribution pattern)
        for ticker in valid_tickers:
            # Calculate what this asset would be worth with the same contribution pattern
            asset_shares = 0
            asset_value_series = pd.Series(index=prices.index, dtype=float)
            asset_value_series.iloc[0] = initial_investment * allocations[ticker] / 100
            asset_shares = asset_value_series.iloc[0] / prices[ticker].iloc[0]
            
            for i in range(1, len(prices)):
                date = prices.index[i]
                previous_date = prices.index[i-1]
                
                # Add contributions to this asset
                contribution_amount = 0
                if periodic_contribution > 0:
                    if self._should_add_money(date, previous_date, contribution_frequency):
                        contribution_amount = periodic_contribution * allocations[ticker] / 100
                        asset_shares += contribution_amount / prices[ticker].iloc[i-1]
                
                # Subtract withdrawals from this asset
                withdrawal_amount = 0
                if periodic_withdrawal > 0:
                    if self._should_add_money(date, previous_date, withdrawal_frequency):
                        withdrawal_amount = periodic_withdrawal * allocations[ticker] / 100
                        asset_shares -= withdrawal_amount / prices[ticker].iloc[i-1]
                
                asset_value_series.iloc[i] = asset_shares * prices[ticker].iloc[i]
            
            results[f'{ticker}_Value'] = asset_value_series.values
        
        return results
    
    def _should_rebalance(self, current_date: pd.Timestamp, previous_date: pd.Timestamp, 
                         frequency: str) -> bool:
        """Check if rebalancing should occur based on frequency."""
        if frequency == 'daily':
            return True
        elif frequency == 'weekly':
            return current_date.week != previous_date.week
        elif frequency == 'monthly':
            return current_date.month != previous_date.month
        elif frequency == 'quarterly':
            return (current_date.month - 1) // 3 != (previous_date.month - 1) // 3
        elif frequency == 'yearly':
            return current_date.year != previous_date.year
        else:  # 'none'
            return False
    
    def _should_add_money(self, current_date: pd.Timestamp, previous_date: pd.Timestamp, 
                         frequency: str) -> bool:
        """Check if money should be added/withdrawn based on frequency."""
        if frequency == 'daily':
            return True
        elif frequency == 'weekly':
            return current_date.week != previous_date.week
        elif frequency == 'monthly':
            return current_date.month != previous_date.month
        elif frequency == 'quarterly':
            return (current_date.month - 1) // 3 != (previous_date.month - 1) // 3
        elif frequency == 'yearly':
            return current_date.year != previous_date.year
        else:
            return False
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Args:
            results: DataFrame with portfolio performance data
            
        Returns:
            Dictionary with performance metrics
        """
        if results.empty or 'Daily_Return' not in results.columns:
            return {}
        
        returns = results['Daily_Return'].dropna()
        portfolio_values = results['Portfolio_Value'].dropna()
        
        if len(returns) == 0:
            return {}
        
        # Get contribution data if available
        net_contributions = results.get('Net_Contributions', portfolio_values.iloc[0])
        total_contributions = results.get('Total_Contributions', portfolio_values.iloc[0])
        total_withdrawals = results.get('Total_Withdrawals', 0)
        
        # Calculate metrics accounting for contributions
        if isinstance(net_contributions, pd.Series):
            # Money-weighted return (IRR approximation)
            final_value = portfolio_values.iloc[-1]
            total_invested = net_contributions.iloc[-1]
            total_return = (final_value / total_invested - 1) * 100 if total_invested > 0 else 0
            
            # Time-weighted return (more accurate for performance measurement)
            time_weighted_return = results['Cumulative_Return'].iloc[-1] if 'Cumulative_Return' in results.columns else total_return
        else:
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
            time_weighted_return = total_return
        
        annualized_return = ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** 
                           (252 / len(returns)) - 1) * 100
        
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
        
        # Best and worst days
        best_day = returns.max() * 100
        worst_day = returns.min() * 100
        
        # Win rate
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = (positive_days / total_days) * 100 if total_days > 0 else 0
        
        metrics = {
            'Total Return (%)': round(total_return, 2),
            'Time-Weighted Return (%)': round(time_weighted_return, 2) if 'time_weighted_return' in locals() else round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Maximum Drawdown (%)': round(max_drawdown, 2),
            'Best Day (%)': round(best_day, 2),
            'Worst Day (%)': round(worst_day, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Total Days': total_days,
            'Final Value': round(portfolio_values.iloc[-1], 2) if not portfolio_values.empty else 0
        }
        
        # Add contribution-related metrics if available
        if isinstance(total_contributions, pd.Series):
            metrics['Total Contributions'] = round(total_contributions.iloc[-1], 2)
            metrics['Total Withdrawals'] = round(total_withdrawals.iloc[-1], 2)
            metrics['Net Contributions'] = round(net_contributions.iloc[-1], 2)
            
        return metrics
    
    def monte_carlo_simulation(self, allocations: Dict[str, float], 
                             data: Dict[str, pd.DataFrame],
                             initial_investment: float = 10000,
                             years: int = 10, 
                             simulations: int = 1000,
                             progress_callback=None) -> Dict:
        """
        Run Monte Carlo simulation for portfolio projections.
        
        Args:
            allocations: Dictionary with ticker as key and allocation percentage as value
            data: Historical data for each ticker
            initial_investment: Starting portfolio value
            years: Number of years to project
            simulations: Number of simulation runs
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with simulation results
        """
        if not allocations or not data:
            return {}
        
        # Calculate historical returns for each asset
        returns_data = {}
        valid_tickers = []
        
        for ticker, allocation in allocations.items():
            if ticker in data and not data[ticker].empty and allocation > 0:
                prices = data[ticker]['Close'].dropna()
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    returns_data[ticker] = returns
                    valid_tickers.append(ticker)
        
        if not valid_tickers:
            return {}
        
        # Run simulations
        simulation_results = []
        trading_days = years * 252
        
        for sim in range(simulations):
            portfolio_value = initial_investment
            
            for day in range(trading_days):
                # Calculate daily portfolio return
                daily_return = 0
                for ticker in valid_tickers:
                    # Sample random return from historical data
                    random_return = np.random.choice(returns_data[ticker])
                    daily_return += (allocations[ticker] / 100) * random_return
                
                portfolio_value *= (1 + daily_return)
            
            simulation_results.append(portfolio_value)
            
            if progress_callback and sim % 100 == 0:
                progress_callback(sim + 1, simulations)
        
        # Calculate statistics
        results = np.array(simulation_results)
        
        return {
            'mean': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'percentile_5': np.percentile(results, 5),
            'percentile_25': np.percentile(results, 25),
            'percentile_75': np.percentile(results, 75),
            'percentile_95': np.percentile(results, 95),
            'min': np.min(results),
            'max': np.max(results),
            'all_results': results.tolist()
        }
    
    def get_asset_correlation(self, tickers: List[str], data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate correlation matrix for selected assets.
        
        Args:
            tickers: List of ticker symbols
            data: Historical data for each ticker
            
        Returns:
            Correlation matrix DataFrame
        """
        if not tickers or not data:
            return pd.DataFrame()
        
        # Create returns matrix
        returns_matrix = pd.DataFrame()
        
        for ticker in tickers:
            if ticker in data and not data[ticker].empty:
                prices = data[ticker]['Close'].dropna()
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    returns_matrix[ticker] = returns
        
        if returns_matrix.empty:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = returns_matrix.corr()
        
        return correlation_matrix
    
    def optimize_portfolio(self, tickers: List[str], data: Dict[str, pd.DataFrame], 
                         target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Simple portfolio optimization using equal-risk contribution.
        
        Args:
            tickers: List of ticker symbols
            data: Historical data for each ticker
            target_return: Target annual return (optional)
            
        Returns:
            Dictionary with optimized allocations
        """
        if not tickers or not data:
            return {}
        
        # Calculate returns and volatilities
        asset_stats = {}
        
        for ticker in tickers:
            if ticker in data and not data[ticker].empty:
                prices = data[ticker]['Close'].dropna()
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    annual_return = (1 + returns.mean()) ** 252 - 1
                    annual_volatility = returns.std() * np.sqrt(252)
                    
                    asset_stats[ticker] = {
                        'return': annual_return,
                        'volatility': annual_volatility,
                        'sharpe': annual_return / annual_volatility if annual_volatility > 0 else 0
                    }
        
        if not asset_stats:
            return {}
        
        # Simple equal-risk allocation (inverse volatility weighting)
        total_inv_vol = sum(1 / stats['volatility'] if stats['volatility'] > 0 else 0 
                           for stats in asset_stats.values())
        
        optimized_allocations = {}
        for ticker, stats in asset_stats.items():
            if stats['volatility'] > 0 and total_inv_vol > 0:
                allocation = (1 / stats['volatility']) / total_inv_vol * 100
                optimized_allocations[ticker] = round(allocation, 2)
            else:
                optimized_allocations[ticker] = 0
        
        return optimized_allocations
