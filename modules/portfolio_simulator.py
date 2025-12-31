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
        
        # Calculate time-weighted return (pure portfolio performance)
        # This measures how well the portfolio strategy performed, independent of contribution timing
        time_weighted_returns = []
        for i in range(1, len(prices)):
            previous_value = portfolio_value.iloc[i-1]
            current_value_before_flows = previous_value
            
            # Check for contributions/withdrawals today
            contribution_amount = 0
            withdrawal_amount = 0
            if periodic_contribution > 0:
                if self._should_add_money(prices.index[i], prices.index[i-1], contribution_frequency):
                    contribution_amount = periodic_contribution
            if periodic_withdrawal > 0:
                if self._should_add_money(prices.index[i], prices.index[i-1], withdrawal_frequency):
                    withdrawal_amount = periodic_withdrawal
            
            # Adjust for flows to get the value after flows but before market movement
            value_after_flows = previous_value + contribution_amount - withdrawal_amount
            
            # Calculate the return due to market movement only
            if value_after_flows > 0:
                market_return = (portfolio_value.iloc[i] - value_after_flows) / value_after_flows
                time_weighted_returns.append(market_return)
        
        # Compound the time-weighted returns
        time_weighted_cumulative = 1.0
        for ret in time_weighted_returns:
            time_weighted_cumulative *= (1 + ret)
        time_weighted_return_total = (time_weighted_cumulative - 1) * 100
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': prices.index,
            'Portfolio_Value': portfolio_value.values,
            'Daily_Return': portfolio_value.pct_change().fillna(0),
            'Cumulative_Return': time_weighted_return_total,  # This is now truly time-weighted
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
            # This is what the investor actually experienced considering contribution timing
            final_value = portfolio_values.iloc[-1]
            total_invested = net_contributions.iloc[-1]
            money_weighted_return = (final_value / total_invested - 1) * 100 if total_invested > 0 else 0
            
            # Time-weighted return (portfolio strategy performance)
            # This measures how well the investment strategy performed independent of timing
            time_weighted_return = results['Cumulative_Return'].iloc[-1] if 'Cumulative_Return' in results.columns else money_weighted_return
            
            # Use time-weighted return for annualized calculation (more accurate for strategy performance)
            if len(returns) > 0:
                days_total = (portfolio_values.index[-1] - portfolio_values.index[0]).days
                if days_total > 0:
                    annualized_time_weighted = ((1 + time_weighted_return/100) ** (365/days_total) - 1) * 100
                else:
                    annualized_time_weighted = 0
            else:
                annualized_time_weighted = 0
                
        else:
            # No contributions/withdrawals - both returns are the same
            money_weighted_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
            time_weighted_return = money_weighted_return
            annualized_time_weighted = ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** 
                               (252 / len(returns)) - 1) * 100 if len(returns) > 0 else 0
        
        # Legacy annualized return calculation (based on portfolio values)
        annualized_return = ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** 
                           (252 / len(returns)) - 1) * 100 if len(returns) > 0 else 0
        
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
            'Total Return (%)': round(money_weighted_return, 2),
            'Time-Weighted Return (%)': round(time_weighted_return, 2),
            'Annualized Return (%)': round(annualized_time_weighted, 2),
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
                             progress_callback=None,
                             periodic_contribution: float = 0,
                             contribution_frequency: str = 'monthly',
                             periodic_withdrawal: float = 0,
                             withdrawal_frequency: str = 'monthly') -> Dict:
        """
        Run Monte Carlo simulation for portfolio projections.
        
        Args:
            allocations: Dictionary with ticker as key and allocation percentage as value
            data: Historical data for each ticker
            initial_investment: Starting portfolio value
            years: Number of years to project
            simulations: Number of simulation runs
            progress_callback: Optional callback for progress updates
            periodic_contribution: Amount to contribute periodically
            contribution_frequency: How often to contribute ('monthly', 'quarterly', 'yearly')
            periodic_withdrawal: Amount to withdraw periodically
            withdrawal_frequency: How often to withdraw ('monthly', 'quarterly', 'yearly')
            
        Returns:
            Dictionary with simulation results
        """
        if not allocations or not data:
            return {}
        
        # Calculate historical returns for each asset and create return matrix
        returns_matrix = []
        valid_tickers = []
        allocation_weights = []
        
        for ticker, allocation in allocations.items():
            if ticker in data and not data[ticker].empty and allocation > 0:
                prices = data[ticker]['Close'].dropna()
                if len(prices) > 1:
                    returns = prices.pct_change().dropna().values
                    returns_matrix.append(returns)
                    valid_tickers.append(ticker)
                    allocation_weights.append(allocation / 100)
        
        if not valid_tickers:
            return {}
        
        # Convert to numpy arrays for vectorization
        returns_matrix = np.array(returns_matrix)  # Shape: (n_assets, n_historical_days)
        allocation_weights = np.array(allocation_weights)  # Shape: (n_assets,)
        
        # Calculate contribution/withdrawal schedules
        trading_days = years * 252
        contributions_per_year = {'monthly': 12, 'quarterly': 4, 'yearly': 1}
        withdrawals_per_year = {'monthly': 12, 'quarterly': 4, 'yearly': 1}
        
        contrib_freq = contributions_per_year.get(contribution_frequency, 12)
        withdraw_freq = withdrawals_per_year.get(withdrawal_frequency, 12)
        
        # Create cash flow schedule arrays
        contrib_schedule = np.zeros(trading_days)
        withdraw_schedule = np.zeros(trading_days)
        
        if periodic_contribution > 0 and contrib_freq > 0:
            contrib_days = 252 // contrib_freq
            for day in range(contrib_days, trading_days, contrib_days):
                contrib_schedule[day] = periodic_contribution
        
        if periodic_withdrawal > 0 and withdraw_freq > 0:
            withdraw_days = 252 // withdraw_freq
            for day in range(withdraw_days, trading_days, withdraw_days):
                withdraw_schedule[day] = periodic_withdrawal
        
        # Vectorized simulation - generate all random returns at once
        n_assets, n_historical_returns = returns_matrix.shape
        
        # Generate random indices for all simulations and days at once
        # Shape: (simulations, trading_days, n_assets)
        random_indices = np.random.randint(0, n_historical_returns, size=(simulations, trading_days, n_assets))
        
        # Sample returns using advanced indexing
        # Shape: (simulations, trading_days, n_assets)
        sampled_returns = returns_matrix[np.arange(n_assets)[None, None, :], random_indices]
        
        # Calculate weighted portfolio returns for all simulations
        # Shape: (simulations, trading_days)
        portfolio_daily_returns = np.sum(sampled_returns * allocation_weights[None, None, :], axis=2)
        
        # Pre-compute net cash flows for efficiency
        net_cashflow = contrib_schedule - withdraw_schedule  # Shape: (trading_days,)
        
        # Memory-efficient vectorized simulation
        if periodic_contribution == 0 and periodic_withdrawal == 0:
            # No cash flows - ultra-fast pure vectorized calculation
            if progress_callback:
                progress_callback(1, 2)
            
            daily_multipliers = 1 + portfolio_daily_returns  # Shape: (simulations, trading_days)
            final_values = initial_investment * np.prod(daily_multipliers, axis=1)
            
            if progress_callback:
                progress_callback(2, 2)
            
        else:
            # With cash flows - optimized loop with memory efficiency
            # Process in smaller chunks if dealing with large simulations to manage memory
            chunk_size = min(simulations, 10000)  # Process max 10k simulations at once
            all_final_values = []
            
            for chunk_start in range(0, simulations, chunk_size):
                chunk_end = min(chunk_start + chunk_size, simulations)
                chunk_sims = chunk_end - chunk_start
                
                # Initialize values for this chunk
                portfolio_values = np.zeros((chunk_sims, trading_days + 1))
                portfolio_values[:, 0] = initial_investment
                
                # Get returns for this chunk
                chunk_returns = portfolio_daily_returns[chunk_start:chunk_end]
                
                # Simulate this chunk
                for day in range(trading_days):
                    current_values = portfolio_values[:, day]
                    
                    # Apply cash flows
                    if net_cashflow[day] != 0:
                        current_values = np.maximum(0, current_values + net_cashflow[day])
                    
                    # Apply market returns
                    portfolio_values[:, day + 1] = current_values * (1 + chunk_returns[:, day])
                
                all_final_values.extend(portfolio_values[:, -1])
                
                # Progress callback
                if progress_callback:
                    progress_callback(chunk_end, simulations)
            
            final_values = np.array(all_final_values)
        
        # Ensure non-negative values
        final_values = np.maximum(0, final_values)
        
        # Calculate statistics from vectorized results
        return {
            'mean': np.mean(final_values),
            'median': np.median(final_values),
            'std': np.std(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'min': np.min(final_values),
            'max': np.max(final_values),
            'all_results': final_values.tolist()
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
