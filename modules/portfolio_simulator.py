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
                
                if not df.empty and len(df) > 5:  # Require at least 5 data points
                    # Cache the data
                    self._data_cache[cache_key] = df
                    return ticker, df
                else:
                    st.warning(f"Insufficient data available for {ticker} ({len(df) if not df.empty else 0} data points)")
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
            
            # Check if rebalancing is needed BEFORE applying market movements
            should_rebalance = self._should_rebalance(date, previous_date, rebalance_frequency)
            
            # Update shares based on contributions/withdrawals or rebalancing
            if contribution_amount > 0 or withdrawal_amount > 0 or (should_rebalance and rebalance_frequency != 'none'):
                # Use previous day's closing prices for allocation decisions (no look-ahead bias)
                for ticker in valid_tickers:
                    allocation_amount = adjusted_value * allocations[ticker] / 100
                    shares[ticker] = allocation_amount / prices[ticker].iloc[i-1]
            
            # Calculate current portfolio value after market movements using current day's prices
            current_value = sum(shares[ticker] * prices[ticker].iloc[i] for ticker in valid_tickers)
            portfolio_value.iloc[i] = current_value
            
            # Track cumulative contributions and withdrawals
            total_contributions.iloc[i] = total_contributions.iloc[i-1] + contribution_amount
            total_withdrawals.iloc[i] = total_withdrawals.iloc[i-1] + withdrawal_amount
        
        # Calculate net contributions (total money invested)
        net_contributions = total_contributions - total_withdrawals
        
        # Calculate time-weighted returns properly
        # This measures portfolio performance independent of cash flow timing
        time_weighted_returns = []
        time_weighted_cumulative_series = pd.Series(index=prices.index, dtype=float)
        time_weighted_cumulative_series.iloc[0] = 0.0  # 0% return at start
        
        cumulative_multiplier = 1.0
        
        for i in range(1, len(prices)):
            previous_value = portfolio_value.iloc[i-1]
            current_value = portfolio_value.iloc[i]
            
            # Check for cash flows on this date
            contribution_amount = 0
            withdrawal_amount = 0
            if periodic_contribution > 0:
                if self._should_add_money(prices.index[i], prices.index[i-1], contribution_frequency):
                    contribution_amount = periodic_contribution
            if periodic_withdrawal > 0:
                if self._should_add_money(prices.index[i], prices.index[i-1], withdrawal_frequency):
                    withdrawal_amount = periodic_withdrawal
            
            net_flow = contribution_amount - withdrawal_amount
            
            # Calculate the portfolio value before market movement (after cash flows)
            value_before_market = previous_value + net_flow
            
            # Calculate pure market return (time-weighted)
            if value_before_market > 0:
                daily_market_return = (current_value - value_before_market) / value_before_market
            else:
                daily_market_return = 0.0
            
            time_weighted_returns.append(daily_market_return)
            
            # Update cumulative time-weighted return
            cumulative_multiplier *= (1 + daily_market_return)
            time_weighted_cumulative_series.iloc[i] = (cumulative_multiplier - 1) * 100
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': prices.index,
            'Portfolio_Value': portfolio_value.values,
            'Daily_Return': np.array([0.0] + time_weighted_returns),  # Pure market returns (time-weighted)
            'Cumulative_Return': time_weighted_cumulative_series.values,  # Time-weighted cumulative returns
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
        
        # Daily_Return now contains pure time-weighted returns (market performance only)
        time_weighted_daily_returns = results['Daily_Return'].dropna()
        portfolio_values = results['Portfolio_Value'].dropna()
        
        if len(time_weighted_daily_returns) == 0 or len(portfolio_values) == 0:
            return {}
        
        # Get contribution data if available
        net_contributions = results.get('Net_Contributions', portfolio_values.iloc[0])
        total_contributions = results.get('Total_Contributions', portfolio_values.iloc[0])
        total_withdrawals = results.get('Total_Withdrawals', 0)
        
        # Time-weighted return (pure portfolio performance, independent of cash flow timing)
        if 'Cumulative_Return' in results.columns:
            time_weighted_total_return = results['Cumulative_Return'].iloc[-1]
        else:
            # Fallback calculation if cumulative return not available
            cumulative_multiplier = 1.0
            for ret in time_weighted_daily_returns:
                cumulative_multiplier *= (1 + ret)
            time_weighted_total_return = (cumulative_multiplier - 1) * 100
        
        # Money-weighted return (what investor experienced with contribution timing)
        if isinstance(net_contributions, pd.Series) and net_contributions.iloc[-1] > 0:
            final_value = portfolio_values.iloc[-1]
            total_invested = net_contributions.iloc[-1]
            money_weighted_return = (final_value / total_invested - 1) * 100
        else:
            # No external contributions - money-weighted equals time-weighted
            money_weighted_return = time_weighted_total_return
        
        # Calculate annualized time-weighted return (using trading days)
        if len(time_weighted_daily_returns) > 0:
            trading_days = len(time_weighted_daily_returns)
            years = trading_days / 252.0  # Standard trading days per year
            if years > 0:
                annualized_time_weighted = ((1 + time_weighted_total_return/100) ** (1/years) - 1) * 100
            else:
                annualized_time_weighted = 0
        else:
            annualized_time_weighted = 0
        
        # Volatility (annualized standard deviation of time-weighted returns)
        if len(time_weighted_daily_returns) > 1:
            volatility = time_weighted_daily_returns.std() * np.sqrt(252) * 100
        else:
            volatility = 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_time_weighted / volatility if volatility > 0 else 0
        
        # Maximum drawdown (based on time-weighted cumulative returns)
        if 'Cumulative_Return' in results.columns:
            cumulative_returns = results['Cumulative_Return'] / 100  # Convert to decimal
            rolling_max = (1 + cumulative_returns).expanding().max()
            drawdowns = ((1 + cumulative_returns) - rolling_max) / rolling_max * 100
            max_drawdown = drawdowns.min()
        else:
            # Fallback: calculate drawdown from portfolio values (less accurate with contributions)
            rolling_max = portfolio_values.expanding().max()
            drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
            max_drawdown = drawdowns.min()
        
        # Best and worst days (time-weighted returns)
        best_day = time_weighted_daily_returns.max() * 100
        worst_day = time_weighted_daily_returns.min() * 100
        
        # Win rate (percentage of positive return days)
        positive_days = (time_weighted_daily_returns > 0).sum()
        total_trading_days = len(time_weighted_daily_returns)
        win_rate = (positive_days / total_trading_days) * 100 if total_trading_days > 0 else 0
        
        metrics = {
            'Total Return (%)': round(money_weighted_return, 2),
            'Time-Weighted Return (%)': round(time_weighted_total_return, 2),
            'Annualized Return (%)': round(annualized_time_weighted, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Maximum Drawdown (%)': round(max_drawdown, 2),
            'Best Day (%)': round(best_day, 2),
            'Worst Day (%)': round(worst_day, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Total Days': total_trading_days,
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
                             simulations: int = 10000,
                             progress_callback=None,
                             periodic_contribution: float = 0,
                             contribution_frequency: str = 'monthly',
                             periodic_withdrawal: float = 0,
                             withdrawal_frequency: str = 'monthly') -> Dict:
        """
        Run Monte Carlo simulation using the exact logic from portfolio-tester.
        Uses monthly returns for more realistic portfolio simulation.
        
        Args:
            allocations: Dictionary with ticker as key and allocation percentage as value
            data: Historical data for each ticker
            initial_investment: Starting portfolio value
            years: Number of years to project
            simulations: Number of simulation runs
            progress_callback: Optional callback for progress updates
            periodic_contribution: Amount to contribute periodically
            contribution_frequency: How often to contribute ('monthly', 'yearly')
            periodic_withdrawal: Amount to withdraw periodically  
            withdrawal_frequency: How often to withdraw ('monthly', 'yearly')
            
        Returns:
            Dictionary with simulation results including best/worst/median scenarios
        """
        if not allocations or not data:
            return {}
        
        # Convert daily data to monthly data for each asset
        monthly_data = {}
        monthly_returns = {}
        valid_tickers = []
        
        for ticker, allocation in allocations.items():
            if ticker in data and not data[ticker].empty and allocation > 0:
                prices = data[ticker]['Close'].dropna()
                if len(prices) > 1:
                    # Convert to monthly data (use last trading day of each month)
                    monthly_prices = prices.resample('ME').last().dropna()
                    
                    if len(monthly_prices) > 1:
                        # Calculate true monthly returns (accounting for compounding)
                        monthly_return_series = monthly_prices.pct_change().dropna()
                        
                        monthly_data[ticker] = monthly_prices
                        monthly_returns[ticker] = monthly_return_series.values
                        valid_tickers.append(ticker)
        
        if not valid_tickers:
            return {}
        
        # Find minimum data points across all assets
        min_data_points = min(len(monthly_returns[ticker]) for ticker in valid_tickers)
        simulation_length_months = years * 12
        
        if min_data_points < simulation_length_months:
            return {}
        
        # Simulation parameters
        months_per_year = 12
        contribution_interval = 1 if contribution_frequency == 'monthly' else 12
        withdrawal_interval = 1 if withdrawal_frequency == 'monthly' else 12
        
        # Net periodic cash flow
        net_periodic_flow = periodic_contribution - periodic_withdrawal
        
        # Run simulations
        all_simulations = []
        small_value_threshold = 0.01
        
        for sim_idx in range(simulations):
            try:
                # Random starting point in historical data
                start_idx = np.random.randint(0, min_data_points - simulation_length_months)
                
                # Initialize simulation state
                portfolio_value = initial_investment
                total_invested = initial_investment
                is_portfolio_active = True
                value_history = []
                
                # Track investments for better return calculation
                investments = [{'amount': initial_investment, 'months_invested': simulation_length_months}]
                
                # Simulate each month
                for month in range(simulation_length_months):
                    if is_portfolio_active:
                        # Calculate weighted portfolio return for this month
                        monthly_portfolio_return = 0.0
                        for ticker in valid_tickers:
                            asset_return = monthly_returns[ticker][start_idx + month]
                            weight = allocations[ticker] / 100
                            monthly_portfolio_return += asset_return * weight
                        
                        # Apply return to current portfolio value
                        portfolio_value *= (1 + monthly_portfolio_return)
                        
                        # Handle periodic contributions/withdrawals
                        if month % contribution_interval == 0 and periodic_contribution != 0:
                            portfolio_value += periodic_contribution
                            if periodic_contribution > 0:
                                total_invested += periodic_contribution
                                investments.append({
                                    'amount': periodic_contribution,
                                    'months_invested': simulation_length_months - month
                                })
                        
                        if month % withdrawal_interval == 0 and periodic_withdrawal != 0:
                            portfolio_value -= periodic_withdrawal
                        
                        # Check for portfolio depletion
                        if portfolio_value < small_value_threshold:
                            portfolio_value = 0
                            is_portfolio_active = False
                    
                    # Record monthly value
                    value_history.append(portfolio_value)
                
                # Calculate returns
                total_return = (portfolio_value / total_invested - 1) if total_invested > 0 else -1
                annualized_return = (1 + total_return) ** (1 / years) - 1
                
                # Calculate maximum drawdown
                max_drawdown = 0.0
                if value_history:
                    peak = value_history[0]
                    for value in value_history:
                        if value > peak:
                            peak = value
                        if peak > 0:
                            drawdown = (peak - value) / peak
                            max_drawdown = max(max_drawdown, drawdown)
                
                # Find portfolio depletion month (if any)
                depletion_month = None
                if not is_portfolio_active:
                    try:
                        depletion_month = next(i for i, v in enumerate(value_history) if v == 0)
                    except StopIteration:
                        pass
                
                # Store simulation result
                simulation_result = {
                    'total_return': total_return * 100,
                    'annualized_return': annualized_return * 100,
                    'initial_investment': initial_investment,
                    'total_invested': total_invested,
                    'final_value': portfolio_value,
                    'value_history': value_history,
                    'max_drawdown': max_drawdown * 100,
                    'portfolio_depletion_month': depletion_month,
                    'is_active': is_portfolio_active
                }
                
                all_simulations.append(simulation_result)
                
            except Exception as e:
                continue  # Skip failed simulations
            
            # Progress callback
            if progress_callback and (sim_idx + 1) % 100 == 0:
                progress_callback(sim_idx + 1, simulations)
        
        if not all_simulations:
            return {}
        
        # Sort by total return
        all_simulations.sort(key=lambda x: x['total_return'], reverse=True)
        
        # Calculate statistics
        best_case = all_simulations[0]
        worst_case = all_simulations[-1]
        median_idx = len(all_simulations) // 2
        median_case = all_simulations[median_idx]
        
        # Success rate (portfolios that didn't deplete)
        success_rate = len([s for s in all_simulations if s['is_active']]) / len(all_simulations) * 100
        
        # Create time axis (in years)
        time_axis = np.linspace(0, years, simulation_length_months)
        
        # Extract final values for distribution statistics
        final_values = [s['final_value'] for s in all_simulations]
        
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
            'success_rate': success_rate,
            'total_simulations': len(all_simulations),
            'time_axis': time_axis.tolist(),
            'best_case_series': best_case['value_history'],
            'worst_case_series': worst_case['value_history'],
            'median_case_series': median_case['value_history'],
            'best_case': {
                'total_return': best_case['total_return'],
                'annualized_return': best_case['annualized_return'],
                'final_value': best_case['final_value'],
                'max_drawdown': best_case['max_drawdown']
            },
            'worst_case': {
                'total_return': worst_case['total_return'],
                'annualized_return': worst_case['annualized_return'],
                'final_value': worst_case['final_value'],
                'max_drawdown': worst_case['max_drawdown']
            },
            'median_case': {
                'total_return': median_case['total_return'],
                'annualized_return': median_case['annualized_return'],
                'final_value': median_case['final_value'],
                'max_drawdown': median_case['max_drawdown']
            }
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
