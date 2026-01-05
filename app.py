#!/usr/bin/env python3
"""
Comprehensive Finance Tool
A comprehensive financial management application combining net worth tracking 
and transaction analysis in a single Streamlit interface.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import yfinance as yf
import tempfile
import io
import zipfile
from typing import Dict, Optional
from modules.transaction_analyzer import TransactionAnalyzer
from modules.portfolio_simulator import PortfolioSimulator

# Page configuration
st.set_page_config(
    page_title="Comprehensive Finance Tool",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive-change {
        color: #00c851;
        font-weight: bold;
    }
    .negative-change {
        color: #ff4444;
        font-weight: bold;
    }
    .stDataFrame {
        background-color: white;
    }
    .summary-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .nav-button {
        background-color: #667eea;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.25rem;
        cursor: pointer;
    }
    .nav-button:hover {
        background-color: #764ba2;
    }
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
    .big-font {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if 'net_worth_data' not in st.session_state:
        st.session_state.net_worth_data = pd.DataFrame()
    if 'transactions_data' not in st.session_state:
        st.session_state.transactions_data = pd.DataFrame()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

def load_sample_net_worth_data():
    """Load sample net worth data from CSV file."""
    csv_path = "data/net_worth_data.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        # Create sample data if file doesn't exist
        dates = pd.date_range(start='2025-01-31', end='2025-12-31', freq='M')
        sample_data = {
            'Date': dates.strftime('%Y-%m-%d'),
            'UBS Account (CHF)': np.random.normal(10000, 2000, len(dates)).round(2),
            'IBKR Account (EUR)': np.random.normal(15000, 3000, len(dates)).round(2),
            'Kutxabank Account (EUR)': np.random.normal(1500, 500, len(dates)).round(2)
        }
        return pd.DataFrame(sample_data)

def get_forex_rate(from_currency: str, to_currency: str) -> float:
    """Get current forex rate using yfinance."""
    if from_currency == to_currency:
        return 1.0
    
    try:
        ticker = f"{from_currency}{to_currency}=X"
        forex_data = yf.download(ticker, period="1d", interval="1d", progress=False)
        if not forex_data.empty:
            rate = forex_data['Close'].iloc[-1]
            return float(rate)
    except:
        pass
    
    # Fallback rates (approximate)
    rates = {
        'EURCHF': 0.93,
        'CHFEUR': 1.08,
        'USDCHF': 0.88,
        'CHFUSD': 1.14,
        'EURUSD': 1.05,
        'USDEUR': 0.95
    }
    
    return rates.get(f"{from_currency}{to_currency}", 1.0)

def convert_to_eur(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all currencies to EUR for unified tracking."""
    df_converted = df.copy()
    
    # Get current exchange rates to EUR
    chf_to_eur = get_forex_rate('CHF', 'EUR')
    usd_to_eur = get_forex_rate('USD', 'EUR')
    
    # Convert CHF columns to EUR
    for col in df_converted.columns:
        if 'CHF' in col:
            eur_col_name = col.replace('CHF', 'EUR')
            df_converted[eur_col_name] = df_converted[col] * chf_to_eur
            df_converted = df_converted.drop(columns=[col])
        elif 'USD' in col:
            eur_col_name = col.replace('USD', 'EUR')
            df_converted[eur_col_name] = df_converted[col] * usd_to_eur
            df_converted = df_converted.drop(columns=[col])
    
    return df_converted

def net_worth_dashboard():
    """Net Worth Tracking Dashboard."""
    st.markdown('<h1 class="main-header">📈 Net Worth Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for data input options
    with st.sidebar:
        st.header("Data Management")
        
        # Option to load sample data
        if st.button("Load Sample Data", key="load_sample"):
            st.session_state.net_worth_data = load_sample_net_worth_data()
            st.success("Sample data loaded!")
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload your net worth data (CSV)",
            type=['csv'],
            help="CSV should have Date column and account columns with currency indicators like (EUR), (USD), (CHF)"
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.net_worth_data = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Manual data entry
        with st.expander("Add New Entry"):
            with st.form("add_entry"):
                entry_date = st.date_input("Date", value=datetime.now().date())
                ubs_chf = st.number_input("UBS Account (CHF)", value=0.0, step=100.0)
                ibkr_eur = st.number_input("IBKR Account (EUR)", value=0.0, step=100.0)
                kutxa_eur = st.number_input("Kutxabank Account (EUR)", value=0.0, step=100.0)
                
                if st.form_submit_button("Add Entry"):
                    new_entry = pd.DataFrame({
                        'Date': [entry_date.strftime('%Y-%m-%d')],
                        'UBS Account (CHF)': [ubs_chf],
                        'IBKR Account (EUR)': [ibkr_eur],
                        'Kutxabank Account (EUR)': [kutxa_eur]
                    })
                    
                    if st.session_state.net_worth_data.empty:
                        st.session_state.net_worth_data = new_entry
                    else:
                        st.session_state.net_worth_data = pd.concat([st.session_state.net_worth_data, new_entry], ignore_index=True)
                    
                    st.success("Entry added successfully!")
                    st.rerun()
    
    # Main dashboard
    if not st.session_state.net_worth_data.empty:
        df = st.session_state.net_worth_data.copy()
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Convert to EUR for unified tracking
        df_eur = convert_to_eur(df)
        
        # Calculate total net worth
        value_columns = [col for col in df_eur.columns if col != 'Date' and 'EUR' in col]
        df_eur['Total_EUR'] = df_eur[value_columns].sum(axis=1)
        
        # Current metrics
        latest = df_eur.iloc[-1]
        if len(df_eur) > 1:
            previous = df_eur.iloc[-2]
            change = latest['Total_EUR'] - previous['Total_EUR']
            change_pct = (change / previous['Total_EUR']) * 100
        else:
            change = 0
            change_pct = 0
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Total Net Worth",
                value=f"EUR {latest['Total_EUR']:,.2f}",
                delta=f"EUR {change:,.2f}"
            )
        
        with col2:
            ubs_value = latest.get('UBS Account (EUR)', 0)
            st.metric(
                label="🏦 UBS Account",
                value=f"EUR {ubs_value:,.2f}"
            )
        
        with col3:
            ibkr_value = latest.get('IBKR Account (EUR)', 0)
            st.metric(
                label="📊 IBKR Account",
                value=f"EUR {ibkr_value:,.2f}"
            )
        
        with col4:
            kutxa_value = latest.get('Kutxabank Account (EUR)', 0)
            st.metric(
                label="🏛️ Kutxabank Account",
                value=f"EUR {kutxa_value:,.2f}"
            )
        
        # Charts section
        st.markdown("---")
        
        # Net worth trend chart
        st.subheader("📈 Net Worth Trend")
        fig_trend = px.line(
            df_eur, 
            x='Date', 
            y='Total_EUR',
            title='Net Worth Over Time',
            labels={'Total_EUR': 'Total Net Worth (EUR)', 'Date': 'Date'}
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Account breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏦 Account Breakdown")
            account_values = {col.replace(' (EUR)', ''): latest[col] for col in value_columns}
            fig_pie = px.pie(
                values=list(account_values.values()),
                names=list(account_values.keys()),
                title='Current Account Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📊 Account Trends")
            fig_multi = go.Figure()
            for col in value_columns:
                fig_multi.add_trace(go.Scatter(
                    x=df_eur['Date'],
                    y=df_eur[col],
                    mode='lines+markers',
                    name=col.replace(' (EUR)', '')
                ))
            fig_multi.update_layout(
                title='Individual Account Trends',
                xaxis_title='Date',
                yaxis_title='Value (EUR)',
                height=400
            )
            st.plotly_chart(fig_multi, use_container_width=True)
        
        # Data table
        st.subheader("📋 Historical Data")
        st.dataframe(df, use_container_width=True)
        
        # Export functionality
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"net_worth_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Save to local data folder
            if st.button("Save to Local Data Folder"):
                os.makedirs("data", exist_ok=True)
                df.to_csv("data/net_worth_data.csv", index=False)
                st.success("Data saved to data/net_worth_data.csv")
    
    else:
        st.info("📊 No data available. Please load sample data or upload your own CSV file.")
        st.markdown("""
        **Expected CSV format:**
        ```
        Date,Bank 1 (EUR),Bank 2 (USD),Bank 3 (CHF)
        2025-01-31,10000.00,15000.00,1500.00
        2025-02-28,10500.00,15500.00,1550.00
        ```
        
        **Format Guidelines:**
        - Date column in YYYY-MM-DD format
        - Account columns with currency indicators (EUR), (USD), (CHF)
        - Numeric values for account balances
        - All currencies will be converted to EUR for unified tracking
        """)

def transaction_analysis_dashboard():
    """Transaction Analysis Dashboard."""
    st.markdown('<h1 class="main-header">💳 Transaction Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Bank Statement")
        
        # PDF upload for bank statements only
        uploaded_pdf = st.file_uploader(
            "Upload Bank Statement (PDF)",
            type=['pdf'],
            help="Upload your bank statement PDF for automatic transaction analysis. Supports UBS and other major banks."
        )
    
    # Process uploaded files
    transactions_df = None
    
    if uploaded_pdf is not None:
        with st.spinner("Analyzing PDF transactions..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                tmp_path = tmp_file.name
            
            try:
                analyzer = TransactionAnalyzer(tmp_path)
                transactions_df = analyzer.analyze_transactions()
                
                if transactions_df is not None and not transactions_df.empty:
                    st.session_state.transactions_data = transactions_df
                    st.session_state.analyzer = analyzer
                    st.success(f"Successfully processed {len(transactions_df)} transactions!")
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    # Use data from session state
    if not st.session_state.transactions_data.empty:
        df = st.session_state.transactions_data
        
        # Filter controls
        st.subheader("🔍 Filter Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(df['date'].min().date(), df['date'].max().date()),
                min_value=df['date'].min().date(),
                max_value=df['date'].max().date()
            )
        
        with col2:
            if 'category' in df.columns:
                categories = st.multiselect(
                    "Categories",
                    options=df['category'].unique(),
                    default=df['category'].unique()
                )
            else:
                categories = []
        
        with col3:
            amount_range = st.slider(
                "Amount Range",
                min_value=float(df['amount'].min()),
                max_value=float(df['amount'].max()),
                value=(float(df['amount'].min()), float(df['amount'].max()))
            )
        
        # Apply filters
        filtered_df = df[
            (df['date'].dt.date >= date_range[0]) &
            (df['date'].dt.date <= date_range[1]) &
            (df['amount'] >= amount_range[0]) &
            (df['amount'] <= amount_range[1])
        ]
        
        if categories and 'category' in df.columns:
            filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
        # Financial summary
        st.markdown("---")
        st.subheader("💰 Financial Summary")
        
        income = filtered_df[filtered_df['amount'] > 0]['amount'].sum()
        expenses = abs(filtered_df[filtered_df['amount'] < 0]['amount'].sum())
        net_savings = income - expenses
        savings_rate = (net_savings / income * 100) if income > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💵 Total Income", f"EUR {income:,.2f}")
        
        with col2:
            st.metric("💸 Total Expenses", f"EUR {expenses:,.2f}")
        
        with col3:
            st.metric("💰 Net Savings", f"EUR {net_savings:,.2f}")
        
        with col4:
            st.metric("📊 Savings Rate", f"{savings_rate:.1f}%")
        
        # Charts
        st.markdown("---")
        
        # Spending by category
        if 'category' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Spending by Category")
                expense_data = filtered_df[filtered_df['amount'] < 0].copy()
                expense_data['amount'] = abs(expense_data['amount'])
                
                if not expense_data.empty:
                    category_spending = expense_data.groupby('category')['amount'].sum().sort_values(ascending=False)
                    
                    fig_cat = px.pie(
                        values=category_spending.values,
                        names=category_spending.index,
                        title='Expense Distribution by Category'
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                st.subheader("📈 Monthly Trends")
                monthly_data = filtered_df.groupby([filtered_df['date'].dt.to_period('M'), 'category'])['amount'].sum().reset_index()
                monthly_data['date'] = monthly_data['date'].astype(str)
                
                fig_trend = px.bar(
                    monthly_data,
                    x='date',
                    y='amount',
                    color='category',
                    title='Monthly Spending by Category'
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        # Income vs Expenses over time
        st.subheader("💹 Income vs Expenses Timeline")
        
        monthly_summary = filtered_df.groupby(filtered_df['date'].dt.to_period('M')).agg({
            'amount': lambda x: [x[x > 0].sum(), abs(x[x < 0].sum())]
        })
        
        monthly_income = monthly_summary['amount'].apply(lambda x: x[0])
        monthly_expenses = monthly_summary['amount'].apply(lambda x: x[1])
        
        fig_income_exp = go.Figure()
        fig_income_exp.add_trace(go.Bar(
            x=[str(period) for period in monthly_income.index],
            y=monthly_income.values,
            name='Income',
            marker_color='green',
            opacity=0.7
        ))
        fig_income_exp.add_trace(go.Bar(
            x=[str(period) for period in monthly_expenses.index],
            y=monthly_expenses.values,
            name='Expenses',
            marker_color='red',
            opacity=0.7
        ))
        
        fig_income_exp.update_layout(
            title='Monthly Income vs Expenses',
            xaxis_title='Month',
            yaxis_title='Amount (EUR)',
            barmode='group'
        )
        st.plotly_chart(fig_income_exp, use_container_width=True)
        
        # Transaction details
        st.subheader("📋 Transaction Details")
        
        # Search functionality
        search_term = st.text_input("🔍 Search transactions", "")
        if search_term:
            search_mask = filtered_df['description'].str.contains(search_term, case=False, na=False)
            display_df = filtered_df[search_mask]
        else:
            display_df = filtered_df
        
        # Display transaction table
        st.dataframe(
            display_df.sort_values('date', ascending=False)[['date', 'description', 'amount', 'category' if 'category' in display_df.columns else 'amount']].head(100),
            use_container_width=True
        )
        
        # Export functionality
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data (CSV)",
                data=csv,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("Save to Local Data Folder"):
                os.makedirs("data", exist_ok=True)
                filtered_df.to_csv("data/processed_transactions.csv", index=False)
                st.success("Transactions saved to data/processed_transactions.csv")
    
    else:
        st.info("📊 No transaction data available. Please upload a bank statement PDF for automatic analysis.")
        st.markdown("""
        **Supported format:**
        
        **PDF Bank Statements**: Automatically extracts and categorizes transactions from bank statements
        
        **Supported Banks:**
        - UBS and other major Swiss banks
        - European banks with standard statement formats
        - Various international banks
        
        **What gets analyzed:**
        - Transaction dates and descriptions
        - Amounts (income and expenses)
        - Automatic categorization of expenses
        - Spending patterns and trends
        
        Simply upload your PDF bank statement and the system will automatically extract and analyze your transactions.
        """)

def overview_dashboard():
    """Combined Overview Dashboard."""
    st.markdown('<h1 class="main-header">🎯 Financial Overview</h1>', unsafe_allow_html=True)
    
    # Check if we have data from both modules
    has_net_worth = not st.session_state.net_worth_data.empty
    has_transactions = not st.session_state.transactions_data.empty
    
    if has_net_worth or has_transactions:
        # Key metrics row
        st.subheader("📊 Key Financial Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Net worth metrics
        if has_net_worth:
            df_nw = st.session_state.net_worth_data.copy()
            df_nw['Date'] = pd.to_datetime(df_nw['Date'])
            df_nw = df_nw.sort_values('Date')
            df_nw_eur = convert_to_eur(df_nw)
            value_columns = [col for col in df_nw_eur.columns if col != 'Date' and 'EUR' in col]
            df_nw_eur['Total_EUR'] = df_nw_eur[value_columns].sum(axis=1)
            
            latest_nw = df_nw_eur.iloc[-1]['Total_EUR']
            
            with col1:
                st.metric(
                    label="💰 Current Net Worth",
                    value=f"EUR {latest_nw:,.2f}"
                )
        
        # Transaction metrics
        if has_transactions:
            df_trans = st.session_state.transactions_data
            
            # Calculate current month metrics
            current_month = datetime.now().replace(day=1)
            current_month_data = df_trans[df_trans['date'] >= current_month]
            
            monthly_income = current_month_data[current_month_data['amount'] > 0]['amount'].sum()
            monthly_expenses = abs(current_month_data[current_month_data['amount'] < 0]['amount'].sum())
            monthly_savings = monthly_income - monthly_expenses
            
            with col2:
                st.metric(
                    label="📈 Monthly Income",
                    value=f"EUR {monthly_income:,.2f}"
                )
            
            with col3:
                st.metric(
                    label="📉 Monthly Expenses",
                    value=f"EUR {monthly_expenses:,.2f}"
                )
            
            with col4:
                st.metric(
                    label="💵 Monthly Savings",
                    value=f"EUR {monthly_savings:,.2f}",
                    delta=f"{(monthly_savings/monthly_income*100) if monthly_income > 0 else 0:.1f}% savings rate"
                )
        
        st.markdown("---")
        
        # Combined visualizations
        if has_net_worth and has_transactions:
            st.subheader("📊 Combined Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Net worth trend
                st.subheader("📈 Net Worth Growth")
                fig_nw = px.line(
                    df_nw_eur,
                    x='Date',
                    y='Total_EUR',
                    title='Net Worth Over Time'
                )
                st.plotly_chart(fig_nw, use_container_width=True)
            
            with col2:
                # Monthly cash flow
                st.subheader("💸 Monthly Cash Flow")
                
                # Group transactions by month
                df_trans_monthly = df_trans.copy()
                df_trans_monthly['month'] = df_trans_monthly['date'].dt.to_period('M')
                
                monthly_flow = df_trans_monthly.groupby('month').agg({
                    'amount': lambda x: [x[x > 0].sum(), abs(x[x < 0].sum())]
                })
                
                monthly_income_flow = monthly_flow['amount'].apply(lambda x: x[0])
                monthly_expenses_flow = monthly_flow['amount'].apply(lambda x: x[1])
                
                fig_flow = go.Figure()
                fig_flow.add_trace(go.Bar(
                    x=[str(m) for m in monthly_income_flow.index],
                    y=monthly_income_flow.values,
                    name='Income',
                    marker_color='green',
                    opacity=0.7
                ))
                fig_flow.add_trace(go.Bar(
                    x=[str(m) for m in monthly_expenses_flow.index],
                    y=monthly_expenses_flow.values,
                    name='Expenses',
                    marker_color='red',
                    opacity=0.7
                ))
                
                fig_flow.update_layout(
                    title='Monthly Income vs Expenses',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_flow, use_container_width=True)
        
        # Financial health indicators
        st.subheader("🏥 Financial Health Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if has_transactions:
                # Savings rate indicator
                total_income = df_trans[df_trans['amount'] > 0]['amount'].sum()
                total_expenses = abs(df_trans[df_trans['amount'] < 0]['amount'].sum())
                savings_rate = (total_income - total_expenses) / total_income * 100 if total_income > 0 else 0
                
                if savings_rate >= 20:
                    color = "green"
                    status = "Excellent"
                elif savings_rate >= 10:
                    color = "orange"
                    status = "Good"
                else:
                    color = "red"
                    status = "Needs Improvement"
                
                st.markdown(f"""
                <div style="padding: 1rem; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                    <h3>Savings Rate</h3>
                    <h2>{savings_rate:.1f}%</h2>
                    <p>{status}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if has_net_worth:
                # Net worth growth
                if len(df_nw_eur) > 1:
                    start_value = df_nw_eur.iloc[0]['Total_EUR']
                    end_value = df_nw_eur.iloc[-1]['Total_EUR']
                    growth_rate = (end_value - start_value) / start_value * 100
                    
                    if growth_rate > 5:
                        color = "green"
                        status = "Growing"
                    elif growth_rate > 0:
                        color = "orange"
                        status = "Stable"
                    else:
                        color = "red"
                        status = "Declining"
                    
                    st.markdown(f"""
                    <div style="padding: 1rem; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                        <h3>Net Worth Growth</h3>
                        <h2>{growth_rate:+.1f}%</h2>
                        <p>{status}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col3:
            if has_transactions and 'category' in df_trans.columns:
                # Expense diversification
                expense_categories = df_trans[df_trans['amount'] < 0]['category'].nunique()
                
                if expense_categories >= 8:
                    color = "green"
                    status = "Well Diversified"
                elif expense_categories >= 5:
                    color = "orange"
                    status = "Moderately Diversified"
                else:
                    color = "red"
                    status = "Needs Diversification"
                
                st.markdown(f"""
                <div style="padding: 1rem; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                    <h3>Expense Categories</h3>
                    <h2>{expense_categories}</h2>
                    <p>{status}</p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.info("🔄 Please load data in the Net Worth or Transaction Analysis sections to see the combined overview.")
        
        # Quick setup options
        st.subheader("🚀 Quick Setup")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Set Up Net Worth Tracking", use_container_width=True):
                st.session_state.page = "Net Worth"
                st.rerun()
        
        with col2:
            if st.button("📊 Test Portfolio Strategies", use_container_width=True):
                st.session_state.page = "Portfolio"
                st.rerun()
        
        with col3:
            if st.button("💳 Analyze Transactions", use_container_width=True):
                st.session_state.page = "Transactions"
                st.rerun()

def portfolio_dashboard():
    """Portfolio simulation and backtesting dashboard."""
    st.markdown('<h1 class="main-header">📊 Portfolio Simulator</h1>', unsafe_allow_html=True)
    
    # Initialize portfolio simulator
    if 'portfolio_simulator' not in st.session_state:
        st.session_state.portfolio_simulator = PortfolioSimulator()
    
    simulator = st.session_state.portfolio_simulator
    
    # Create tabs for different portfolio functions
    tab1, tab2 = st.tabs(["🎯 Portfolio Simulator", "📊 Analysis"])
    
    with tab1:
        st.subheader("🎯 Portfolio Simulator with Monte Carlo")
        
        # Important notice about total returns
        st.info("💡 **Total Returns**: All simulations include dividends and stock splits for accurate total return calculations.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Select Assets:**")
            
            # Asset selection
            available_assets = list(simulator.AVAILABLE_ASSETS.keys())
            selected_assets = st.multiselect(
                "Choose assets for your portfolio:",
                available_assets,
                default=["S&P 500", "Gold"],
                help="Select 2-10 assets for your portfolio"
            )
            
            if selected_assets:
                st.markdown("**Set Allocations (%):**")
                st.caption("💡 Tip: Enter any positive values - they will be normalized to relative weights automatically")
                allocations = {}
                
                # Quick data availability check when assets are selected
                with st.spinner("Checking data availability..."):
                    tickers = [simulator.AVAILABLE_ASSETS[asset] for asset in selected_assets]
                    # Use longer period for validation to match simulation needs
                    historical_end_date = datetime.now().strftime('%Y-%m-%d')
                    historical_start_date = '1990-01-01'  # Use longer period for better validation
                    
                    try:
                        test_data = simulator.get_historical_data(
                            tickers,
                            historical_start_date,
                            historical_end_date,
                            lambda x, y: None  # No progress callback for quick check
                        )
                        
                        # Check data availability with more detailed analysis
                        valid_assets = []
                        limited_assets = []
                        
                        for asset in selected_assets:
                            ticker = simulator.AVAILABLE_ASSETS[asset]
                            if ticker in test_data and not test_data[ticker].empty:
                                data_years = len(test_data[ticker]) / 252  # Approximate trading days per year
                                if data_years >= 20:  # Good for most simulations
                                    valid_assets.append(f"{asset} ({data_years:.1f} years)")
                                elif data_years >= 5:  # Limited but usable
                                    limited_assets.append(f"{asset} ({data_years:.1f} years)")
                        
                        # Show detailed data availability status
                        if valid_assets:
                            st.success(f"✅ Good data coverage: {', '.join(valid_assets)}")
                        
                        if limited_assets:
                            st.warning(f"⚠️ Limited data coverage: {', '.join(limited_assets)}. May limit simulation periods.")
                        
                        if not valid_assets and not limited_assets:
                            st.error("❌ Insufficient data for selected assets. Please choose different assets.")
                        
                    except Exception as e:
                        st.info("💡 Data availability will be verified when simulation runs.")
                
                # Quick asset compatibility check
                asset_info = []
                for asset in selected_assets:
                    ticker = simulator.AVAILABLE_ASSETS[asset]
                    asset_info.append(f"• **{asset}** ({ticker})")
                
                # Create allocation sliders
                col_count = min(len(selected_assets), 3)
                cols = st.columns(col_count)
                
                for i, asset in enumerate(selected_assets):
                    with cols[i % col_count]:
                        allocations[simulator.AVAILABLE_ASSETS[asset]] = st.slider(
                            asset,
                            0, 100, 
                            50,  # Default to equal weights that will be normalized
                            step=1,
                            key=f"allocation_{asset}",
                            help="Any positive value works - all allocations will be normalized to relative weights"
                        )
                
                # Check allocations and normalize to relative weights
                total_allocation = sum(allocations.values())
                if total_allocation > 0:
                    # Normalize allocations to relative weights (don't require exactly 100%)
                    normalized_allocations = {}
                    for ticker, allocation in allocations.items():
                        normalized_allocations[ticker] = (allocation / total_allocation) * 100
                    allocations = normalized_allocations
                    
                    # Show normalized allocations
                    allocation_display = []
                    for ticker, allocation in allocations.items():
                        asset_name = next((name for name, t in simulator.AVAILABLE_ASSETS.items() if t == ticker), ticker)
                        raw_allocation = sum(v for k, v in st.session_state.items() 
                                           if k.startswith('allocation_') and simulator.AVAILABLE_ASSETS.get(k.replace('allocation_', '')) == ticker)
                        allocation_display.append(f"**{asset_name}**: {raw_allocation:.0f}% → {allocation:.1f}%")
                else:
                    st.warning("⚠️ Please set at least one asset allocation above 0%.")
        
        with col2:
            st.markdown("**Simulation Settings:**")
            
            # Initial investment
            initial_investment = st.number_input(
                "Initial Investment ($):",
                min_value=1000,
                max_value=10000000,
                value=10000,
                step=1000
            )
            
            # Projection period
            years_to_project = st.slider(
                "Years to project:",
                1, 30, 10,
                help="Number of years to simulate forward"
            )
            
            # Validate projection period against available data
            if selected_assets and years_to_project > 15:
                st.info(f"💡 For {years_to_project}-year simulations, ensure your assets have sufficient historical data. Gold typically has ~25 years, while S&P 500 has 100+ years.")
            elif years_to_project > 25:
                st.warning(f"⚠️ {years_to_project}-year simulations require extensive historical data. Consider reducing to 20-25 years for more reliable results.")
            
            # Monte Carlo settings - hardcoded for optimal performance
            num_simulations = 250
            
            st.markdown("**💰 Periodic Contributions/Withdrawals:**")
            
            # Periodic contributions/withdrawals
            periodic_contribution = st.number_input(
                "Periodic Cash Flow ($):",
                min_value=-50000,
                max_value=50000,
                value=500,
                step=100,
                help="Amount to add (+) or withdraw (-) from portfolio. Positive = contributions, Negative = withdrawals"
            )
            
            contribution_frequency = st.selectbox(
                "Cash Flow Frequency:",
                ["monthly", "quarterly", "yearly"],
                index=0,
                help="How often to add or withdraw money from the portfolio"
            )
            
            # Run simulation button
            run_simulation = st.button(
                "🚀 Run Simulation",
                type="primary",
                disabled=total_allocation <= 0 or len(selected_assets) < 1,
                use_container_width=True
            )
        
        # Store settings in session state
        if selected_assets and total_allocation > 0:
            st.session_state.portfolio_settings = {
                'allocations': allocations,
                'selected_assets': selected_assets,
                'initial_investment': initial_investment,
                'years_to_project': years_to_project,
                'num_simulations': num_simulations,
                'periodic_contribution': periodic_contribution,
                'contribution_frequency': contribution_frequency
            }
        
        # Run simulation and show results
        if 'portfolio_settings' in st.session_state and run_simulation:
            settings = st.session_state.portfolio_settings
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Calculate optimal historical data range
                # Use all available historical data up to the present for better statistical accuracy
                # This is NOT look-ahead bias - we're using historical patterns to simulate futures
                historical_end_date = datetime.now().strftime('%Y-%m-%d')
                # Start date: Use maximum available history for robust Monte Carlo simulation
                historical_start_date = '1920-01-01'
                
                # Fetch historical data
                status_text.text("Fetching historical data...")
                tickers = list(settings['allocations'].keys())
                
                def progress_callback(current, total):
                    progress = current / total * 0.3  # First 30% for data fetching
                    progress_bar.progress(progress)
                    status_text.text(f"Fetching data: {current}/{total} assets")
                
                historical_data = simulator.get_historical_data(
                    tickers,
                    historical_start_date,
                    historical_end_date,
                    progress_callback
                )
                
                # Validate we have sufficient data for simulation
                valid_tickers = [ticker for ticker, data in historical_data.items() 
                               if not data.empty and len(data) > 30]
                
                if len(valid_tickers) == 0:
                    st.error("❌ No valid historical data found for the selected assets. Please try different assets.")
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                # Run backtest for baseline
                status_text.text("Running baseline backtest...")
                progress_bar.progress(0.4)
                
                backtest_results = simulator.calculate_portfolio_returns(
                    settings['allocations'],
                    historical_data,
                    settings['initial_investment'],
                    settings.get('periodic_contribution', 0),
                    settings.get('contribution_frequency', 'monthly')
                )
                
                # Calculate backtest metrics
                backtest_metrics = simulator.calculate_metrics(backtest_results)
                
                # Run Monte Carlo simulation
                status_text.text("Running Monte Carlo simulation...")
                
                def mc_progress_callback(current, total):
                    progress = 0.4 + (current / total * 0.6)  # 40% + remaining 60%
                    progress_bar.progress(progress)
                    status_text.text(f"Monte Carlo simulation: {current}/{total}")
                
                try:
                    mc_results = simulator.monte_carlo_simulation(
                        settings['allocations'],
                        historical_data,
                        settings['initial_investment'],
                        settings['years_to_project'],
                        settings['num_simulations'],
                        mc_progress_callback,
                        settings.get('periodic_contribution', 0),
                        settings.get('contribution_frequency', 'monthly')
                    )
                    
                    if not mc_results:
                        st.error("❌ Monte Carlo simulation failed. This may be due to insufficient historical data.")
                        progress_bar.empty()
                        status_text.empty()
                        return
                        
                except Exception as mc_error:
                    error_msg = str(mc_error)
                    if "insufficient historical data" in error_msg.lower() or "months" in error_msg:
                        # Extract useful info from error message
                        st.error(f"❌ **Insufficient Historical Data**")
                        st.error(f"**Error Details:** {error_msg}")
                    else:
                        st.error(f"❌ Monte Carlo simulation error: {error_msg}")
                    
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                progress_bar.empty()
                status_text.empty()
                
                if backtest_results is not None and not backtest_results.empty and mc_results:
                    
                    st.markdown("---")
                    st.markdown("## 🎲 Portfolio Simulation Results")

                    # Calculate expected cash flow over projection period
                    years = settings['years_to_project']
                    periodic_cash_flow = settings.get('periodic_contribution', 0)
                    
                    if settings.get('contribution_frequency', 'monthly') == 'monthly':
                        total_cash_flow = periodic_cash_flow * 12 * years
                    elif settings.get('contribution_frequency', 'monthly') == 'quarterly':
                        total_cash_flow = periodic_cash_flow * 4 * years
                    else:  # yearly
                        total_cash_flow = periodic_cash_flow * years
                    
                    expected_invested = settings['initial_investment'] + total_cash_flow
                    
                    # Monte Carlo results - focusing on best and worst case scenarios
                    st.markdown(f"**Portfolio Projections after {years} years ({settings['num_simulations']:,} simulations):**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Best Case Scenario
                        best_case_value = mc_results['percentile_95']
                        best_case_gain = best_case_value - expected_invested
                        best_case_return = (best_case_gain / expected_invested * 100) if expected_invested > 0 else 0
                        # Calculate annualized return: (final_value / initial_invested)^(1/years) - 1
                        best_case_annualized = ((best_case_value / expected_invested) ** (1 / years) - 1) * 100 if expected_invested > 0 and years > 0 else 0
                        
                        st.markdown("""
                        <div style="padding: 1rem; border-radius: 10px; background-color: #28a745; color: white; text-align: center;">
                            <h3>🚀 Best Case Scenario</h3>
                            <h2>${:,.0f}</h2>
                            <p>95th Percentile</p>
                            <p>Total: ${:,.0f} ({:.1f}%)</p>
                            <p><strong>Annualized: {:.1f}%</strong></p>
                        </div>
                        """.format(best_case_value, best_case_gain, best_case_return, best_case_annualized), unsafe_allow_html=True)
                    
                    with col2:
                        # Expected/Median Value
                        median_value = mc_results['median']
                        median_gain = median_value - expected_invested
                        median_return = (median_gain / expected_invested * 100) if expected_invested > 0 else 0
                        # Calculate annualized return
                        median_annualized = ((median_value / expected_invested) ** (1 / years) - 1) * 100 if expected_invested > 0 and years > 0 else 0
                        
                        st.markdown("""
                        <div style="padding: 1rem; border-radius: 10px; background-color: #007bff; color: white; text-align: center;">
                            <h3>📊 Expected Outcome</h3>
                            <h2>${:,.0f}</h2>
                            <p>50th Percentile (Median)</p>
                            <p>Total: ${:,.0f} ({:.1f}%)</p>
                            <p><strong>Annualized: {:.1f}%</strong></p>
                        </div>
                        """.format(median_value, median_gain, median_return, median_annualized), unsafe_allow_html=True)
                    
                    with col3:
                        # Worst Case Scenario
                        worst_case_value = mc_results['percentile_5']
                        worst_case_gain = worst_case_value - expected_invested
                        worst_case_return = (worst_case_gain / expected_invested * 100) if expected_invested > 0 else 0
                        # Calculate annualized return
                        worst_case_annualized = ((worst_case_value / expected_invested) ** (1 / years) - 1) * 100 if expected_invested > 0 and years > 0 else 0
                        
                        # Determine if it's a loss or gain
                        result_type = "Loss" if worst_case_gain < 0 else "Gain"
                        
                        st.markdown("""
                        <div style="padding: 1rem; border-radius: 10px; background-color: #dc3545; color: white; text-align: center;">
                            <h3>⚠️ Worst Case Scenario</h3>
                            <h2>${:,.0f}</h2>
                            <p>5th Percentile</p>
                            <p>Total: ${:,.0f} ({:.1f}%)</p>
                            <p><strong>Annualized: {:.1f}%</strong></p>
                        </div>
                        """.format(worst_case_value, abs(worst_case_gain), abs(worst_case_return), worst_case_annualized), unsafe_allow_html=True)
                    
                    # Additional summary metrics
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Initial Investment",
                            f"${settings['initial_investment']:,.0f}",
                            help="Starting portfolio value"
                        )
                    
                    with col2:
                        st.metric(
                            "Total Expected Investment",
                            f"${expected_invested:,.0f}",
                            help="Initial + contributions - withdrawals over time"
                        )
                    
                    with col3:
                        prob_positive = sum(1 for x in mc_results['all_results'] if x > expected_invested) / len(mc_results['all_results']) * 100
                        st.metric(
                            "Probability of Gain",
                            f"{prob_positive:.1f}%",
                            help="Chance of positive returns vs total invested"
                        )
                    
                    with col4:
                        prob_loss = sum(1 for x in mc_results['all_results'] if x < expected_invested) / len(mc_results['all_results']) * 100
                        st.metric(
                            "Probability of Loss",
                            f"{prob_loss:.1f}%",
                            help="Chance of ending below total invested amount"
                        )
                    
                    # Additional annualized performance metrics
                    st.markdown("**📊 Annualized Performance Summary:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # Best case annualized return
                        best_case_annualized = ((best_case_value / expected_invested) ** (1 / years) - 1) * 100 if expected_invested > 0 and years > 0 else 0
                        st.metric(
                            "Best Case (Annual)",
                            f"{best_case_annualized:.1f}%",
                            help="95th percentile annualized return"
                        )
                    
                    with col2:
                        # Median annualized return
                        median_annualized = ((median_value / expected_invested) ** (1 / years) - 1) * 100 if expected_invested > 0 and years > 0 else 0
                        st.metric(
                            "Expected (Annual)",
                            f"{median_annualized:.1f}%",
                            help="Median annualized return"
                        )
                    
                    with col3:
                        # Worst case annualized return
                        worst_case_annualized = ((worst_case_value / expected_invested) ** (1 / years) - 1) * 100 if expected_invested > 0 and years > 0 else 0
                        st.metric(
                            "Worst Case (Annual)",
                            f"{worst_case_annualized:.1f}%",
                            help="5th percentile annualized return"
                        )
                    
                    with col4:
                        # Calculate average annualized return across all simulations
                        all_annualized = [((result / expected_invested) ** (1 / years) - 1) * 100 
                                        for result in mc_results['all_results'] 
                                        if expected_invested > 0 and years > 0]
                        avg_annualized = sum(all_annualized) / len(all_annualized) if all_annualized else 0
                        st.metric(
                            "Average (Annual)",
                            f"{avg_annualized:.1f}%",
                            help="Mean annualized return across all simulations"
                        )
                    
                    # Portfolio Evolution Chart
                    st.markdown("---")
                    st.markdown("**📈 Portfolio Evolution Over Time:**")
                    
                    if 'time_axis' in mc_results and 'best_case_series' in mc_results:
                        fig_evolution = go.Figure()
                        
                        # Add best case trajectory
                        fig_evolution.add_trace(go.Scatter(
                            x=mc_results['time_axis'],
                            y=mc_results['best_case_series'],
                            mode='lines',
                            name='Best Case (Actual Path)',
                            line=dict(color='green', width=3),
                            fill=None
                        ))
                        
                        # Add median case trajectory
                        fig_evolution.add_trace(go.Scatter(
                            x=mc_results['time_axis'],
                            y=mc_results['median_case_series'],
                            mode='lines',
                            name='Median Case (Actual Path)',
                            line=dict(color='blue', width=2),
                            fill=None
                        ))
                        
                        # Add worst case trajectory
                        fig_evolution.add_trace(go.Scatter(
                            x=mc_results['time_axis'],
                            y=mc_results['worst_case_series'],
                            mode='lines',
                            name='Worst Case (Actual Path)',
                            line=dict(color='red', width=3),
                            fill=None
                        ))
                        
                        # Add total invested reference line (if cash flow exists)
                        if settings.get('periodic_contribution', 0) != 0:
                            # Calculate cumulative investment over time
                            time_years = mc_results['time_axis']
                            cumulative_invested = []
                            
                            for t in time_years:
                                if settings.get('contribution_frequency', 'monthly') == 'monthly':
                                    cash_flow_per_year = settings.get('periodic_contribution', 0) * 12
                                elif settings.get('contribution_frequency', 'monthly') == 'quarterly':
                                    cash_flow_per_year = settings.get('periodic_contribution', 0) * 4
                                else:  # yearly
                                    cash_flow_per_year = settings.get('periodic_contribution', 0)
                                
                                total_invested_at_time = settings['initial_investment'] + (cash_flow_per_year * t)
                                cumulative_invested.append(max(0, total_invested_at_time))
                            
                            fig_evolution.add_trace(go.Scatter(
                                x=time_years,
                                y=cumulative_invested,
                                mode='lines',
                                name='Total Money Invested',
                                line=dict(color='orange', width=2, dash='dot'),
                                fill=None
                            ))
                        
                        fig_evolution.update_layout(
                            title=f"Portfolio Value Evolution - Best, Median, and Worst Case Scenarios ({years} Years)",
                            xaxis_title="Years",
                            yaxis_title="Portfolio Value ($)",
                            hovermode='x unified',
                            height=500,
                            legend=dict(x=0.02, y=0.98)
                        )
                        
                        st.plotly_chart(fig_evolution, use_container_width=True)
                        
                        # Add explanation
                        st.info("""
                        **Chart Explanation:**
                        - 🟢 **Green line**: Best case scenario - an actual simulated path that ended in the 95th percentile
                        - 🔵 **Blue line**: Median case scenario - an actual simulated path that ended near the median
                        - 🔴 **Red line**: Worst case scenario - an actual simulated path that ended in the 5th percentile
                        - 🟠 **Orange dotted line**: Total amount of money invested over time (positive cash flow adds, negative withdraws)
                        
                        These are real portfolio trajectories from the Monte Carlo simulation, showing how your portfolio value could evolve over time with daily market movements.
                        """)
                    else:
                        st.warning("Time series data not available. Please re-run the simulation.")
                    
                    # Monte Carlo Distribution Chart
                    st.markdown("---")
                    st.markdown("**� Monte Carlo Portfolio Value Distribution:**")
                    
                    fig_hist = go.Figure()
                    
                    fig_hist.add_trace(go.Histogram(
                        x=mc_results['all_results'],
                        nbinsx=50,
                        name='Portfolio Value',
                        opacity=0.7,
                        marker_color='lightblue'
                    ))
                    
                    # Add scenario lines
                    scenarios = [
                        (mc_results['percentile_5'], '5th Percentile (Worst Case)', 'red'),
                        (mc_results['median'], 'Median (Expected)', 'blue'), 
                        (mc_results['percentile_95'], '95th Percentile (Best Case)', 'green')
                    ]
                    
                    for value, label, color in scenarios:
                        fig_hist.add_vline(
                            x=value,
                            line_dash="dash",
                            line_color=color,
                            line_width=3,
                            annotation_text=f"{label}<br>${value:,.0f}",
                            annotation_position="top"
                        )
                    
                    # Add expected investment line
                    fig_hist.add_vline(
                        x=expected_invested,
                        line_dash="dot",
                        line_color="orange",
                        line_width=2,
                        annotation_text=f"Total Invested<br>${expected_invested:,.0f}",
                        annotation_position="bottom"
                    )
                    
                    fig_hist.update_layout(
                        title=f"Portfolio Value Distribution after {years} years ({settings['num_simulations']:,} simulations)",
                        xaxis_title="Portfolio Value ($)",
                        yaxis_title="Frequency",
                        height=500,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Store results
                    st.session_state.portfolio_results = backtest_results
                    st.session_state.portfolio_metrics = backtest_metrics
                    st.session_state.historical_data = historical_data
                    st.session_state.monte_carlo_results = mc_results
                
                else:
                    st.error("Unable to run simulation. Please check your settings and try again.")
            
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error running simulation: {str(e)}")
        
        elif 'portfolio_settings' not in st.session_state:
            st.info("� Configure your portfolio settings above and click 'Run Simulation' to see results.")
        
        else:
            st.info("Click 'Run Simulation' to see your portfolio projections.")
    
    with tab2:
        st.subheader("📊 Portfolio Analysis")
        
        if 'portfolio_settings' in st.session_state and 'historical_data' in st.session_state:
            settings = st.session_state.portfolio_settings
            historical_data = st.session_state.historical_data
            
            # Asset correlation analysis
            st.markdown("**🔗 Asset Correlation Matrix:**")
            
            tickers = list(settings['allocations'].keys())
            correlation_matrix = simulator.get_asset_correlation(tickers, historical_data)
            
            if not correlation_matrix.empty:
                # Create correlation heatmap
                fig = px.imshow(
                    correlation_matrix.values,
                    x=[settings['selected_assets'][i] for i, ticker in enumerate(tickers) if ticker in correlation_matrix.columns],
                    y=[settings['selected_assets'][i] for i, ticker in enumerate(tickers) if ticker in correlation_matrix.columns],
                    color_continuous_scale='RdBu',
                    aspect="auto",
                    color_continuous_midpoint=0
                )
                
                fig.update_layout(
                    title="Asset Correlation Matrix",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation insights
                st.markdown("**🔍 Correlation Insights:**")
                
                high_corr_pairs = []
                low_corr_pairs = []
                
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        asset1 = correlation_matrix.columns[i]
                        asset2 = correlation_matrix.columns[j]
                        
                        # Find asset names
                        name1 = next((name for name, ticker in simulator.AVAILABLE_ASSETS.items() if ticker == asset1), asset1)
                        name2 = next((name for name, ticker in simulator.AVAILABLE_ASSETS.items() if ticker == asset2), asset2)
                        
                        if corr_value > 0.7:
                            high_corr_pairs.append((name1, name2, corr_value))
                        elif corr_value < 0.3:
                            low_corr_pairs.append((name1, name2, corr_value))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if high_corr_pairs:
                        st.markdown("**🔴 High Correlation (>70%):**")
                        for asset1, asset2, corr in high_corr_pairs[:3]:
                            st.write(f"• {asset1} ↔ {asset2}: {corr:.2f}")
                        st.caption("These assets tend to move together")
                    else:
                        st.markdown("**🟢 Good Diversification:**")
                        st.write("No highly correlated asset pairs found")
                
                with col2:
                    if low_corr_pairs:
                        st.markdown("**🟢 Low Correlation (<30%):**")
                        for asset1, asset2, corr in low_corr_pairs[:3]:
                            st.write(f"• {asset1} ↔ {asset2}: {corr:.2f}")
                        st.caption("These assets provide good diversification")
            
            # Portfolio optimization suggestion
            st.markdown("---")
            st.markdown("**⚡ Portfolio Optimization:**")
            
            if st.button("🎯 Suggest Optimized Allocation"):
                with st.spinner("Optimizing portfolio..."):
                    optimized_allocations = simulator.optimize_portfolio(tickers, historical_data)
                    
                    if optimized_allocations:
                        st.markdown("**Optimized Allocations (Equal Risk Contribution):**")
                        
                        # Compare current vs optimized
                        comparison_data = []
                        for ticker in tickers:
                            asset_name = next((name for name, t in simulator.AVAILABLE_ASSETS.items() if t == ticker), ticker)
                            current_alloc = settings['allocations'].get(ticker, 0)
                            optimized_alloc = optimized_allocations.get(ticker, 0)
                            
                            comparison_data.append({
                                'Asset': asset_name,
                                'Current (%)': current_alloc,
                                'Optimized (%)': optimized_alloc,
                                'Difference': optimized_alloc - current_alloc
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Style the dataframe
                        def highlight_changes(row):
                            if row['Difference'] > 5:
                                return ['background-color: lightgreen'] * len(row)
                            elif row['Difference'] < -5:
                                return ['background-color: lightcoral'] * len(row)
                            else:
                                return [''] * len(row)
                        
                        styled_df = comparison_df.style.apply(highlight_changes, axis=1)
                        st.dataframe(styled_df, use_container_width=True)
                        
                        st.caption("🟢 Green: Increase allocation, 🔴 Red: Decrease allocation")
        
        else:
            st.info("👈 Run a backtest first to enable portfolio analysis.")

def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    # Initialize page in session state if not present
    if 'page' not in st.session_state:
        st.session_state.page = "Overview"
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🏦 Finance Tool")
        
        # Navigation - sync with session state
        page_options = ["Overview", "Net Worth", "Portfolio", "Transactions", "Settings"]
        current_index = page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
        
        page = st.selectbox(
            "Choose a section:",
            page_options,
            index=current_index
        )
        
        # Update session state when selectbox changes
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats if data is available
        if not st.session_state.net_worth_data.empty:
            st.markdown("**📈 Net Worth Data**: ✅ Available")
        else:
            st.markdown("**📈 Net Worth Data**: ❌ Not loaded")
        
        if not st.session_state.transactions_data.empty:
            st.markdown(f"**💳 Transactions**: ✅ {len(st.session_state.transactions_data)} records")
        else:
            st.markdown("**💳 Transactions**: ❌ Not loaded")
        
        # Portfolio status
        if 'portfolio_results' in st.session_state:
            st.markdown("**📊 Portfolio**: ✅ Backtest completed")
        else:
            st.markdown("**📊 Portfolio**: ❌ No backtest run")
        
        st.markdown("---")
        
        # Information section
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            **Comprehensive Finance Tool** combines:
            
            - 📈 **Net Worth Tracking**: Monitor your assets across multiple accounts and currencies
            - � **Portfolio Simulation**: Backtest investment strategies with real Yahoo Finance data
            - �💳 **Transaction Analysis**: Analyze spending patterns from bank statements
            - 🎯 **Financial Overview**: Get insights into your overall financial health
            
            **Features:**
            - Multi-currency support (CHF, EUR, USD) - unified to EUR
            - PDF bank statement processing
            - Interactive charts and analytics
            - Export capabilities
            - Financial health indicators
            """)
    
    # Main content based on selected page
    if st.session_state.page == "Overview":
        overview_dashboard()
    elif st.session_state.page == "Net Worth":
        net_worth_dashboard()
    elif st.session_state.page == "Portfolio":
        portfolio_dashboard()
    elif st.session_state.page == "Transactions":
        transaction_analysis_dashboard()
    elif st.session_state.page == "Settings":
        st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
        
        st.subheader("🗃️ Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Clear Data**")
            if st.button("Clear Net Worth Data"):
                st.session_state.net_worth_data = pd.DataFrame()
                st.success("Net worth data cleared!")
                st.rerun()
            
            if st.button("Clear Transaction Data"):
                st.session_state.transactions_data = pd.DataFrame()
                st.session_state.analyzer = None
                st.success("Transaction data cleared!")
                st.rerun()
        
        with col2:
            st.markdown("**Export All Data**")
            
            if st.button("Export All Data as ZIP"):
                import zipfile
                import io
                
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    if not st.session_state.net_worth_data.empty:
                        csv_data = st.session_state.net_worth_data.to_csv(index=False)
                        zip_file.writestr("net_worth_data.csv", csv_data)
                    
                    if not st.session_state.transactions_data.empty:
                        csv_data = st.session_state.transactions_data.to_csv(index=False)
                        zip_file.writestr("transactions_data.csv", csv_data)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="Download ZIP File",
                    data=zip_buffer,
                    file_name=f"finance_data_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip"
                )
        
        st.markdown("---")
        
        st.subheader("🎨 Application Info")
        st.info("""
        **Version**: 1.0.0  
        **Created**: December 2025  
        **Technologies**: Streamlit, Plotly, pandas, yfinance  
        
        This comprehensive financial tool helps you track your net worth and analyze spending patterns in one integrated application.
        """)

if __name__ == "__main__":
    main()
