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
            'IBKR Account (CHF)': np.random.normal(15000, 3000, len(dates)).round(2),
            'Kutxabank Account (EUR)': np.random.normal(1500, 500, len(dates)).round(2)
        }
        return pd.DataFrame(sample_data)

def get_forex_rate(from_currency: str, to_currency: str) -> float:
    """Get current forex rate using yfinance."""
    if from_currency == to_currency:
        return 1.0
    
    try:
        ticker = f"{from_currency}{to_currency}=X"
        forex_data = yf.download(ticker, period="1d", interval="1d")
        if not forex_data.empty:
            return forex_data['Close'].iloc[-1]
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

def convert_to_chf(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all currencies to CHF for unified tracking."""
    df_converted = df.copy()
    
    # Get current EUR to CHF rate
    eur_to_chf = get_forex_rate('EUR', 'CHF')
    
    # Convert EUR columns to CHF
    for col in df_converted.columns:
        if 'EUR' in col:
            chf_col_name = col.replace('EUR', 'CHF')
            df_converted[chf_col_name] = df_converted[col] * eur_to_chf
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
            help="CSV should have Date column and account columns with currency indicators"
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
                ibkr_chf = st.number_input("IBKR Account (CHF)", value=0.0, step=100.0)
                kutxa_eur = st.number_input("Kutxabank Account (EUR)", value=0.0, step=100.0)
                
                if st.form_submit_button("Add Entry"):
                    new_entry = pd.DataFrame({
                        'Date': [entry_date.strftime('%Y-%m-%d')],
                        'UBS Account (CHF)': [ubs_chf],
                        'IBKR Account (CHF)': [ibkr_chf],
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
        
        # Convert to CHF for unified tracking
        df_chf = convert_to_chf(df)
        
        # Calculate total net worth
        value_columns = [col for col in df_chf.columns if col != 'Date' and 'CHF' in col]
        df_chf['Total_CHF'] = df_chf[value_columns].sum(axis=1)
        
        # Current metrics
        latest = df_chf.iloc[-1]
        if len(df_chf) > 1:
            previous = df_chf.iloc[-2]
            change = latest['Total_CHF'] - previous['Total_CHF']
            change_pct = (change / previous['Total_CHF']) * 100
        else:
            change = 0
            change_pct = 0
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Total Net Worth",
                value=f"CHF {latest['Total_CHF']:,.2f}",
                delta=f"CHF {change:,.2f}"
            )
        
        with col2:
            ubs_value = latest.get('UBS Account (CHF)', 0)
            st.metric(
                label="🏦 UBS Account",
                value=f"CHF {ubs_value:,.2f}"
            )
        
        with col3:
            ibkr_value = latest.get('IBKR Account (CHF)', 0)
            st.metric(
                label="📊 IBKR Account",
                value=f"CHF {ibkr_value:,.2f}"
            )
        
        with col4:
            kutxa_value = latest.get('Kutxabank Account (CHF)', 0)
            st.metric(
                label="🏛️ Kutxabank Account",
                value=f"CHF {kutxa_value:,.2f}"
            )
        
        # Charts section
        st.markdown("---")
        
        # Net worth trend chart
        st.subheader("📈 Net Worth Trend")
        fig_trend = px.line(
            df_chf, 
            x='Date', 
            y='Total_CHF',
            title='Net Worth Over Time',
            labels={'Total_CHF': 'Total Net Worth (CHF)', 'Date': 'Date'}
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Account breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏦 Account Breakdown")
            account_values = {col.replace(' (CHF)', ''): latest[col] for col in value_columns}
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
                    x=df_chf['Date'],
                    y=df_chf[col],
                    mode='lines+markers',
                    name=col.replace(' (CHF)', '')
                ))
            fig_multi.update_layout(
                title='Individual Account Trends',
                xaxis_title='Date',
                yaxis_title='Value (CHF)',
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
        Date,UBS Account (CHF),IBKR Account (CHF),Kutxabank Account (EUR)
        2025-01-31,10000.00,15000.00,1500.00
        2025-02-28,10500.00,15500.00,1550.00
        ```
        """)

def transaction_analysis_dashboard():
    """Transaction Analysis Dashboard."""
    st.markdown('<h1 class="main-header">💳 Transaction Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Transaction Data")
        
        # PDF upload for bank statements
        uploaded_pdf = st.file_uploader(
            "Upload Bank Statement (PDF)",
            type=['pdf'],
            help="Upload your bank statement PDF for automatic transaction analysis"
        )
        
        # CSV upload for manual transaction data
        uploaded_csv = st.file_uploader(
            "Upload Transaction Data (CSV)",
            type=['csv'],
            help="Upload CSV with columns: date, description, amount, category (optional)"
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
    
    elif uploaded_csv is not None:
        try:
            transactions_df = pd.read_csv(uploaded_csv)
            
            # Validate required columns
            required_cols = ['date', 'description', 'amount']
            if all(col in transactions_df.columns for col in required_cols):
                transactions_df['date'] = pd.to_datetime(transactions_df['date'])
                st.session_state.transactions_data = transactions_df
                st.success(f"Successfully loaded {len(transactions_df)} transactions!")
            else:
                st.error(f"CSV must contain columns: {', '.join(required_cols)}")
        
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
    
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
            st.metric("💵 Total Income", f"CHF {income:,.2f}")
        
        with col2:
            st.metric("💸 Total Expenses", f"CHF {expenses:,.2f}")
        
        with col3:
            st.metric("💰 Net Savings", f"CHF {net_savings:,.2f}")
        
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
            yaxis_title='Amount (CHF)',
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
        st.info("📊 No transaction data available. Please upload a bank statement PDF or CSV file.")
        st.markdown("""
        **Supported formats:**
        
        **PDF**: Bank statements from UBS and other major banks
        
        **CSV**: Should contain at minimum:
        ```
        date,description,amount,category
        2025-12-01,Grocery Shopping,-50.00,Food & Dining
        2025-12-01,Salary,3000.00,Income
        ```
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
            df_nw_chf = convert_to_chf(df_nw)
            value_columns = [col for col in df_nw_chf.columns if col != 'Date' and 'CHF' in col]
            df_nw_chf['Total_CHF'] = df_nw_chf[value_columns].sum(axis=1)
            
            latest_nw = df_nw_chf.iloc[-1]['Total_CHF']
            
            with col1:
                st.metric(
                    label="💰 Current Net Worth",
                    value=f"CHF {latest_nw:,.2f}"
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
                    value=f"CHF {monthly_income:,.2f}"
                )
            
            with col3:
                st.metric(
                    label="📉 Monthly Expenses",
                    value=f"CHF {monthly_expenses:,.2f}"
                )
            
            with col4:
                st.metric(
                    label="💵 Monthly Savings",
                    value=f"CHF {monthly_savings:,.2f}",
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
                    df_nw_chf,
                    x='Date',
                    y='Total_CHF',
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
                if len(df_nw_chf) > 1:
                    start_value = df_nw_chf.iloc[0]['Total_CHF']
                    end_value = df_nw_chf.iloc[-1]['Total_CHF']
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
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Portfolio Builder", "📈 Backtest Results", "🎲 Monte Carlo", "📊 Analysis"])
    
    with tab1:
        st.subheader("🎯 Build Your Portfolio")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Select Assets:**")
            
            # Asset selection
            available_assets = list(simulator.AVAILABLE_ASSETS.keys())
            selected_assets = st.multiselect(
                "Choose assets for your portfolio:",
                available_assets,
                default=["S&P 500", "MSCI World", "Gold"],
                help="Select 2-10 assets for your portfolio"
            )
            
            if selected_assets:
                st.markdown("**Set Allocations (%):**")
                allocations = {}
                
                # Create allocation sliders
                col_count = min(len(selected_assets), 3)
                cols = st.columns(col_count)
                
                for i, asset in enumerate(selected_assets):
                    with cols[i % col_count]:
                        allocations[simulator.AVAILABLE_ASSETS[asset]] = st.slider(
                            asset,
                            0, 100, 
                            100 // len(selected_assets),  # Equal allocation by default
                            step=1,
                            key=f"allocation_{asset}"
                        )
                
                # Check if allocations sum to 100%
                total_allocation = sum(allocations.values())
                if total_allocation != 100:
                    st.warning(f"⚠️ Total allocation: {total_allocation}%. Please adjust to 100%.")
                else:
                    st.success(f"✅ Total allocation: {total_allocation}%")
        
        with col2:
            st.markdown("**Backtest Settings:**")
            
            # Date range selection
            end_date = st.date_input(
                "End Date:",
                datetime.now().date(),
                max_value=datetime.now().date()
            )
            
            start_date = st.date_input(
                "Start Date:",
                datetime.now().date() - timedelta(days=5*365),  # 5 years default
                max_value=end_date
            )
            
            # Initial investment
            initial_investment = st.number_input(
                "Initial Investment ($):",
                min_value=1000,
                max_value=10000000,
                value=10000,
                step=1000
            )
            
            # Rebalancing frequency
            rebalance_freq = st.selectbox(
                "Rebalancing:",
                ["monthly", "quarterly", "yearly", "none"],
                index=0
            )
            
            # Run backtest button
            run_backtest = st.button(
                "🚀 Run Backtest",
                type="primary",
                disabled=total_allocation != 100 or len(selected_assets) < 1,
                use_container_width=True
            )
        
        # Store settings in session state
        if selected_assets and total_allocation == 100:
            st.session_state.portfolio_settings = {
                'allocations': allocations,
                'selected_assets': selected_assets,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'initial_investment': initial_investment,
                'rebalance_freq': rebalance_freq
            }
    
    with tab2:
        st.subheader("📈 Backtest Results")
        
        if 'portfolio_settings' in st.session_state and run_backtest:
            settings = st.session_state.portfolio_settings
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Fetch data
                status_text.text("Fetching historical data...")
                tickers = list(settings['allocations'].keys())
                
                def progress_callback(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Fetching data: {current}/{total} assets")
                
                historical_data = simulator.get_historical_data(
                    tickers,
                    settings['start_date'],
                    settings['end_date'],
                    progress_callback
                )
                
                # Calculate portfolio performance
                status_text.text("Calculating portfolio performance...")
                results = simulator.calculate_portfolio_returns(
                    settings['allocations'],
                    historical_data,
                    settings['initial_investment'],
                    settings['rebalance_freq']
                )
                
                # Calculate metrics
                metrics = simulator.calculate_metrics(results)
                
                progress_bar.empty()
                status_text.empty()
                
                if not results.empty:
                    # Display metrics
                    st.markdown("**📊 Portfolio Performance:**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Return",
                            f"{metrics.get('Total Return (%)', 0):.2f}%",
                            help="Total return over the entire period"
                        )
                        st.metric(
                            "Annualized Return",
                            f"{metrics.get('Annualized Return (%)', 0):.2f}%",
                            help="Average yearly return"
                        )
                    
                    with col2:
                        st.metric(
                            "Volatility",
                            f"{metrics.get('Volatility (%)', 0):.2f}%",
                            help="Annual volatility (risk measure)"
                        )
                        st.metric(
                            "Sharpe Ratio",
                            f"{metrics.get('Sharpe Ratio', 0):.2f}",
                            help="Risk-adjusted return measure"
                        )
                    
                    with col3:
                        st.metric(
                            "Max Drawdown",
                            f"{metrics.get('Maximum Drawdown (%)', 0):.2f}%",
                            help="Largest peak-to-trough decline"
                        )
                        st.metric(
                            "Win Rate",
                            f"{metrics.get('Win Rate (%)', 0):.1f}%",
                            help="Percentage of positive days"
                        )
                    
                    with col4:
                        st.metric(
                            "Final Value",
                            f"${metrics.get('Final Value', 0):,.0f}",
                            help="Portfolio value at the end"
                        )
                        st.metric(
                            "Best/Worst Day",
                            f"+{metrics.get('Best Day (%)', 0):.1f}% / {metrics.get('Worst Day (%)', 0):.1f}%",
                            help="Best and worst single day returns"
                        )
                    
                    # Portfolio value chart
                    st.markdown("**📈 Portfolio Growth:**")
                    
                    fig = go.Figure()
                    
                    # Add portfolio value line
                    fig.add_trace(go.Scatter(
                        x=results['Date'],
                        y=results['Portfolio_Value'],
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Add individual asset comparisons if requested
                    if st.checkbox("Show individual asset performance"):
                        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                        for i, asset in enumerate(settings['selected_assets']):
                            ticker = simulator.AVAILABLE_ASSETS[asset]
                            if f'{ticker}_Value' in results.columns:
                                fig.add_trace(go.Scatter(
                                    x=results['Date'],
                                    y=results[f'{ticker}_Value'],
                                    mode='lines',
                                    name=asset,
                                    line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
                                    opacity=0.7
                                ))
                    
                    fig.update_layout(
                        title="Portfolio Value Over Time",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        hovermode='x',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store results for other tabs
                    st.session_state.portfolio_results = results
                    st.session_state.portfolio_metrics = metrics
                    st.session_state.historical_data = historical_data
                
                else:
                    st.error("No data available for the selected period and assets.")
            
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error running backtest: {str(e)}")
        
        elif 'portfolio_settings' not in st.session_state:
            st.info("👈 Configure your portfolio in the Portfolio Builder tab first.")
        
        else:
            st.info("Click 'Run Backtest' in the Portfolio Builder tab to see results.")
    
    with tab3:
        st.subheader("🎲 Monte Carlo Simulation")
        
        if 'portfolio_settings' in st.session_state and 'historical_data' in st.session_state:
            settings = st.session_state.portfolio_settings
            historical_data = st.session_state.historical_data
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                years_to_project = st.slider(
                    "Years to project:",
                    1, 30, 10,
                    help="Number of years to simulate forward"
                )
                
                num_simulations = st.select_slider(
                    "Number of simulations:",
                    options=[100, 250, 500, 1000, 2000],
                    value=1000,
                    help="More simulations = more accurate results"
                )
            
            with col2:
                if st.button("🎲 Run Monte Carlo", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current, total):
                        progress = current / total
                        progress_bar.progress(progress)
                        status_text.text(f"Running simulation: {current}/{total}")
                    
                    try:
                        mc_results = simulator.monte_carlo_simulation(
                            settings['allocations'],
                            historical_data,
                            settings['initial_investment'],
                            years_to_project,
                            num_simulations,
                            progress_callback
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if mc_results:
                            # Display results
                            st.markdown("**🎯 Projected Portfolio Value:**")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Expected Value",
                                    f"${mc_results['mean']:,.0f}",
                                    help="Average outcome across all simulations"
                                )
                                st.metric(
                                    "Median Value",
                                    f"${mc_results['median']:,.0f}",
                                    help="Middle value (50th percentile)"
                                )
                            
                            with col2:
                                st.metric(
                                    "Best Case (95th)",
                                    f"${mc_results['percentile_95']:,.0f}",
                                    help="95% chance of being below this value"
                                )
                                st.metric(
                                    "Worst Case (5th)",
                                    f"${mc_results['percentile_5']:,.0f}",
                                    help="95% chance of being above this value"
                                )
                            
                            with col3:
                                st.metric(
                                    "Standard Deviation",
                                    f"${mc_results['std']:,.0f}",
                                    help="Measure of variability"
                                )
                                prob_positive = sum(1 for x in mc_results['all_results'] if x > settings['initial_investment']) / len(mc_results['all_results']) * 100
                                st.metric(
                                    "Prob. of Gain",
                                    f"{prob_positive:.1f}%",
                                    help="Probability of positive returns"
                                )
                            
                            # Distribution histogram
                            st.markdown("**📊 Return Distribution:**")
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Histogram(
                                x=mc_results['all_results'],
                                nbinsx=50,
                                name='Portfolio Value',
                                opacity=0.7
                            ))
                            
                            # Add percentile lines
                            for percentile, label in [(5, '5th'), (50, 'Median'), (95, '95th')]:
                                value = mc_results[f'percentile_{percentile}'] if percentile != 50 else mc_results['median']
                                fig.add_vline(
                                    x=value,
                                    line_dash="dash",
                                    annotation_text=f"{label}: ${value:,.0f}"
                                )
                            
                            fig.update_layout(
                                title=f"Portfolio Value Distribution ({years_to_project} years, {num_simulations:,} simulations)",
                                xaxis_title="Portfolio Value ($)",
                                yaxis_title="Frequency",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"Error running Monte Carlo simulation: {str(e)}")
        
        else:
            st.info("👈 Run a backtest first to enable Monte Carlo simulation.")
    
    with tab4:
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
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🏦 Finance Tool")
        
        # Navigation
        page = st.selectbox(
            "Choose a section:",
            ["Overview", "Net Worth", "Portfolio", "Transactions", "Settings"],
            index=0
        )
        
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
            - Multi-currency support (CHF, EUR, USD)
            - PDF bank statement processing
            - Interactive charts and analytics
            - Export capabilities
            - Financial health indicators
            """)
    
    # Main content based on selected page
    if page == "Overview":
        overview_dashboard()
    elif page == "Net Worth":
        net_worth_dashboard()
    elif page == "Portfolio":
        portfolio_dashboard()
    elif page == "Transactions":
        transaction_analysis_dashboard()
    elif page == "Settings":
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
