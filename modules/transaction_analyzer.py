#!/usr/bin/env python3
"""
Bank Transaction Analyzer
Analyzes bank transaction PDF to calculate savings and spending patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdfplumber
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import os
import streamlit as st


class TransactionAnalyzer:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.transactions = pd.DataFrame()
        
    def extract_text_from_pdf(self) -> str:
        """Extract all text content from the PDF."""
        text = ""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
        return text
    
    def parse_transactions(self, text: str) -> pd.DataFrame:
        """
        Parse transaction data from extracted text.
        Enhanced parser for UBS bank format and other common formats.
        """
        lines = text.split('\n')
        transactions = []
        
        # Skip until we find actual transaction data
        transaction_started = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip header and account info lines
            if any(skip_word in line.lower() for skip_word in [
                'ubs switzerland', 'po box', 'www.ubs', 'monsieur', 'route de', 
                'iban', 'bic', 'created on', 'account transactions', 'filter criteria',
                'booking amount', 'time period', 'tradedate', 'bookingdate description',
                'displayed in assets', 'page', '---'
            ]):
                continue
            
            # Look for closing balance to stop parsing
            if 'closingbalance' in line.lower() or 'closing balance' in line.lower():
                continue
                
            # Skip reference lines (like card payment references)
            if re.match(r'^\d{8}-\d{3}/\d{2};', line) or line.startswith('20813217-007/28;'):
                continue
                
            # UBS format: Date Description Amount Date Balance
            # Example: 14.12.2025 ALIMENTATIONDELAGARE -9.50 14.12.2025 7,741.08
            ubs_pattern = r'^(\d{2}\.\d{2}\.\d{4})\s+([A-Za-z].*?)\s+([+-]?[\d,]+\.?\d*)\s+\d{2}\.\d{2}\.\d{4}\s+[\d,]+\.?\d*'
            
            match = re.match(ubs_pattern, line)
            if match:
                try:
                    date_str, description, amount_str = match.groups()
                    
                    # Clean description (remove extra spaces and weird characters)
                    description = re.sub(r'\s+', ' ', description).strip()
                    
                    # Clean amount (remove commas, handle Swiss number format)
                    amount_str = amount_str.replace(',', '').replace("'", "")
                    amount = float(amount_str)
                    
                    transactions.append({
                        'date': date_str,
                        'amount': amount,
                        'description': description,
                        'raw_line': line
                    })
                    
                except (ValueError, AttributeError) as e:
                    continue
            
            # Alternative pattern for some UBS transactions that might be formatted differently
            elif re.match(r'^\d{2}\.\d{2}\.\d{4}', line) and any(char.isdigit() for char in line) and '-' in line:
                # Try to extract date, description and amount more flexibly
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        date_str = parts[0]
                        # Look for amount (contains decimal point and potentially minus sign)
                        amount_str = None
                        amount_idx = -1
                        
                        for i, part in enumerate(parts):
                            if re.match(r'^[+-]?[\d,]+\.?\d*$', part.replace(',', '').replace("'", "")):
                                amount_str = part
                                amount_idx = i
                                break
                        
                        if amount_str and amount_idx > 1:
                            # Description is everything between date and amount
                            description = ' '.join(parts[1:amount_idx])
                            
                            # Clean amount
                            amount_str = amount_str.replace(',', '').replace("'", "")
                            amount = float(amount_str)
                            
                            transactions.append({
                                'date': date_str,
                                'amount': amount,
                                'description': description,
                                'raw_line': line
                            })
                    except (ValueError, IndexError):
                        continue
        
        if not transactions:
            st.warning("No transactions found. Please check the PDF format.")
            return pd.DataFrame()
        
        df = pd.DataFrame(transactions)
        
        # Convert date column (UBS uses DD.MM.YYYY format)
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        except:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        
        # Remove rows where date parsing failed
        df = df.dropna(subset=['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        st.success(f"Successfully parsed {len(df)} transactions from PDF")
        
        return df
    
    def categorize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize transactions based on description."""
        if df.empty:
            return df
            
        def get_category(description):
            description = description.lower()
            
            # Income categories (Swiss/UBS specific)
            if any(word in description for word in ['neuralconcept', 'salary', 'payroll', 'deposit', 'income', 'transfer in']):
                return 'Income'
            
            # Food & Dining
            elif any(word in description for word in ['migros', 'coop', 'aldi', 'lidl', 'alimentation', 'restaurant', 'food', 'burger', 'taco', 'cafe', 'coffee', 'holycow', 'sumup', 'toogoodtogo']):
                return 'Food & Dining'
            
            # Transportation
            elif any(word in description for word in ['sbb', 'easyride', 'mobile', 'ticket', 'parking', 'tamoit', 'gas', 'fuel', 'uber', 'taxi', 'transport']):
                return 'Transportation'
            
            # Shopping & Retail
            elif any(word in description for word in ['ikea', 'jumbo', 'digitec', 'galaxus', 'dosenbach', 'amazon', 'shopping', 'store', 'retail']):
                return 'Shopping'
            
            # Healthcare & Insurance
            elif any(word in description for word in ['visana', 'calingo', 'insurance', 'medical', 'health', 'pharmacy', 'doctor']):
                return 'Healthcare & Insurance'
            
            # Housing & Utilities
            elif any(word in description for word in ['rent', 'mortgage', 'commune', 'utilities', 'electric', 'water']):
                return 'Housing & Utilities'
            
            # Entertainment & Lifestyle
            elif any(word in description for word in ['netflix', 'spotify', 'entertainment', 'movie', 'gym', 'elevate', 'escape']):
                return 'Entertainment & Lifestyle'
            
            # Travel
            elif any(word in description for word in ['easyjet', 'iberia', 'hotel', 'etcar', 'hire', 'airport']):
                return 'Travel'
            
            # Financial Services & Fees
            elif any(word in description for word in ['interactive', 'brokers', 'revolut', 'fee', 'charge', 'penalty', 'balance closing']):
                return 'Financial Services'
            
            # Personal Transfers
            elif any(word in description for word in ['helfer', 'errasti', 'odriozola', 'abad', 'rocamora', 'urwyler', 'kiener']):
                return 'Personal Transfers'
            
            # Telecommunications
            elif any(word in description for word in ['sunrise', 'yallo', 'mobile', 'phone']):
                return 'Telecommunications'
            
            # Government & Official
            elif any(word in description for word in ['douane', 'office', 'population', 'commune', 'pax', 'versicherung']):
                return 'Government & Official'
            
            # Cash & ATM
            elif any(word in description for word in ['atm', 'withdrawal', 'cash', 'quartier', 'cedres']):
                return 'Cash & Other'
            
            else:
                return 'Other'
        
        df['category'] = df['description'].apply(get_category)
        return df
    
    def analyze_transactions(self):
        """Main analysis function."""
        # Extract text
        text = self.extract_text_from_pdf()
        if not text:
            st.error("Failed to extract text from PDF")
            return None
        
        # Parse transactions
        self.transactions = self.parse_transactions(text)
        if self.transactions.empty:
            st.error("No transactions could be parsed from the PDF")
            return None
        
        # Categorize transactions
        self.transactions = self.categorize_transactions(self.transactions)
        
        return self.transactions
    
    def calculate_savings_and_spending(self) -> Dict:
        """Calculate total income, expenses, and savings."""
        if self.transactions.empty:
            return {}
        
        # Separate income and expenses
        income = self.transactions[self.transactions['amount'] > 0]['amount'].sum()
        expenses = abs(self.transactions[self.transactions['amount'] < 0]['amount'].sum())
        savings = income - expenses
        
        results = {
            'total_income': income,
            'total_expenses': expenses,
            'net_savings': savings,
            'savings_rate': (savings / income * 100) if income > 0 else 0
        }
        
        return results
    
    def spending_by_category(self) -> pd.DataFrame:
        """Analyze spending by category."""
        if self.transactions.empty:
            return pd.DataFrame()
        
        expenses = self.transactions[self.transactions['amount'] < 0].copy()
        expenses['amount'] = abs(expenses['amount'])
        
        category_spending = expenses.groupby('category')['amount'].agg(['sum', 'count']).round(2)
        category_spending.columns = ['total_spent', 'transaction_count']
        category_spending = category_spending.sort_values('total_spent', ascending=False)
        
        return category_spending
    
    def monthly_spending_trend(self) -> pd.DataFrame:
        """Analyze monthly spending trends."""
        if self.transactions.empty:
            return pd.DataFrame()
        
        self.transactions['year_month'] = self.transactions['date'].dt.to_period('M')
        
        monthly_income = self.transactions[self.transactions['amount'] > 0].groupby('year_month')['amount'].sum()
        monthly_expenses = abs(self.transactions[self.transactions['amount'] < 0].groupby('year_month')['amount'].sum())
        
        monthly_df = pd.DataFrame({
            'income': monthly_income,
            'expenses': monthly_expenses,
        }).fillna(0)
        
        monthly_df['savings'] = monthly_df['income'] - monthly_df['expenses']
        
        return monthly_df
