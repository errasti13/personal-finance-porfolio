# Comprehensive Finance Tool

A powerful, all-in-one financial management application built with Streamlit that combines net worth tracking and transaction analysis in a single, intuitive interface.

## 🌟 Features

### 📈 Net Worth Tracking
- **Multi-Account Support**: Track assets across multiple bank accounts and investment platforms
- **Multi-Currency Support**: Handle CHF, EUR, USD with automatic currency conversion
- **Interactive Charts**: Visualize your net worth growth over time
- **Account Breakdown**: See how your wealth is distributed across different accounts
- **Data Management**: Import/export data, manual entry, and persistent storage

### 💳 Transaction Analysis
- **PDF Processing**: Automatically extract transactions from bank statement PDFs (UBS and other formats)
- **CSV Import**: Load transaction data from CSV files
- **Smart Categorization**: Automatic categorization of transactions with customizable categories
- **Spending Insights**: Detailed analysis of spending patterns by category and time period
- **Cash Flow Analysis**: Track income vs expenses over time
- **Advanced Filtering**: Filter by date range, amount, category, and search terms

### 🎯 Financial Overview
- **Unified Dashboard**: Combined view of net worth and cash flow
- **Financial Health Indicators**: Automated assessment of savings rate, growth, and diversification
- **Key Metrics**: Quick overview of current financial status
- **Trend Analysis**: Identify patterns and trends across all your financial data

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
   ```bash
   cd comprehensive-finance-tool
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   This will install all required dependencies and create necessary directories.

3. **Start the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   The application will automatically open in your default web browser at `http://localhost:8501`

## 📊 Getting Started with Your Data

### Net Worth Tracking

1. **Load Sample Data**: Use the "Load Sample Data" button to see how the tool works
2. **Upload CSV**: Upload your own net worth data in CSV format
3. **Manual Entry**: Add entries one by one using the form in the sidebar

**Expected CSV format:**
```csv
Date,UBS Account (CHF),IBKR Account (CHF),Kutxabank Account (EUR)
2025-01-31,10000.00,15000.00,1500.00
2025-02-28,10500.00,15500.00,1550.00
```

### Transaction Analysis

1. **PDF Upload**: Upload bank statement PDFs (works well with UBS and similar formats)
2. **CSV Upload**: Upload transaction data in CSV format

**Expected CSV format:**
```csv
date,description,amount,category
2025-12-01,Grocery Shopping,-50.00,Food & Dining
2025-12-01,Salary,3000.00,Income
2025-12-02,Coffee Shop,-4.50,Food & Dining
```

## 💡 Key Features in Detail

### Multi-Currency Support
- Automatic conversion to CHF for unified tracking
- Real-time forex rates via yfinance
- Support for EUR, USD, CHF, and more

### Intelligent Transaction Categorization
The tool automatically categorizes transactions into:
- Income
- Food & Dining
- Transportation
- Shopping
- Healthcare & Insurance
- Housing & Utilities
- Entertainment & Lifestyle
- Travel
- Financial Services
- Personal Transfers
- Telecommunications
- Government & Official
- Cash & Other

### Interactive Visualizations
- Net worth trend charts
- Account distribution pie charts
- Monthly income vs expenses
- Spending by category
- Cash flow analysis
- Financial health indicators

### Data Export & Import
- Export data as CSV or Excel
- Download filtered transaction data
- Save to local data folder
- ZIP export of all data
- Persistent data storage

## 🔧 Technical Details

### Built With
- **Streamlit** - Web interface framework
- **Plotly** - Interactive charts and visualizations
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **yfinance** - Real-time forex rates
- **pdfplumber** - PDF text extraction
- **Matplotlib/Seaborn** - Additional plotting capabilities

### Project Structure
```
comprehensive-finance-tool/
├── app.py                 # Main Streamlit application
├── setup.py              # Installation and setup script
├── requirements.txt      # Python dependencies
├── modules/
│   └── transaction_analyzer.py  # PDF processing and categorization
├── data/                 # Local data storage
│   └── net_worth_data.csv
├── exports/              # Export files
└── uploads/              # Temporary file uploads
```

## 📈 Usage Examples

### Tracking Net Worth Growth
1. Upload your account balances monthly
2. View automatic currency conversion to CHF
3. Monitor growth trends and account performance
4. Export data for external analysis

### Analyzing Spending Patterns
1. Upload bank statement PDFs or CSV files
2. Review automatic transaction categorization
3. Filter and analyze spending by category/time
4. Identify areas for potential savings

### Financial Health Assessment
1. Load both net worth and transaction data
2. View the Overview dashboard
3. Check financial health indicators
4. Monitor savings rate and growth trends

## 🛠️ Customization

### Adding New Transaction Categories
Edit the `categorize_transactions` method in `modules/transaction_analyzer.py` to add custom category rules.

### Supporting New Bank Formats
Modify the PDF parsing patterns in `parse_transactions` method to support additional bank statement formats.

### Custom Metrics
Add new financial health indicators in the `overview_dashboard` function.

## 🐛 Troubleshooting

### PDF Processing Issues
- Ensure PDF is text-based (not scanned images)
- Check if transaction format matches expected patterns
- Use CSV upload as alternative

### Currency Conversion Issues
- Check internet connection for live forex rates
- Fallback rates are used when live rates unavailable

### Data Not Saving
- Ensure write permissions in project directory
- Check data folder exists and is accessible

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section
2. Review the code comments for technical details
3. Create an issue with detailed description and error messages

## 🔮 Future Enhancements

- Investment portfolio tracking integration
- Budget planning and forecasting
- Multi-user support with authentication
- Mobile app companion
- Bank API integrations
- Advanced financial analytics and reporting
- Goal tracking and progress monitoring

---

**Happy financial tracking! 💰📊**
