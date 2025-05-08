# Financial Planning Tool

## Overview

This financial planning tool is a comprehensive web application designed to help users navigate their investment journey through personalized risk assessment and portfolio construction. Built with Streamlit, the tool combines machine learning with traditional financial planning approaches to deliver tailored investment recommendations.

## Key Features

- **Risk Assessment Quiz**: Interactive questionnaire that evaluates user risk tolerance through various factors including age, income, investment experience, and market volatility preferences
- **Stock Screening**: Analyze and filter stocks based on technical indicators, fundamental metrics, and sector analysis
- **Portfolio Construction**: Build personalized investment portfolios based on user risk profiles with proper diversification across risk categories
- **Machine Learning Integration**: Hybrid approach using both rule-based and ML models for risk assessment, with continuous improvement as more data is collected
- **Interactive Visualizations**: Intuitive charts and graphs to help users understand investment concepts and portfolio composition

## Project Structure

```
├── main.py                      # Main application entry point
├── pages/                       # Streamlit pages
│   ├── risk_assessment_app.py   # Risk assessment questionnaire
│   ├── stock_screener.py        # Stock filtering and analysis
│   └── portfolio_summary.py     # Portfolio builder and visualization
├── core/                        # Core functionality modules
│   ├── scoring_mechanism.py     # Risk tolerance calculation
│   ├── ml.py                    # Machine learning components
│   ├── technical_analysis.py    # Stock technical indicators
│   ├── sector_analysis.py       # Sector-based stock analysis
│   ├── portfolio_function.py    # Portfolio construction algorithms
│   └── utils.py                 # Helper functions
├── classified_stocks.csv        # Stock database with classifications
├── user_data.csv                # Anonymized user data for ML training
├── risk_assessment_model.joblib # Saved ML model for risk assessment
└── requirements.txt             # Project dependencies
```

## Technologies Used

- **Streamlit**: Interactive web application framework
- **Pandas & NumPy**: Data manipulation and numerical calculations
- **Scikit-learn**: Machine learning model for risk assessment
- **SQLite**: Database for storing user profiles
- **Matplotlib, Seaborn & Altair**: Data visualization libraries

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/financial-planning-tool.git
cd financial-planning-tool
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run main.py
```

The app will be available at `http://localhost:8501` by default.

## Application Flow

1. **Risk Assessment**: Users complete a questionnaire about their financial situation and investment preferences
2. **Risk Profile Generation**: The system calculates a risk tolerance score and classification
3. **Portfolio Construction**: Based on the risk profile, the system builds a diversified portfolio of stocks
4. **Portfolio Analysis**: Users can view their portfolio composition, expected performance, and risk metrics

## Machine Learning Model

The risk assessment uses a Random Forest Regressor model that:
- Initially falls back to rule-based scoring when data is limited
- Transitions to ML-based scoring as more user data is collected
- Continuously improves with new user inputs

## Security and Privacy

- User data is anonymized via hashing before storage
- No sensitive financial information is stored in plain text
- Database includes only information necessary for model improvement

## Future Enhancements

- Real-time market data integration
- Portfolio backtesting capabilities
- Expanded stock database
- Advanced portfolio optimization using modern portfolio theory
- Mobile-responsive design

## Disclaimer

This tool is for educational and informational purposes only. It does not constitute financial advice. Users should consult with qualified financial advisors before making investment decisions.


## Acknowledgments

Supervisor : Dr Tan Chye Cheah