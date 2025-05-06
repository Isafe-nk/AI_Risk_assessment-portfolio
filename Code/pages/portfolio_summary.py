import streamlit as st
import pandas as pd
from core.scoring_mechanism import get_allocation
from core.portfolio_function import build_portfolio
import os

st.set_page_config(page_title="Portfolio Summary", page_icon="📊", layout="centered")
st.title("Recommendation stocks in your Portfolio")

# Step 1: Get risk tolerance from session
if "risk_tolerance" not in st.session_state:
    st.error("Please complete the risk assessment first.")
    st.stop()

risk_profile = st.session_state.risk_tolerance
allocation = get_allocation(risk_profile)

st.info(f"Risk Profile: **{risk_profile}** — Allocation (High/Med/Low): {allocation}")

# Step 2: Load classified stocks
try:
    if not os.path.exists("classified_stocks.csv"):
        st.warning("No stock classifications found. Please run the stock screener first.")
        if st.button("Go to Stock Screener"):
            st.switch_page("pages/stock_screener.py")
        st.stop()
    
    df_stocks = pd.read_csv("classified_stocks.csv")
    if df_stocks.empty:
        st.warning("No stocks have been classified yet. Please run the stock screener first.")
        if st.button("Go to Stock Screener"):
            st.switch_page("pages/stock_screener.py")
        st.stop()

except Exception as e:
    st.error(f"Error loading stock data: {str(e)}")
    st.stop()

# Step 3: Build portfolio
try:
    # num_stocks can be adjusted or made configurable
    num_stocks = st.slider("Number of stocks in portfolio", min_value=5, max_value=20, value=12)
    portfolio_df = build_portfolio(allocation, df_stocks, num_stocks=num_stocks)
    
    if portfolio_df.empty:
        st.warning("No suitable portfolio could be built with the current stock classifications.")
        st.stop()

    st.subheader("Your Portfolio")
    
    # Determine what columns to display in a clean format
    display_columns = ['Ticker', 'Company Name', 'Sector', 'Current Price', 'Recommendation', 'Valuation Score', 'Risk Level', 'Weight %']
    display_columns = [col for col in display_columns if col in portfolio_df.columns]
    
    st.dataframe(portfolio_df[display_columns])

    # Show portfolio metrics
    st.subheader("Portfolio Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk = len(portfolio_df[portfolio_df['Risk Level'] == 'High'])
        med_risk = len(portfolio_df[portfolio_df['Risk Level'] == 'Medium'])
        low_risk = len(portfolio_df[portfolio_df['Risk Level'] == 'Low'])
        st.metric("Risk Distribution", f"H:{high_risk} M:{med_risk} L:{low_risk}")
        
    with col2:
        if 'Valuation Score' in portfolio_df.columns:
            avg_val_score = portfolio_df['Valuation Score'].mean()
            st.metric("Avg. Valuation Score", f"{avg_val_score:.1f}")
    
    with col3:
        if 'Sector' in portfolio_df.columns:
            sectors = portfolio_df['Sector'].value_counts()
            top_sector = sectors.index[0] if not sectors.empty else "N/A"
            st.metric("Top Sector", top_sector, f"{sectors.iloc[0]} stocks")
    
    # Create downloadable portfolio file
    st.download_button(
        "Download Portfolio",
        portfolio_df.to_csv(index=False),
        file_name="portfolio.csv"
    )

except Exception as e:
    st.error(f"Error building portfolio: {str(e)}")
    st.stop()

