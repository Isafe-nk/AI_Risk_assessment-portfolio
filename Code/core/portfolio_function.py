import numpy as np
import pandas as pd

def calculate_risk_score_alpha(row, w1=0.3, w2=0.2, w3=0.2, w4=0.2, w5=0.1):
    """
    Calculate the risk score α for a stock using specified weights.
    Returns a float score.
    Handles missing Rt (Daily_Return) and Pt (Close) gracefully.
    """
    sigma = row.get('Volatility', 0)
    Rt = row.get('Daily_Return', None)
    Pt = row.get('Close', None)
    MA50 = row.get('MA50', None)
    MA200 = row.get('MA200', None)
    RSI = row.get('RSI', 50)

    # Only compute deviations if price and MA are available and nonzero
    ma50_dev = abs(Pt - MA50) / MA50 if Pt not in (None, 0) and MA50 not in (None, 0) else 0
    ma200_dev = abs(Pt - MA200) / MA200 if Pt not in (None, 0) and MA200 not in (None, 0) else 0
    rsi_dev = abs(RSI - 50) / 50

    # Only use return if available
    abs_return = abs(Rt) if Rt is not None else 0

    alpha = (
        w1 * sigma +
        w2 * abs_return +
        w3 * ma50_dev +
        w4 * ma200_dev +
        w5 * rsi_dev
    )
    return alpha

def classify_stocks_alpha(row, w1=0.3, w2=0.2, w3=0.2, w4=0.2, w5=0.1, theta1=0.33, theta2=0.66):
    """
    Classify a single stock's risk category using alpha and static thresholds.
    """
    alpha = calculate_risk_score_alpha(row, w1, w2, w3, w4, w5)
    if alpha <= theta1:
        return 'Low'
    elif alpha <= theta2:
        return 'Medium'
    else:
        return 'High'

def build_portfolio(risk_allocation, stock_df, num_stocks=12):
    """
    Build a portfolio based on (High%, Medium%, Low%) allocation tuple.
    Returns a DataFrame with stock details and portfolio weights.
    """
    high_pct, med_pct, low_pct = risk_allocation
    num_high = int(round(num_stocks * high_pct / 100))
    num_med = int(round(num_stocks * med_pct / 100))
    num_low = num_stocks - num_high - num_med

    weight = round(100.0 / num_stocks, 2)
    buckets = {
        'High': stock_df[stock_df['Risk Category'] == 'High'],
        'Medium': stock_df[stock_df['Risk Category'] == 'Medium'],
        'Low': stock_df[stock_df['Risk Category'] == 'Low']
    }

    picks = []
    for level, count in zip(['High', 'Medium', 'Low'], [num_high, num_med, num_low]):
        pool = buckets[level]
        n = min(count, len(pool))
        if n > 0:
            # Sample according to valuation score (higher is better)
            if 'Valuation Score' in pool.columns:
                # Normalize scores to ensure all are positive
                normalized_scores = pool['Valuation Score'] + 1
                sampled = pool.sample(n=n, weights=normalized_scores, random_state=42)
            else:
                sampled = pool.sample(n=n, random_state=42)
                
            for _, row in sampled.iterrows():
                # Create a dictionary with all information from the row
                stock_info = row.to_dict()
                # Add portfolio specific information
                stock_info['Weight %'] = weight
                stock_info['Risk Level'] = level
                picks.append(stock_info)

    return pd.DataFrame(picks)
