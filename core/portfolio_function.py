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
    Fills in with stocks from other categories if needed to reach num_stocks.
    """
    high_pct, med_pct, low_pct = risk_allocation
    num_high = int(round(num_stocks * high_pct / 100))
    num_med = int(round(num_stocks * med_pct / 100))
    num_low = num_stocks - num_high - num_med

    # Initial weight calculation (will be adjusted later if needed)
    buckets = {
        'High': stock_df[stock_df['Risk Category'] == 'High'],
        'Medium': stock_df[stock_df['Risk Category'] == 'Medium'],
        'Low': stock_df[stock_df['Risk Category'] == 'Low']
    }

    # First pass: try to get the requested number from each category
    picks = []
    remaining_counts = {}
    for level, count in zip(['High', 'Medium', 'Low'], [num_high, num_med, num_low]):
        pool = buckets[level]
        n = min(count, len(pool))
        remaining_counts[level] = count - n
        
        if n > 0:
            if 'Valuation Score' in pool.columns:
                normalized_scores = pool['Valuation Score'] + 1
                sampled = pool.sample(n=n, weights=normalized_scores, random_state=42)
            else:
                sampled = pool.sample(n=n, random_state=42)
            
            for _, row in sampled.iterrows():
                stock_info = row.to_dict()
                stock_info['Risk Level'] = level
                picks.append(stock_info)
    
    # Second pass: fill in with remaining stocks if we don't have enough
    remaining_slots = num_stocks - len(picks)
    if remaining_slots > 0:
        # Combine all remaining stocks
        used_indices = set()
        for p in picks:
            if 'index' in p:  # Keeping track of indices we've already used
                used_indices.add(p['index'])
        
        # Get remaining stocks not already picked
        remaining_stocks = stock_df[~stock_df.index.isin(used_indices)]
        
        # Sample from remaining stocks if there are any
        if not remaining_stocks.empty:
            n_to_sample = min(remaining_slots, len(remaining_stocks))
            if n_to_sample > 0:
                if 'Valuation Score' in remaining_stocks.columns:
                    normalized_scores = remaining_stocks['Valuation Score'] + 1
                    extra_stocks = remaining_stocks.sample(n=n_to_sample, weights=normalized_scores, random_state=43)
                else:
                    extra_stocks = remaining_stocks.sample(n=n_to_sample, random_state=43)
                
                # Add these to our picks
                for _, row in extra_stocks.iterrows():
                    stock_info = row.to_dict()
                    stock_info['Risk Level'] = row.get('Risk Category', 'Medium')  # Default to Medium if missing
                    picks.append(stock_info)
    
    # Recalculate weight based on actual number of stocks selected
    actual_num_stocks = len(picks)
    actual_weight = round(100.0 / actual_num_stocks, 2) if actual_num_stocks > 0 else 0
    
    # Assign the recalculated weight to each stock
    for stock_info in picks:
        stock_info['Weight %'] = actual_weight

    return pd.DataFrame(picks)
