import math

# Risk assessment questions
questions = [
    {"type": "text", "question": "What is your name?", "key": "name"},
    {"type": "initial", "key": "initial", "questions": [
        {"type": "number", "question": "Age", "key": "age"},
        {"type": "select", "question": "Marital status", "key": "marital_status",
         "options": ["Single", "Married", "Separated", "Divorced", "Widowed"]},
        {"type": "number", "question": "How many dependents do you have?", "min_value": 0, "max_value": 10, "key": "dependents"}
    ]},
    {"type": "employment_income", "key": "employment_income", "questions": [
        {"type": "radio", "question": "Are you currently employed?", "key": "employed",
         "options": ["Yes", "No"]},
        {"type": "number", "question": "What is your annual household income?", "key": "income"},
        {"type": "buttons", "question": "Which statement best describes your home ownership status?", "key": "home_ownership",
         "options": ["I don't own a home", "I'm paying a mortgage", "My mortgage is paid off"]}
    ]},
    {"type": "assets_liabilities", "key": "assets_liabilities", "questions": [
        {"type": "number", "question": "What is the total value of all your assets?", "key": "total_assets"},
        {"type": "number", "question": "What is the value of your fixed assets (e.g., property, vehicles)?", "key": "fixed_assets"},
        {"type": "number", "question": "What is the total value of your liabilities?", "key": "liabilities"}
    ]},
    {"type": "multiselect", "question": "What are your primary financial goals?", 
     "options": ["Retirement", "Home purchase", "Education", "Emergency fund", "Wealth accumulation"], 
     "key": "financial_goals"},
    {"type": "select", "question": "Which life stage best describes you?", 
     "options": ["Starting out", "Career building", "Peak earning years", "Pre-retirement", "Retirement"], 
     "key": "life_stage"},
    {"type": "image_buttons", "question": "How would you describe your investment experience?", 
     "options": [
         {"text": "Mostly Cash Savings", "image": "💰", "key": "cash_savings"},
         {"text": "Bonds, Income funds, GICs", "image": "📊", "key": "bonds_income"},
         {"text": "Mutual Funds and Exchange Traded Funds (ETFs)", "image": "📈", "key": "mutual_etfs"},
         {"text": "Self-Directed Investor: Stocks, Equities, Cryptocurrencies", "image": "🚀", "key": "self_directed"}
     ], 
     "key": "investment_experience"},
    {"type": "radio", "question": "How would you react if your investment lost 20% in a year?", 
     "options": ["Sell all investments", "Sell some", "Hold steady", "Buy more", "Buy a lot more"], "key": "market_reaction"},
    {"type": "chart", "question": "What level of volatility would you be the most comfortable with?", 
     "options": ["Low Volatility", "Medium", "High Volatility"], 
     "key": "volatility_preference"},
    {"type": "radio", "question": "How long do you plan to hold your investments?", 
     "options": ["0-3 years", "3-8 years", "8+ years"], "key": "investment_horizon"},
    {"type": "radio", "question": "What's your risk capacity (ability to take risks)?", 
     "options": ["Very low", "Low", "Medium", "High", "Very high"], "key": "risk_capacity"},
    {"type": "slider", "question": "How confident are you in your investment knowledge?", 
     "min_value": 0, "max_value": 10, "step": 1, "key": "investment_confidence"}
]

def calculate_risk_score_rule_based(answers):
    score = 0
    income = answers.get('income', 0)
    weights = {
        "age": lambda x: max(0, min(10, (60 - x) / 4)),  # Increased impact, 0-10 range
        "marital_status": {"Single": 6, "Married": 4, "Separated": 3, "Divorced": 2, "Widowed": 1},
        "dependents": lambda x: max(0, 8 - x * 2 + min(income/25000,2)),  # 0 dependents = 8, 1 = 6, 2 = 4, 3 = 2, 4+ = 0
        "employed": {"Yes": 5, "No": 0},
        "income": lambda x: min(8, x / 25000),  # 1 point per $25k, max 8 points
        "home_ownership": {"I don't own a home": 0, "I'm paying a mortgage": 4, "My mortgage is paid off": 8},
        "investment_experience": {
            "Mostly Cash Savings and GICs": 0,
            "Bonds, Income funds, GICs": 3,
            "Mutual Funds and Exchange Traded Funds (ETFs)": 6,
            "Self-Directed Investor: Stocks, Equities, Cryptocurrencies": 10
        },
        "market_reaction": {"Sell all investments": 0, "Sell some": 3, "Hold steady": 6, "Buy more": 8, "Buy a lot more": 10},
        "volatility_preference": {"Low Volatility": 0, "Medium": 5, "High Volatility": 10},
        "investment_horizon": {"0-3 years": 0, "3-8 years": 5, "8+ years": 10},
        "risk_capacity": {"Very low": 0, "Low": 3, "Medium": 6, "High": 8, "Very high": 10}
    }
    
    for key, value in answers.items():
        if key in weights:
            if callable(weights[key]):
                score += weights[key](value)
            elif isinstance(weights[key], dict):
                score += weights[key].get(value, 0)
            elif isinstance(value, (int, float)):
                score += value * weights[key]
    
    # Calculate net worth and add to score
    total_assets = answers.get('total_assets', 0)
    liabilities = answers.get('liabilities', 0)
    net_worth = total_assets - liabilities
    
    # Add net worth factor to score (0-10 points)
    net_worth_score = min(10, max(0, 10 * math.sqrt(net_worth / 1_000_000))) if net_worth > 0 else 0
    score += net_worth_score
    
    # Add liquidity factor to score (0-5 points)
    liquid_assets = total_assets - answers.get('fixed_assets', 0)
    liquidity_ratio = liquid_assets / total_assets if total_assets > 0 else 0
    liquidity_score = liquidity_ratio * 5  # 0-5 points based on liquidity ratio
    score += liquidity_score
    # print score in terminal
    print(f"Risk Score: {score}")
    return score  # Note: We're not converting to int here to allow for more granularity


def get_risk_tolerance(score):
    if score < 30:
        return "Conservative"
    elif score < 50:
        return "Moderately Conservative"
    elif score < 70:
        return "Moderate"
    elif score < 90:
        return "Moderately Aggressive"
    else:
        return "Aggressive"
    
def get_allocation(risk_tolerance):
    allocations = {
        "Conservative": (10, 20, 70),
        "Moderately Conservative": (20, 30, 50),
        "Moderate": (40, 40, 20),
        "Moderately Aggressive": (60, 30, 10),
        "Aggressive": (90, 10, 0)
    }
    return allocations[risk_tolerance]