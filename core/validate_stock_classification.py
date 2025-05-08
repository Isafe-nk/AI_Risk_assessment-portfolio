import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from datetime import datetime, timedelta

from core.portfolio_function import calculate_risk_score_alpha, classify_stocks_alpha, build_portfolio
from core.sector_analysis import SectorType

def validate_stock_classification():
    """
    Validate the stock classification model by analyzing classification accuracy and portfolio performance.
    """
    print("Starting stock classification model validation...")
    
    # Check if we have stock data for validation
    if not os.path.exists('classified_stocks.csv'):
        print("No classified stocks data found. Please run stock screener first.")
        return
    
    # Load the data
    df = pd.read_csv('classified_stocks.csv')
    print(f"Loaded {len(df)} stocks for validation.")
    
    # 1. Analyze risk category distribution
    print("\n=== Risk Category Distribution ===")
    category_counts = df['Risk Category'].value_counts()
    print(category_counts)
    
    # Visualize risk category distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(y='Risk Category', data=df, order=['Low', 'Medium', 'High'])
    for i, count in enumerate(category_counts):
        ax.text(count + 0.1, i, str(count), va='center')
    plt.title('Stock Risk Category Distribution')
    plt.tight_layout()
    plt.savefig('stock_category_distribution.png')
    
    # 2. Analyze risk category by sector
    print("\n=== Risk Category by Sector ===")
    sector_risk = pd.crosstab(df['Sector'], df['Risk Category'])
    print(sector_risk)
    
    # Visualize sector risk distribution
    plt.figure(figsize=(12, 8))
    sector_risk_pct = sector_risk.div(sector_risk.sum(axis=1), axis=0)
    sector_risk_pct.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Risk Category Distribution by Sector')
    plt.ylabel('Percentage')
    plt.legend(title='Risk Category')
    plt.tight_layout()
    plt.savefig('sector_risk_distribution.png')
    
    # 3. Analyze technical indicators by risk category
    print("\n=== Technical Indicators by Risk Category ===")
    indicators = ['30D Volatility', 'RSI', 'P/E Ratio', 'P/B Ratio']
    available_indicators = [ind for ind in indicators if ind in df.columns]
    
    for indicator in available_indicators:
        print(f"\n{indicator} by Risk Category:")
        print(df.groupby('Risk Category')[indicator].describe())
    
    # Visualize technical indicators by risk category
    for indicator in available_indicators:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Risk Category', y=indicator, data=df, order=['Low', 'Medium', 'High'])
        plt.title(f'{indicator} Distribution by Risk Category')
        plt.tight_layout()
        plt.savefig(f'{indicator.replace("/", "_")}_by_category.png')
    
    # 4. Validate alpha score calculation
    print("\n=== Alpha Score Validation ===")
    
    # Calculate alpha scores using the model
    df['Alpha Score'] = df.apply(calculate_risk_score_alpha, axis=1)
    
    # Analyze alpha scores
    print("\nAlpha Score by Risk Category:")
    alpha_by_category = df.groupby('Risk Category')['Alpha Score'].describe()
    print(alpha_by_category)
    
    # Visualize alpha scores
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Risk Category', y='Alpha Score', data=df, order=['Low', 'Medium', 'High'])
    plt.title('Alpha Score Distribution by Risk Category')
    plt.tight_layout()
    plt.savefig('alpha_score_distribution.png')
    
    # 5. Reclassification validation
    print("\n=== Classification Validation ===")
    
    # Reclassify using the model
    df['Reclassified'] = df.apply(classify_stocks_alpha, axis=1)
    
    # Create confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(df['Risk Category'], df['Reclassified'], 
                                  labels=['Low', 'Medium', 'High'])
    print(pd.DataFrame(conf_matrix, 
                      index=['True Low', 'True Medium', 'True High'], 
                      columns=['Pred Low', 'Pred Medium', 'Pred High']))
    
    # Classification report
    print("\nClassification Report:")
    class_report = classification_report(df['Risk Category'], df['Reclassified'])
    print(class_report)
    
    # Calculate accuracy
    accuracy = (df['Risk Category'] == df['Reclassified']).mean()
    print(f"Classification Accuracy: {accuracy:.2%}")
    
    # 6. Test portfolio construction
    print("\n=== Portfolio Construction Validation ===")
    
    # Test with different risk allocations
    risk_profiles = {
        'Conservative': (10, 20, 70),
        'Moderately Conservative': (20, 30, 50),
        'Moderate': (40, 40, 20),
        'Moderately Aggressive': (60, 30, 10),
        'Aggressive': (90, 10, 0)
    }
    
    for profile, allocation in risk_profiles.items():
        try:
            portfolio = build_portfolio(allocation, df)
            
            if not portfolio.empty:
                risk_dist = portfolio['Risk Level'].value_counts(normalize=True) * 100
                print(f"\n{profile} Portfolio Risk Distribution:")
                print(risk_dist)
                
                # Check if risk distribution matches target allocation
                high_pct, med_pct, low_pct = allocation
                actual_high = risk_dist.get('High', 0)
                actual_med = risk_dist.get('Medium', 0)
                actual_low = risk_dist.get('Low', 0)
                
                print(f"Target: High={high_pct}%, Medium={med_pct}%, Low={low_pct}%")
                print(f"Actual: High={actual_high:.1f}%, Medium={actual_med:.1f}%, Low={actual_low:.1f}%")
                
                # Calculate deviation from target allocation
                deviation = abs(high_pct - actual_high) + abs(med_pct - actual_med) + abs(low_pct - actual_low)
                print(f"Total deviation from target: {deviation:.1f}%")
            else:
                print(f"\nCould not build {profile} portfolio - insufficient stock data.")
        except Exception as e:
            print(f"Error building {profile} portfolio: {str(e)}")
    
    # 7. Calculate correlation between key risk indicators
    print("\n=== Risk Indicator Correlations ===")
    
    corr_indicators = ['Alpha Score', '30D Volatility', 'RSI']
    corr_indicators = [ind for ind in corr_indicators if ind in df.columns]
    
    if corr_indicators:
        correlation_matrix = df[corr_indicators].corr()
        print(correlation_matrix)
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Between Risk Indicators')
        plt.tight_layout()
        plt.savefig('risk_indicator_correlation.png')
    
    print("\nStock classification validation complete.")

if __name__ == "__main__":
    validate_stock_classification()