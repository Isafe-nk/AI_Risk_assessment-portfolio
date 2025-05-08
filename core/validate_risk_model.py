import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import joblib
import os

from core.ml import calculate_risk_score_ml, prepare_data_for_ml, train_ml_model
from core.scoring_mechanism import calculate_risk_score_rule_based, get_risk_tolerance

def validate_risk_model():
    """
    Validate the risk assessment model by comparing rule-based vs ML predictions
    and analyzing accuracy metrics.
    """
    print("Starting risk assessment model validation...")
    
    # Check if we have enough data for validation
    if not os.path.exists('user_data.csv'):
        print("No user data found. Please collect data first.")
        return
    
    # Load the data
    data = pd.read_csv('user_data.csv')
    print(f"Loaded {len(data)} user profiles for validation.")
    
    # Convert risk_score to numeric, coercing errors to NaN
    data['risk_score'] = pd.to_numeric(data['risk_score'], errors='coerce')
    
    # Drop any rows with NaN risk_score
    before_count = len(data)
    data = data.dropna(subset=['risk_score'])
    after_count = len(data)
    if before_count > after_count:
        print(f"Removed {before_count - after_count} rows with invalid risk scores.")
    
    if len(data) < 10:
        print("Warning: Sample size is very small. Results may not be reliable.")
    
    # Analyze risk score distribution
    print("\n=== Risk Score Distribution ===")
    score_stats = data['risk_score'].describe()
    print(score_stats)
    
    # Check for outliers
    q1 = data['risk_score'].quantile(0.25)
    q3 = data['risk_score'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data['risk_score'] < lower_bound) | (data['risk_score'] > upper_bound)]
    
    if not outliers.empty:
        print(f"\n=== Found {len(outliers)} outliers ===")
        print("Outlier risk scores:")
        print(outliers['risk_score'].values)
    
    # Map scores to risk tolerance categories
    data['risk_tolerance'] = data['risk_score'].apply(get_risk_tolerance)
    category_counts = data['risk_tolerance'].value_counts()
    print("\n=== Risk Category Distribution ===")
    print(category_counts)
    
    # Visualize risk score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['risk_score'], bins=20, kde=True)
    plt.title('Risk Score Distribution')
    plt.xlabel('Risk Score')
    plt.ylabel('Count')
    plt.savefig('risk_score_distribution.png')
    
    # Visualize risk category distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(y='risk_tolerance', data=data, order=data['risk_tolerance'].value_counts().index)
    plt.title('Risk Category Distribution')
    plt.tight_layout()
    plt.savefig('risk_category_distribution.png')
    
    # ML model validation (if we have enough data)
    if len(data) >= 30:
        print("\n=== Machine Learning Model Validation ===")
        
        try:
            # Prepare the data - explicitly convert to numpy arrays to avoid pandas issues
            X = data.drop(['risk_score', 'risk_tolerance'], axis=1, errors='ignore').select_dtypes(include=[np.number])
            
            # Handle potential duplicate columns - common issue in ML error
            X = X.loc[:, ~X.columns.duplicated()]
            
            # Convert to numpy array to avoid pandas DataFrame issues with some sklearn operations
            X_np = X.to_numpy()
            y_np = data['risk_score'].to_numpy()
            
            print(f"Feature matrix shape: {X_np.shape}, Target vector shape: {y_np.shape}")
            print(f"Features used: {X.columns.tolist()}")
            
            # Create model or load existing
            try:
                model = joblib.load('risk_assessment_model.joblib')
                print("Loaded existing ML model.")
            except:
                print("Training new ML model.")
                # Train on original data format since that's what the function expects
                model = train_ml_model()
            
            # Simple train/test evaluation instead of cross-validation to avoid pandas issues
            print("\n=== Simple Train/Test Evaluation ===")
            # Use 80% for training, 20% for testing
            train_size = int(0.8 * len(X_np))
            indices = np.random.permutation(len(X_np))
            train_idx, test_idx = indices[:train_size], indices[train_size:]
            
            X_train, y_train = X_np[train_idx], y_np[train_idx]
            X_test, y_test = X_np[test_idx], y_np[test_idx]
            
            # Train a new model on the train set
            from sklearn.ensemble import RandomForestRegressor
            model_eval = RandomForestRegressor(n_estimators=100, random_state=42)
            model_eval.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = model_eval.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Mean Absolute Error: {mae:.2f}")
            print(f"R-squared: {r2:.2f}")
            print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model_eval.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\n=== Top 10 Important Features ===")
            print(feature_importance.head(10))
            
            # Visualize feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title('Top 15 Feature Importance for Risk Score Prediction')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            
            # Compare rule-based and ML predictions
            print("\n=== Rule-based vs ML Model Comparison ===")
            # Get a sample of the data for rule-based evaluation
            sample_size = min(len(data), 10)
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            sample = data.iloc[sample_indices]
            
            results = []
            for _, row in sample.iterrows():
                # Convert row to dict format needed for rule-based calculation
                answers = {}
                for col in row.index:
                    if col != 'risk_score' and col != 'risk_tolerance':
                        answers[col] = row[col]
                
                # Get rule-based prediction
                try:
                    rule_score = calculate_risk_score_rule_based(answers)
                except Exception as e:
                    print(f"Error in rule-based calculation: {e}")
                    rule_score = None
                
                # Get ML prediction
                try:
                    # Extract features similar to what we used for training
                    features = {col: row[col] for col in X.columns if col in row.index}
                    features_array = np.array([list(features.values())])
                    ml_score = model_eval.predict(features_array)[0]
                except Exception as e:
                    print(f"Error in ML prediction: {e}")
                    ml_score = None
                
                results.append({
                    'Actual': row['risk_score'],
                    'Rule-based': rule_score,
                    'ML': ml_score,
                    'Rule_diff': abs(rule_score - row['risk_score']) if rule_score is not None else None,
                    'ML_diff': abs(ml_score - row['risk_score']) if ml_score is not None else None
                })
            
            comparison_df = pd.DataFrame(results)
            print(comparison_df)
            
            # Calculate average errors (ignoring None values)
            rule_diffs = [diff for diff in comparison_df['Rule_diff'] if diff is not None]
            ml_diffs = [diff for diff in comparison_df['ML_diff'] if diff is not None]
            
            rule_mean = np.mean(rule_diffs) if rule_diffs else float('nan')
            ml_mean = np.mean(ml_diffs) if ml_diffs else float('nan')
            
            print(f"\nAverage rule-based error: {rule_mean:.2f}")
            print(f"Average ML error: {ml_mean:.2f}")
            
            # Correlation analysis between important variables and risk score
            print("\n=== Correlation with Risk Score ===")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            correlations = data[numeric_cols].corr()['risk_score'].sort_values(ascending=False)
            print(correlations.head(10))
            
        except Exception as e:
            print(f"Error during ML validation: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nSkipping ML validation due to errors.")
    else:
        print("\nNot enough data for ML validation. Need at least 30 samples.")
    
    print("\nRisk model validation complete.")

if __name__ == "__main__":
    validate_risk_model()