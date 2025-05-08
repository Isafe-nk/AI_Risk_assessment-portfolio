import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

from core.scoring_mechanism import questions, calculate_risk_score_rule_based

def prepare_data_for_ml(answers):
    data = {}
    for key, value in answers.items():
        question = next((q for q in questions if q.get('key') == key), None)
        if question:
            if key == 'name':
                # Skip the name to maintain anonymity
                continue
            elif question['type'] == 'image_buttons' and key == 'investment_experience':
                # Handle investment experience separately
                for option in question['options']:
                    data[f"{key}_{option['key']}"] = 1 if value == option['text'] else 0
            elif 'options' in question:
                if isinstance(value, list):  # For multiselect questions
                    for option in question['options']:
                        data[f"{key}_{option}"] = 1 if option in value else 0
                elif isinstance(value, str):  # For categorical questions
                    for option in question['options']:
                        data[f"{key}_{option}"] = 1 if option == value else 0
            else:  # For numerical or text questions without options
                data[key] = value
        else:
            # Handle nested questions
            for q in questions:
                if q['type'] in ['initial', 'employment_income', 'assets_liabilities']:
                    sub_question = next((sq for sq in q['questions'] if sq.get('key') == key), None)
                    if sub_question:
                        if 'options' in sub_question and isinstance(value, str):
                            for option in sub_question['options']:
                                data[f"{key}_{option}"] = 1 if option == value else 0
                        else:
                            data[key] = value
                        break
    return data


def train_ml_model():
    data = pd.read_csv('user_data.csv')
    X = data.drop('risk_score', axis=1)
    y = data['risk_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'risk_assessment_model.joblib')
    return model

def calculate_risk_score_ml(answers):
    MIN_SAMPLES = 30  # Minimum number of samples required for ML

    if os.path.exists('user_data.csv'):
        data = pd.read_csv('user_data.csv')
        if len(data) >= MIN_SAMPLES:
            try:
                model = joblib.load('risk_assessment_model.joblib')
            except:
                model = train_ml_model()
            
            data = prepare_data_for_ml(answers)
            features = pd.DataFrame([data])
            return int(model.predict(features)[0])
        else:
            print(f"Using rule-based model. Need {MIN_SAMPLES - len(data)} more samples for ML.")
    else:
        print("No data file found. Using rule-based model.")

    return calculate_risk_score_rule_based(answers)