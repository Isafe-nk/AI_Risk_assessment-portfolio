import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import altair as alt
import os
import hashlib


from core.scoring_mechanism import questions, get_risk_tolerance, get_allocation
from core.ml import prepare_data_for_ml, calculate_risk_score_ml


# Function to get a new database connection
def get_db_connection():
    return sqlite3.connect('user_profiles.db')

# Database setup
def setup_database():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_profiles
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name_hash TEXT,
                  age INTEGER,
                  income REAL,
                  dependents INTEGER,
                  investment_experience TEXT,
                  risk_score INTEGER,
                  risk_tolerance TEXT,
                  equity_allocation INTEGER,
                  income_allocation INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  feedback TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Call setup_database at the start of your script
setup_database()

def create_investment_chart(volatility_level):
    np.random.seed(42)  # For reproducibility
    x = np.arange(100)
    
    if volatility_level == "Low Volatility":
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 1, 100)
        y = 100 + trend + np.cumsum(noise) * 0.3
    elif volatility_level == "Balanced":
        trend = np.linspace(0, 20, 100)
        noise = np.random.normal(0, 1, 100)
        y = 100 + trend + np.cumsum(noise)
    else:  # High Volatility
        trend = np.linspace(0, 40, 100)  # Steeper overall trend
        volatility = np.random.normal(0, 1, 100) * 3  # Increased volatility
        momentum = np.cumsum(np.random.normal(0, 0.1, 100))  # Add momentum
        y = 100 + trend + np.cumsum(volatility) + momentum * 10
    
    df = pd.DataFrame({'x': x, 'y': y})
    
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('x', axis=alt.Axis(title='Time')),
        y=alt.Y('y', axis=alt.Axis(title='Value'), scale=alt.Scale(domain=[df.y.min()-10, df.y.max()+10])),
        tooltip=['x', 'y']
    ).properties(
        width=200,
        height=150,
        title=f"{volatility_level}"
    )
    
    return chart

# Function to anonymize user data
def anonymize_data(data):
    if 'name' in data:
        data['name_hash'] = hashlib.sha256(data['name'].encode()).hexdigest()
        del data['name']
    return data

# Function to save user profile to the database
def save_user_profile(user_data):
    conn = get_db_connection()
    c = conn.cursor()
    anonymized_data = anonymize_data(user_data)
    c.execute('''INSERT INTO user_profiles 
                 (name_hash, age, income, dependents, investment_experience, risk_score, risk_tolerance, equity_allocation, income_allocation) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
              (anonymized_data.get('name_hash', 'Anonymous'), anonymized_data['age'], anonymized_data['income'], 
               anonymized_data.get('dependents', 'No'), anonymized_data['investment_experience'], 
               anonymized_data['risk_score'], anonymized_data['risk_tolerance'], 
               anonymized_data['equity_allocation'], anonymized_data['income_allocation']))
    conn.commit()
    conn.close()
    
    # Save data for ML
    ml_data = prepare_data_for_ml(anonymized_data)
    ml_data['risk_score'] = anonymized_data['risk_score']
    csv_path = 'user_data.csv'
    df = pd.DataFrame([ml_data])

    if not os.path.exists(csv_path):
        # First time: write header
        df.to_csv(csv_path, mode='w', header=True, index=False)
    else:
        # Append later rows without header
        df.to_csv(csv_path, mode='a', header=False, index=False)

def generate_summary(answers, risk_tolerance):
    summary = "## Assessment Summary\n\n"
    for question in questions:
        if question['key'] in answers:
            summary += f"**{question['question']}** Your answer: {answers[question['key']]}\n\n"
    
    summary += f"## Risk Tolerance Explanation\n\n"
    summary += f"Based on your answers, your risk tolerance is: **{risk_tolerance}**\n\n"
    summary += "This assessment considers factors such as your age, financial situation, investment experience, and attitude towards market fluctuations. A higher risk tolerance suggests you might be more comfortable with investments that have potential for higher returns but also higher volatility."
    
    return summary
def main():
    # st.set_page_config(layout="wide")
    
    # Custom CSS for larger and evenly spaced investment experience buttons
    st.markdown("""
    <style>
    .investment-button {
        width: 100%;
        height: 120px;
        white-space: normal;
        word-wrap: break-word;
        padding: 10px;
        font-size: 14px;
        line-height: 1.2;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .investment-button .emoji {
        font-size: 24px;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Investment Risk Tolerance Assessment")

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Display progress
    st.progress(st.session_state.step / len(questions))

    # Back and Start Over buttons
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        if st.button("← Back") and st.session_state.step > 0:
            st.session_state.step -= 1
            st.rerun()
    with col3:
        if st.button("↻ Start over"):
            st.session_state.step = 0
            st.session_state.user_answers = {}
            st.session_state.assessment_complete = False
            st.session_state.summary = None
            st.session_state.results = None
            st.rerun()

    # Display current question(s) or results
    if not st.session_state.assessment_complete:
        if st.session_state.step < len(questions):
            q = questions[st.session_state.step]
            
            if q['type'] == 'initial':
                st.header("Tell us a little bit about yourself")
                for sub_q in q['questions']:
                    if sub_q['type'] == 'number':
                        st.session_state.user_answers[sub_q['key']] = st.number_input(sub_q['question'], min_value=0, key=sub_q['key'])
                    elif sub_q['type'] == 'select':
                        st.session_state.user_answers[sub_q['key']] = st.selectbox(sub_q['question'], sub_q['options'], key=sub_q['key'])
                    elif sub_q['type'] == 'radio':
                        st.session_state.user_answers[sub_q['key']] = st.radio(sub_q['question'], sub_q['options'], key=sub_q['key'])
            elif q['type'] == 'employment_income':
                st.header("Tell us a little bit about yourself")
                for sub_q in q['questions']:
                    if sub_q['type'] == 'radio':
                        st.session_state.user_answers[sub_q['key']] = st.radio(sub_q['question'], sub_q['options'], horizontal=True, key=sub_q['key'])
                    elif sub_q['type'] == 'number':
                        st.session_state.user_answers[sub_q['key']] = st.number_input(sub_q['question'], min_value=0, key=sub_q['key'])
                    elif sub_q['type'] == 'buttons':
                        st.write(sub_q['question'])
                        cols = st.columns(3)
                        for i, option in enumerate(sub_q['options']):
                            if cols[i].button(option, key=f"{sub_q['key']}_{i}"):
                                st.session_state.user_answers[sub_q['key']] = option
                                
            elif q['type'] == 'assets_liabilities':
                st.header("Tell us about your assets and liabilities")
                for sub_q in q['questions']:
                    st.session_state.user_answers[sub_q['key']] = st.number_input(sub_q['question'], min_value=0.0, value=0.0, step=1000.0, key=sub_q['key'])
            elif q['type'] == 'radio':
                st.session_state.user_answers[q['key']] = st.radio(q['question'], q['options'], key=q['key'])
            elif q['type'] == 'chart':
                st.write(q['question'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.altair_chart(create_investment_chart("Low Volatility"))
                with col2:
                    st.altair_chart(create_investment_chart("Balanced"))
                with col3:
                    st.altair_chart(create_investment_chart("High Volatility"))
                st.session_state.user_answers[q['key']] = st.radio("Select your preferred volatility level:", q['options'], key=q['key'])
            elif q['type'] == 'text':
                st.session_state.user_answers[q['key']] = st.text_input(q['question'], key=q['key'])
            elif q['type'] == 'multiselect':
                st.session_state.user_answers[q['key']] = st.multiselect(q['question'], q['options'], key=q['key'])
            elif q['type'] == 'select':
                st.session_state.user_answers[q['key']] = st.selectbox(q['question'], q['options'], key=q['key'])
            elif q['type'] == 'slider':
                st.session_state.user_answers[q['key']] = st.slider(q['question'], q['min_value'], q['max_value'], q['step'], key=q['key'])
            elif q['type'] == 'image_buttons':
                st.write(q['question'])
                cols = st.columns(len(q['options']))
                for i, option in enumerate(q['options']):
                    if cols[i].button(f"{option['image']} {option['text']}", key=f"{q['key']}_{i}"):
                        st.session_state.user_answers[q['key']] = option['text']
                st.write(f"Selected: {st.session_state.user_answers.get(q['key'], 'None')}")
            
            # Next button with validation
            if st.button("Next"):
                if q['type'] == 'initial' or q['type'] == 'employment_income' or q['type'] == 'assets_liabilities':
                    # Check if all sub-questions are answered
                    all_answered = all(sub_q['key'] in st.session_state.user_answers for sub_q in q['questions'])
                else:
                    # For other question types, check if the main question is answered
                    all_answered = q['key'] in st.session_state.user_answers and st.session_state.user_answers[q['key']]
                
                if all_answered:
                    st.session_state.step += 1
                    st.rerun()
                else:
                    st.error("Please answer all questions to continue.")

        # Final submission
        elif st.session_state.step == len(questions):
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("Submit", use_container_width=True):
                    risk_score = calculate_risk_score_ml(st.session_state.user_answers)
                    if risk_score is None:
                        st.error("Failed to compute risk score. Please check your inputs.")
                        st.stop
                    risk_tolerance = get_risk_tolerance(risk_score)
                    high_pct, med_pct, low_pct = get_allocation(risk_tolerance)
                    equity_allocation = high_pct + med_pct
                    income_allocation = low_pct
                    

                    st.session_state.results = {
                        'risk_tolerance': risk_tolerance,
                        'equity_allocation': equity_allocation,
                        'income_allocation': income_allocation
                    }

                    user_data = {
                        **st.session_state.user_answers,
                        'risk_score': risk_score,
                        'risk_tolerance': risk_tolerance,
                        'equity_allocation': equity_allocation,
                        'income_allocation': income_allocation
                    }
                    save_user_profile(user_data)
                    st.session_state.summary = generate_summary(st.session_state.user_answers, risk_tolerance)
                    st.session_state.assessment_complete = True
                    st.rerun()

    # Display results
    if st.session_state.assessment_complete:
        # if st.session_state.summary:
        #     st.markdown(st.session_state.summary)
        st.subheader("Risk Tolerance Explanation")
        st.write(f"Based on your answers, your risk tolerance is: **{st.session_state.results['risk_tolerance']}**")
        st.write("This assessment considers factors such as your age, financial situation, investment experience, and attitude towards market fluctuations. A higher risk tolerance suggests you might be more comfortable with investments that have potential for higher returns but also higher volatility.")

        if st.session_state.results:
            st.write(f"Your risk tolerance is: {st.session_state.results['risk_tolerance']}")
            st.write(f"Recommended allocation: {st.session_state.results['equity_allocation']}% Equities, {st.session_state.results['income_allocation']}% Income")
        st.success("Your profile has been saved!")
        st.subheader("Investment Strategy Visualization")
        st.subheader("You are ready to invest!")

# Button to build the portfolio
        if st.button("Build My Portfolio"):
            with st.spinner("Building your portfolio..."):
                # Save the risk tolerance and equity allocation to session state
                st.session_state.risk_tolerance = st.session_state.results['risk_tolerance']
                st.session_state.equity_allocation = st.session_state.results['equity_allocation']

            # Display success message
            st.success("Your portfolio has been built successfully!")

            # Navigate to the Stock Screener page
            st.switch_page("pages/portfolio_summary.py")

        st.subheader("Feedback")
        feedback = st.text_area("Please provide your feedback on the assessment process:")
        if st.button("Submit Feedback"):
            if feedback:
                try:
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT INTO user_feedback (feedback) VALUES (?)", (feedback,))
                    conn.commit()
                    conn.close()
                    st.success("Thank you for your feedback!")
                except Exception as e:
                    st.error(f"An error occurred while saving your feedback: {e}")
            else:
                st.error("Please enter your feedback before submitting.")
        

        # Add disclaimer on the final page
        st.markdown("""
        ---
        **Disclaimer**: This risk assessment tool is for educational purposes only and does not constitute financial advice. 
        Please consult with a qualified financial advisor before making any investment decisions.
        """)


# Add this at the end of your script
def close_db_connection(conn):
    conn.close()

if __name__ == "__main__":
    main()

