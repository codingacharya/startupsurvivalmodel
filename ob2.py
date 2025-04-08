import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# Simulated training step (run once to train and save model)
def train_and_save_model():
    # Simulated dataset
    data = pd.DataFrame({
        'team_experience_years': np.random.randint(1, 20, 100),
        'past_successes': np.random.randint(0, 5, 100),
        'funding_rounds': np.random.randint(1, 10, 100),
        'total_funding': np.random.uniform(0.5, 50, 100),
        'has_top_investor': np.random.choice(['yes', 'no'], 100),
        'debt_ratio': np.random.uniform(0.1, 1.0, 100),
        'cash_flow_score': np.random.uniform(0, 1, 100),
        'industry': np.random.choice(['Fintech', 'Healthtech', 'Edtech', 'E-commerce'], 100),
        'regulatory_risk_score': np.random.uniform(0, 1, 100),
        'customer_retention_rate': np.random.uniform(0, 1, 100),
        'customer_satisfaction_score': np.random.uniform(0, 10, 100),
        'survived_5_years': np.random.choice([0, 1], 100)
    })

    X = data.drop('survived_5_years', axis=1)
    y = data['survived_5_years']

    numeric_cols = [
        'team_experience_years', 'past_successes', 'funding_rounds',
        'total_funding', 'debt_ratio', 'cash_flow_score',
        'regulatory_risk_score', 'customer_retention_rate',
        'customer_satisfaction_score'
    ]
    categorical_cols = ['industry', 'has_top_investor']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, 'startup_model.pkl')

# Uncomment this line if running for the first time to generate a dummy model
# train_and_save_model()

# Load trained model
model = joblib.load('startup_model.pkl')

# Streamlit UI
st.title("🚀 Startup Survival Predictor")

st.markdown("Predict whether a startup will survive beyond 5 years based on key metrics.")

# User Inputs
team_experience = st.slider("Team Experience (Years)", 0, 30, 5)
past_successes = st.slider("Past Successful Startups", 0, 10, 1)
funding_rounds = st.slider("Funding Rounds", 0, 15, 3)
total_funding = st.number_input("Total Funding (in millions USD)", min_value=0.0, value=5.0)
has_top_investor = st.selectbox("Has Top-Tier Investor?", ['yes', 'no'])
debt_ratio = st.slider("Debt Ratio", 0.0, 1.0, 0.5)
cash_flow_score = st.slider("Cash Flow Score", 0.0, 1.0, 0.5)
industry = st.selectbox("Industry", ['Fintech', 'Healthtech', 'Edtech', 'E-commerce'])
reg_risk = st.slider("Regulatory Risk Score", 0.0, 1.0, 0.3)
retention_rate = st.slider("Customer Retention Rate", 0.0, 1.0, 0.7)
satisfaction = st.slider("Customer Satisfaction Score (0–10)", 0.0, 10.0, 7.0)

# Prepare input for prediction
input_df = pd.DataFrame([{
    'team_experience_years': team_experience,
    'past_successes': past_successes,
    'funding_rounds': funding_rounds,
    'total_funding': total_funding,
    'has_top_investor': has_top_investor,
    'debt_ratio': debt_ratio,
    'cash_flow_score': cash_flow_score,
    'industry': industry,
    'regulatory_risk_score': reg_risk,
    'customer_retention_rate': retention_rate,
    'customer_satisfaction_score': satisfaction
}])

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.success(f"✅ The startup is likely to survive (Confidence: {prob:.2%})")
    else:
        st.error(f"⚠️ The startup is unlikely to survive (Confidence: {prob:.2%})")
