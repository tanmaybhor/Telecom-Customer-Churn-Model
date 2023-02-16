import streamlit as st
import pickle
import pandas as pd
import pickle



load_model = pickle.load(open('model.sav', 'rb'))

# Load the dataset
df = pd.read_csv('Churn.csv')

# Define the UI
st.title('Telecom Customer Churn Prediction')
st.markdown('Please fill in the following details to get the prediction.')

# Get the user inputs

# Define the input fields
st.title('Customer Churn Prediction')
st.header('Input Features')
account_length = st.number_input('Account Length',min_value=1)
voice_plan = st.selectbox('Voice Plan', ['Yes', 'No'])
intl_plan = st.selectbox('International Plan', ['Yes', 'No'])
intl_calls = st.number_input('International Calls', min_value=1)
intl_charge = st.number_input('International Charge', min_value=1)
day_calls = st.number_input('Day Calls', min_value=1)
day_charge = st.number_input('Day Charge', min_value=1)
eve_calls = st.number_input('Evening Calls',min_value=1)
eve_charge = st.number_input('Evening Charge',min_value=1)
night_calls = st.number_input('Night Calls', min_value=1)
night_charge = st.number_input('Night Charge',min_value=1)
customer_calls = st.number_input('Customer Service Calls',min_value=1)
area_code = st.selectbox('Area Code', ['408', '415', '510'])

# Convert categorical variables to numerical
voice_plan = 1 if voice_plan == 'Yes' else 0
intl_plan = 1 if intl_plan == 'Yes' else 0
area_code_408 = 1 if area_code == '408' else 0
area_code_415 = 1 if area_code == '415' else 0

# Scale the input features
input_data = pd.DataFrame({
    'account_length': [account_length],
    'voice_plan': [voice_plan],
    'intl_plan': [intl_plan],
    'intl_calls': [intl_calls],
    'intl_charge': [intl_charge],
    'day_calls': [day_calls],
    'day_charge': [day_charge],
    'eve_calls': [eve_calls],
    'eve_charge': [eve_charge],
    'night_calls': [night_calls],
    'night_charge': [night_charge],
    'customer_calls': [customer_calls],
    'area_code_408': [area_code_408],
    'area_code_415': [area_code_415]
})

# Predict the churn probability
churn_probability = load_model.predict_proba(input_data)[:, 1][0]

# Display the output
st.write('### Churn Probability')
st.write(f'The probability of this customer churning is {churn_probability:.2%}')
