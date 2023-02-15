import streamlit as st
import pickle
import pandas as pd
import pickle



load_model = pickle.load(open('model.sav', 'rb'))

# Load the dataset
df = pd.read_csv('Churn.csv')

# Define the UI
st.sidebar.title('Telecom Customer Churn Prediction')
st.sidebar.markdown('Please fill in the following details to get the prediction.')

# Get the user inputs

# Define the input fields
st.sidebar.title('Customer Churn Prediction')
st.sidebar.header('Input Features')
account_length = st.number_input('Account Length',min_value=1)
voice_plan = st.sidebar.selectbox('Voice Plan', ['Yes', 'No'])
intl_plan = st.sidebar.selectbox('International Plan', ['Yes', 'No'])
intl_calls = st.sidebar.slider('International Calls', 0, 20, 10)
intl_charge = st.sidebar.slider('International Charge', 0.0, 5.40, 2.0, step=0.1)
day_calls = st.sidebar.slider('Day Calls', 0, 165, 100)
day_charge = st.sidebar.slider('Day Charge', 0.0, 59.0, 32.0, step=0.1)
eve_calls = st.sidebar.slider('Evening Calls', 0, 170, 100)
eve_charge = st.sidebar.slider('Evening Charge', 0.0, 30.0, 25.0, step=0.1)
night_calls = st.sidebar.slider('Night Calls', 0, 175, 100)
night_charge = st.sidebar.slider('Night Charge', 0.0, 17.0, 10.0, step=0.1)
customer_calls = st.sidebar.slider('Customer Service Calls', 0, 9, 5)
area_code = st.sidebar.selectbox('Area Code', ['408', '415', '510'])

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
