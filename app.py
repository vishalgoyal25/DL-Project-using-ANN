import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle



# Load the trained model
model = load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



## Streamlit App
st.title('Customer Churn Prediction')


# User Inputs
CreditScore = st.number_input('Credit Score')
Gender = st.selectbox('Gender', label_encoder_gender.classes_)
Age = st.slider('Age', 18, 92)
Tenure = st.slider('Tenure', 0, 10)
Balance = st.number_input('Balance')
NumOfProducts = st.slider('Number of Products', 1, 4)
HasCrCard = st.selectbox('Has Credit Card', [0, 1])
IsActiveMember = st.selectbox('Is Active Member', [0, 1])
EstimatedSalary = st.number_input('Estimated Salary')

Geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Gender': [label_encoder_gender.transform([Gender])[0]],    
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})


# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
