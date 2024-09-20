import streamlit as st
import pickle
import numpy as np
import plotly.express as px
import pandas as pd


df= pd.read_csv(r'C:\Users\Mary\Desktop\Project\fraud test.csv')

with open(r'C:\Users\Mary\Desktop\Project\model.pkl', 'rb') as file:
    model = pickle.load(file)

with open(r'C:\Users\Mary\Desktop\Project\scaler.pkl', 'rb') as file:
    sc = pickle.load(file)
    
with open(r'C:\Users\Mary\Desktop\Project\gender_encoder.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

with open(r'C:\Users\Mary\Desktop\Project\category_encoder.pkl', 'rb') as file:
    category_encoder = pickle.load(file)

with open(r'C:\Users\Mary\Desktop\Project\city_encoder.pkl', 'rb') as file:
    city_encoder = pickle.load(file)

with open(r'C:\Users\Mary\Desktop\Project\state_encoder.pkl', 'rb') as file:
    state_encoder = pickle.load(file)

with open(r'C:\Users\Mary\Desktop\Project\merchant_encoder.pkl', 'rb') as file:
    merchant_encoder = pickle.load(file)

with open(r'C:\Users\Mary\Desktop\Project\job_encoder.pkl', 'rb') as file:
    job_encoder = pickle.load(file)
    
    
    
    
def main():
    
    st.title('Credit Card Fraud Detection App')
    st.subheader('Developed By: Cyntech')
    st.write('This app uses machine learning to predict fraudulent transactions through neccessary information that would be entered by the Card holder.')
    from PIL import Image
    img = Image.open(r"C:\Users\Mary\Desktop\Project\Marketing Quotes.jpg")
    st.image(img, width=500)
    st.sidebar.title('Prediction features')

# User inputs
    st.sidebar.subheader('Enter your details here')

#Categorical features
    gender = st.sidebar.selectbox('Gender', gender_encoder.classes_)
    job = st.sidebar.selectbox('Job', job_encoder.classes_)
    category = st.sidebar.selectbox('Category', category_encoder.classes_)
    city = st.sidebar.selectbox('City', city_encoder.classes_)
    state = st.sidebar.selectbox('State', state_encoder.classes_)
    merchant = st.sidebar.selectbox('Merchant', merchant_encoder.classes_)
    
    
    # Numerical features
    age = st.sidebar.number_input('Age', min_value=18, max_value=99)
    Amount_withdraw = st.sidebar.number_input('Amount', min_value=9, max_value=3000)
    latitude = st.sidebar.slider('Latitude', min_value=20.0271, max_value=65.6899)
    longitude=st.sidebar.slider('Longitude', min_value=-165.6723, max_value = -67.9503)
    merchant_lat=st.sidebar.slider('Merchant latitude',min_value=19.163455, max_value=65.951727)
    merchant_long=st.sidebar.slider('Merchant Longitude', min_value=-166.464422, max_value=-66.960745)
    
    data = {
        'merchant': [merchant],
        'category': [category],
        'amt': [Amount_withdraw],  # Ensure this matches the name used during training
        'gender': [gender],
        'city': [city],
        'state': [state],
        'lat': [latitude],  # Ensure this matches the name used during training
        'long': [longitude],  # Ensure this matches the name used during training
        'job': [job],
        'merch_lat': [merchant_lat],
        'merch_long': [merchant_long],  # Ensure this matches the name used during training
        'Age': [age],
        }
    
    input_df = pd.DataFrame(data, index = [0])


    input_df.rename(columns={
        'amount_withdrawn': 'amt',  # Rename to match training feature name
        'latitude': 'lat',
        'longitude': 'lng',
        'merchant_long': 'merchant_lng'},inplace=True)
    
            # Encoding categorical features
    input_df['gender'] = gender_encoder.transform(input_df['gender'])
    input_df['job'] = job_encoder.transform(input_df['job'])
    input_df['category'] = category_encoder.transform(input_df['category'])
    input_df['city'] = city_encoder.transform(input_df['city'])
    input_df['state'] = state_encoder.transform(input_df['state'])
    input_df['merchant'] = merchant_encoder.transform(input_df['merchant'])
    
    
            # Scaling numerical features
    scaled_features = sc.transform(input_df)
    
    if st.button('Predict'):
        prediction = model.predict(scaled_features)
        prediction_prob = model.predict_proba(scaled_features)
        print(prediction)
        
        st.subheader('Prediction')
        st.write('Fraud' if prediction[0] == 1 else 'Not Fraud')
        st.write(f'Probability of Fraud: {prediction_prob[0][1]:.2f}')
        
if __name__ == "__main__":
    main()

