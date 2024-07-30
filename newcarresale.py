import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import pickle

with open('best_catboost_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

st.write("""
# Car Resale Price Prediction App
This app predicts the **resale price of cars** based on various features!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    engine_capacity = st.sidebar.slider('Engine Capacity (cc)', 600, 5000, 1600)
    kms_driven = st.sidebar.slider('Kilometers Driven', 0, 300000, 50000)
    max_power = st.sidebar.slider('Max Power (bhp)', 40, 600, 100)
    seats = st.sidebar.slider('Seats', 2, 8, 5)
    mileage = st.sidebar.slider('Mileage (kmpl)', 5.0, 30.0, 15.0)
    insurance = st.sidebar.selectbox('Insurance', ('Third Party', 'Zero Dep', 'Comprehensive', 'Not Available'))
    owner_type = st.sidebar.selectbox('Owner Type', ('First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner'))
    fuel_type = st.sidebar.selectbox('Fuel Type', ('Petrol', 'Diesel', 'LPG', 'Electric'))
    body_type = st.sidebar.selectbox('Body Type', ('Hatchback', 'Sedan', 'SUV', 'MUV', 'Convertible', 'Minivans', 'Pickup'))
    city = st.sidebar.selectbox('City', ('Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata', 'Jaipur', 'Ahmedabad', 'Chandigarh', 'Gurgaon', 'Lucknow'))
    transmission_type = st.sidebar.selectbox('Transmission Type', ('Manual', 'Automatic'))
    car_brand = st.sidebar.selectbox('Car Brand', ('Toyota', 'Coupe', 'Chevrolet', 'Mercedes-Benz', 'Audi', 'Maruti', 'Pickup', 'Porsche', 'Tata', 'Mahindra', 'Volvo', 'Jaguar', 'BMW', 'Cars', 'Datsun', 'Hyundai', 'Honda', 'Wagon', 'Skoda', 'Isuzu', 'Volkswagen'))

    data = {
        'engine_capacity': engine_capacity,
        'kms_driven': kms_driven,
        'max_power': max_power,
        'seats': seats,
        'mileage': mileage,
        'insurance': insurance,
        'owner_type': owner_type,
        'fuel_type': fuel_type,
        'body_type': body_type,
        'city': city,
        'transmission_type': transmission_type,
        'car_brand': car_brand
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

columns_to_encode = ['insurance', 'owner_type', 'fuel_type', 'body_type', 'city', 'transmission_type', 'car_brand']
df_encoded = pd.get_dummies(input_df, columns=columns_to_encode, drop_first=True)

expected_columns = ['engine_capacity', 'kms_driven', 'max_power', 'seats', 'mileage',
                    'insurance_Comprehensive', 'insurance_Not Available', 'insurance_Zero Dep',
                    'owner_type_Second Owner', 'owner_type_Third Owner', 'owner_type_Fourth Owner',
                    'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_LPG',
                    'body_type_Convertible', 'body_type_Hatchback', 'body_type_Minivans',
                    'body_type_MUV', 'body_type_Pickup', 'body_type_Sedan', 'body_type_SUV',
                    'city_Ahmedabad', 'city_Bangalore', 'city_Chandigarh', 'city_Chennai', 'city_Delhi',
                    'city_Gurgaon', 'city_Hyderabad', 'city_Jaipur', 'city_Kolkata', 'city_Lucknow',
                    'city_Mumbai', 'city_Pune', 'transmission_type_Manual',
                    'car_brand_BMW', 'car_brand_Chevrolet', 'car_brand_Coupe', 'car_brand_Datsun',
                    'car_brand_Honda', 'car_brand_Hyundai', 'car_brand_Isuzu', 'car_brand_Jaguar',
                    'car_brand_Mahindra', 'car_brand_Maruti', 'car_brand_Mercedes-Benz',
                    'car_brand_Porsche', 'car_brand_Skoda', 'car_brand_Tata', 'car_brand_Toyota',
                    'car_brand_Volvo', 'car_brand_Volkswagen', 'car_brand_Unknown']

for col in expected_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

df_encoded = df_encoded[expected_columns]

st.subheader('User Input parameters')
st.write(input_df)

prediction = model.predict(df_encoded)

st.subheader('Prediction')
st.write(f'The estimated resale price is: {prediction[0]:.2f} SGD')
