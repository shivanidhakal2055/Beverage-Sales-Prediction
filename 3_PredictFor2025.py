import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import pickle

print("1. Read the Tamplate data fro the Prediction ....")
PDF = pd.read_csv(r"files\2025_tamplate_predict.csv",parse_dates=['Order_Date']) 


PDF.drop('Unnamed: 0',axis=1,inplace=True)
# Date-based features
PDF['year'] = PDF['Order_Date'].dt.year
PDF['month'] = PDF['Order_Date'].dt.month
PDF['day'] = PDF['Order_Date'].dt.day
PDF['day_of_week'] = PDF['Order_Date'].dt.dayofweek
PDF['is_weekend'] = (PDF['day_of_week'] >= 5).astype(int)

PDF['quantity_lag_1'] = 0
PDF['quantity_lag_2'] = 0
PDF['quantity_lag_3'] = 0
PDF['rolling_mean_3'] = 0
PDF['rolling_std_3'] = 0


filename = 'label_encoders.pkl'
with open(filename, 'wb') as file:
    label_encoders = pickle.load(file)

# Encode categorical columns
cat_cols = ['Customer_Type', 'Product', 'Category', 'Region']

for col in cat_cols:
    le = label_encoders[col]
    PDF[col] = le.fit_transform(PDF[col])

features = [
        'Customer_Type', 'Product', 'Category', 'Unit_Price', 'Discount', 'Region',
        'year', 'month', 'day', 'day_of_week', 'is_weekend',
        'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3',
        'rolling_mean_3', 'rolling_std_3'
    ]

filename = 'Models\RandomForestRegressor.pkl'
with open(filename, 'wb') as file:
    rf_model = pickle.load (file)

rf_model_p = rf_model.predict(PDF[features])
rf_model_p = list(map(lambda x: round(x),rf_model_p))
PDF['Quantity'] = rf_model_p
PDF['Total_Price'] = PDF.apply(lambda row: row['Unit_Price'] * row['Quantity'] - (row['Discount'] * row['Unit_Price'] * row['Quantity']), axis=1)
PDF['Total_Price'] = PDF['Total_Price'].apply(lambda x: round(x,2))
PDF.to_csv('RandomForestRegressor2025.csv')

filename = 'Models\XGBRegressor.pkl'
with open(filename, 'wb') as file:
    xgb_model = pickle.load (file)

xgb_model_p = xgb_model.predict(PDF[features])
xgb_model_p = list(map(lambda x: round(x),xgb_model_p))
PDF['Quantity'] = rf_model_p
PDF['Total_Price'] = PDF.apply(lambda row: row['Unit_Price'] * row['Quantity'] - (row['Discount'] * row['Unit_Price'] * row['Quantity']), axis=1)
PDF['Total_Price'] = PDF['Total_Price'].apply(lambda x: round(x,2))
PDF.to_csv('XGBRegressor.csv')

filename = 'Models\lgb_model.pkl'
with open(filename, 'wb') as file:
    lgb_model = pickle.load (file)
lgb_model_p = lgb_model.predict(PDF[features])
lgb_model_p = list(map(lambda x: round(x),lgb_model_p))
PDF['Quantity'] = lgb_model_p
PDF['Total_Price'] = PDF.apply(lambda row: row['Unit_Price'] * row['Quantity'] - (row['Discount'] * row['Unit_Price'] * row['Quantity']), axis=1)
PDF['Total_Price'] = PDF['Total_Price'].apply(lambda x: round(x,2))
PDF.to_csv('LGBMRegressor.csv')
