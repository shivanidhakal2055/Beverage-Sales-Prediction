import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

print("1. Read the Preprocessed data ....")
df = pd.read_csv(r'/Users/shivanidhakal/Documents/Shivani/files/2025_tamplate_predict.csv', parse_dates=['Order_Date']) 

features = [
    'Customer_Type', 'Product', 'Category', 'Unit_Price', 'Discount', 'Region',
    'year', 'month', 'day', 'day_of_week', 'is_weekend',
    'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3',
    'rolling_mean_3', 'rolling_std_3'
]
target = 'Quantity'

X = df[features]
y = df[target]

#2. Train/Test Split based on time
print("2. Performaing a Train test Split ....")
train_size = int(0.8 * len(df))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print("3. Traning Models ....")
# 6. Train Models
print("4. Starting Random Forest ....")
## Random Forest
rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10, verbose=0,n_jobs=-1)
rf_model.fit(X_train, y_train)

import pickle
filename = 'RandomForestRegressor.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rf_model, file)
print("5.Random Forest Completed ....")

print("6. Starting XGBRegressor ....")
## XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, max_depth=10,n_jobs=-1)
xgb_model.fit(X_train, y_train)
filename = 'XGBRegressor.pkl'
with open(filename, 'wb') as file:
    pickle.dump(xgb_model, file)
print("7.XGBRegressor Completed ....")

print("8. Starting LightGBM ....")
## LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, max_depth=10,n_jobs=-1)
lgb_model.fit(X_train, y_train)
filename = 'lgb_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(lgb_model, file)

print("9.LightGBM Completed ....")
print("10 Traning and validation succesful ..")