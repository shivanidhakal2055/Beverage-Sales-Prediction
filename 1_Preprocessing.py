import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load your dataset
df = pd.read_csv(r'/Users/shivanidhakal/Documents/Shivani/files/2025_tamplate_predict.csv', parse_dates=['Order_Date'])  # Adjust filename and date column

# 2. Sort by date
df = df.sort_values('Order_Date')

# 3. Feature Engineering

# Date-based features
df['year'] = df['Order_Date'].dt.year
df['month'] = df['Order_Date'].dt.month
df['day'] = df['Order_Date'].dt.day
df['day_of_week'] = df['Order_Date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Lag features
for lag in [1, 2, 3]:
    df[f'quantity_lag_{lag}'] = df['Quantity'].shift(lag)

# Rolling mean and std
df['rolling_mean_3'] = df['Quantity'].shift(1).rolling(window=3).mean()
df['rolling_std_3'] = df['Quantity'].shift(1).rolling(window=3).std()

# Drop rows with NaNs from shifting
df = df.dropna()

# Encode categorical columns
cat_cols = ['Customer_Type', 'Product', 'Category', 'Region']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df.to_csv(r"/Users/shivanidhakal/Documents/Shivani/files/2025_tamplate_predict.csv",index_label=False)

filename = 'Models\label_encoders.pkl'
with open(filename, 'wb') as file:
    pickle.dump(label_encoders, file)