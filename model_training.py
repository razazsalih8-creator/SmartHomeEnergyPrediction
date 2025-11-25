# model_training.py
# Script to train RandomForest on energydata_sample.csv and save model+scaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 1) Load data (ensure data/energydata_sample.csv exists in repo or local)
df = pd.read_csv("data/energydata_sample.csv")

# 2) Basic cleaning and parse date
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df = df.drop(columns=['date'], errors='ignore')

# 3) Drop any irrelevant or placeholder columns if exist
for col in ['rv1','rv2']:
    if col in df.columns:
        df = df.drop(columns=[col])

# 4) Target and features
y = df['Appliances']
X = df.drop(columns=['Appliances'])

# 5) Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6) Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7) Train Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# 8) Evaluate
y_pred = rf.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# 9) Save model and scaler to models/ folder (locally after running)
import os
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Saved models/random_forest_model.pkl and models/scaler.pkl")
