# Data processing & Random Forest training

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load dataset (CSV needs to be uploaded later)
df = pd.read_csv("energydata_completee.csv", sep=";")

# Datetime processing
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['week_of_year'] = df['date'].dt.isocalendar().week

# Drop unnecessary columns
df = df.drop(['date','rv1','rv2'], axis=1)

# Target & features
y = df['Appliances']
X = df.drop('Appliances', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest training
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Save model & scaler
import os
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Saved: models/random_forest_model.pkl and models/scaler.pkl")
