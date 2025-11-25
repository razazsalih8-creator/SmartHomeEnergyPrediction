# Smart Home Energy Prediction Project

## 📌 Overview
This project aims to build a machine learning model capable of predicting **appliance energy consumption** in a smart home environment.  

The dataset contains **temperature, humidity, weather, and environmental sensor readings** collected from different rooms in the house over ~4.5 months, recorded every 10 minutes. The data was collected using a ZigBee wireless sensor network, merged with weather data from Chievres Airport (Belgium), and includes two random variables for testing purposes.

The goal of the project is to develop a **dashboard** where users can enter sensor values and instantly see the predicted energy usage.

---

## 📌 Dataset Information
- **Date/Time**: year-month-day hour:minute:second  
- **Appliances**: energy use in Wh  
- **lights**: energy use of light fixtures in Wh  
- **T1-RH_1, …, T9-RH_9**: Temperature and Humidity in various rooms in Celsius and %  
- **T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint**: Weather data from Chievres station  
- **rv1, rv2**: Random variables for regression testing  

**Data Notes:**  
- 10-minute interval averages from raw sensor readings (~3.3 min per node)  
- Weather data interpolated to match the sensor timestamp  
- Two random variables included to test non-predictive features  

---

## 📌 Preprocessing & Workflow
1. Loaded the dataset (`energydata_completee.csv`) using Pandas with the correct separator (`;`).  
2. Converted the `date` column to datetime and extracted new features:  
   - `hour`, `day_of_week`, `month`, `week_of_year`  
3. Dropped unnecessary columns: `date`, `rv1`, `rv2`  
4. Defined **target variable** (`Appliances`) and features (`X` = all other columns)  
5. Split the dataset into **training** (80%) and **testing** (20%) sets  
6. Standardized the features using `StandardScaler` to ensure all inputs are on the same scale  
7. Trained multiple regression models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting  
8. Evaluated models using **MAE** and **RMSE**, and selected **Random Forest** as the best-performing model  
9. Saved the **scaler** (`scaler.pkl`) for use in the dashboard to preprocess user inputs  

---

## 📌 Project Files Description
1. **model_training.ipynb**  
   - Full machine learning workflow including preprocessing, model training, evaluation, and saving the scaler  

2. **app.py**  
   - Streamlit dashboard that loads the scaler, receives user sensor inputs, scales them, and predicts energy usage  

3. **scaler.pkl**  
   - Contains the trained `StandardScaler` to ensure the same preprocessing is applied to new inputs  

4. **requirements.txt**  
   - Lists all Python libraries required for running the project (`pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`, etc.)  

5. **README.md**  
   - This file: provides full documentation, dataset info, workflow, and instructions  

> ⚠️ `random_forest_model.pkl` is **not included** in the GitHub repository because it is a large file. It will be used in the **AWS deployment** for running the dashboard.  

---

## 📌 How to Run the Project

**Locally (optional):**  
1. Install required libraries:  
```bash
pip install -r requirements.txt
