
# Smart Home Energy Prediction Project

## 📌 Overview
This project aims to build a machine learning model capable of predicting appliance energy consumption in a smart home environment.  
The dataset contains temperature, humidity, weather, and environmental sensor readings collected from different rooms in the house.

The goal of the project is to develop a dashboard where the user can enter sensor values and instantly see the predicted energy usage.

---

## 📌 Objectives
- Analyze the smart home dataset.
- Preprocess and scale the data for machine learning.
- Train several regression models and compare their performance.
- Select the best model (Random Forest).
- Save the model and scaler for deployment.
- Build an interactive Streamlit dashboard for real-time prediction.

---

## 📌 Project Files Description

### **1. model_training.ipynb**
This notebook contains the full machine learning workflow:
- Data loading
- Splitting into training and testing sets
- Standardization
- Training multiple regression models
- Evaluating MAE and RMSE
- Saving the trained model and scaler

---

### **2. app.py**
This is the Streamlit web application that serves as the user interface.  
It loads the trained model and scaler, receives sensor inputs from the user, processes them, and displays the predicted energy consumption.

---

### **3. random_forest_model.pkl**
This file contains the trained Random Forest regression model.  
It is produced during training and used only by the Streamlit dashboard.

---

### **4. scaler.pkl**
This file stores the StandardScaler used to normalize the dataset before training.  
It ensures that incoming user inputs receive the same preprocessing as the training data.

---

### **5. requirements.txt**
Lists all Python libraries needed to run the project.

---

### **6. README.md**
Provides documentation for the project, explaining its purpose, structure, files, and how to run it.

---

## 📌 How to Run the Project

1. Install the required libraries:
    ```
    pip install -r requirements.txt
    ```

2. Run the Streamlit dashboard:
    ```
    streamlit run app.py
    ```

3. Enter the sensor values in the interface and the model will predict the energy consumption.

---

## 📌 Conclusion
This project demonstrates how machine learning can be applied to real-world smart home applications.  
By integrating the trained model with a Streamlit interface, we provide an interactive tool that predicts appliance energy consumption based on environmental data.
