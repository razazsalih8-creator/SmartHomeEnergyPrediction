🏠 Smart Home Energy Consumption Prediction Dashboard
📌 Project Overview

This project presents a complete IoT and Machine Learning solution for predicting appliance energy consumption in a smart home environment.
It combines data collected from multiple indoor sensors, weather stations, machine learning models, and a cloud-based interactive dashboard.

The system enables users to input sensor values through a web dashboard and instantly receive a prediction of energy consumption, without requiring any programming background.

📊 Dataset Description

The dataset contains environmental, indoor, and weather data collected over approximately 4.5 months, with measurements recorded every 10 minutes.

Data Sources

Indoor temperature and humidity sensors deployed in multiple rooms

Outdoor weather data from Chievres Airport (Belgium)

ZigBee wireless sensor network for indoor data transmission

Main Features

Date/Time: Timestamp of each record

Appliances: Appliance energy consumption in Wh (target variable)

Lights: Energy consumption of lighting fixtures in Wh

T1–T9: Indoor temperature readings (°C)

RH_1–RH_9: Indoor humidity readings (%)

T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint: Outdoor weather features

rv1, rv2: Random variables for regression testing

Data Notes

Data aggregated at 10-minute intervals

Weather data interpolated to align with sensor timestamps

Random variables included to validate model robustness

🧠 Machine Learning Workflow

The machine learning pipeline follows these steps:

Load the dataset using Pandas with the correct delimiter (;)

Convert the date column to datetime format

Extract additional temporal features:

hour

day_of_week

month

week_of_year

Remove non-useful columns:

date

rv1

rv2

Define:

Target variable: Appliances

Feature matrix: all remaining columns

Split data into training (80%) and testing (20%)

Apply StandardScaler to normalize feature values

Train multiple regression models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Evaluate models using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Select Random Forest Regressor as the final model

Save the trained scaler for deployment

📓 Google Colab Implementation
iot_project_colab_code.ipynb

This notebook contains the full machine learning implementation executed in Google Colab.

It includes:

Dataset loading and cleaning

Feature engineering and preprocessing

Model training and comparison

Model performance evaluation

Saving the preprocessing scaler

Google Colab was used to:

Leverage cloud computing resources

Avoid local environment setup issues

Enable reproducibility and easy collaboration

The notebook serves as the development and experimentation environment, while the dashboard uses the trained components for prediction.

🖥️ Dashboard Implementation
app.py

The Streamlit dashboard provides a user-friendly interface that allows users to:

Enter indoor sensor values and environmental parameters

Automatically apply the same preprocessing steps used during training

Generate real-time energy consumption predictions

View results instantly in a web browser

The dashboard acts as the final deployment layer, bridging machine learning with user interaction.

📁 Project File Structure
File	Description
iot_project_colab_code.ipynb	Full ML implementation in Google Colab
model_training.ipynb	Local notebook for training and evaluation
app.py	Streamlit dashboard for prediction
scaler.pkl	Saved StandardScaler for preprocessing
requirements.txt	Required Python dependencies
README.md	Project documentation

⚠️ Note:
random_forest_model.pkl is not included in this repository due to file size limitations.
It is uploaded manually to the AWS EC2 instance during deployment.

▶️ Running the Project Locally (Optional)
pip install -r requirements.txt
streamlit run app.py


Make sure the trained model file (random_forest_model.pkl) is present in the project directory.

☁️ AWS Deployment

The dashboard was deployed on Amazon Web Services (AWS) using an EC2 instance, allowing remote access through a web browser.

AWS Setup

Service: AWS EC2

Operating System: Ubuntu Linux

Instance Type: t2.micro

Security Group Rules:

Port 22 (SSH)

Port 8501 (Streamlit dashboard)

Deployment Steps

Connect to the EC2 instance via SSH using Windows PowerShell

Create and activate a Python virtual environment

Install required dependencies

Upload project files and trained model

Run the dashboard:

streamlit run app.py --server.port 8501 --server.address 0.0.0.0


Access the dashboard:

http://<EC2_PUBLIC_IP>:8501

✅ Conclusion

This project demonstrates an end-to-end IoT-based machine learning system, combining data analysis, model training, cloud deployment, and interactive visualization.
It highlights the practical application of machine learning in smart home energy management and showcases real-world skills in Python, cloud computing, and dashboard development.
