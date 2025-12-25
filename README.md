# Smart Home Energy Consumption Prediction Dashboard

## Project Overview

This project presents a complete IoT and Machine Learning solution for predicting
appliance energy consumption in a smart home environment. The system integrates
sensor data, machine learning models, and a cloud-based interactive dashboard.

Users can input sensor values through a web interface and instantly receive
energy consumption predictions without needing any programming knowledge.

---

## System Architecture

The project follows an end-to-end pipeline:

1. Data collection from indoor sensors and weather stations  
2. Data preprocessing and feature engineering  
3. Machine learning model training and evaluation  
4. Deployment of the trained model using a Streamlit dashboard  
5. Cloud hosting and public access via AWS EC2  

---

## Dataset Description

The dataset contains environmental, indoor, and weather data collected over
approximately **4.5 months**, with measurements recorded every **10 minutes**.

### Data Sources
- Indoor temperature and humidity sensors (multiple rooms)
- Outdoor weather data from **Chievres Airport, Belgium**
- ZigBee wireless sensor network for indoor data transmission

### Main Features
- **Date/Time**: Timestamp of each record  
- **Appliances**: Appliance energy consumption in Wh *(target variable)*  
- **Lights**: Energy consumption of lighting fixtures  
- **T1–T9**: Indoor temperature readings (°C)  
- **RH_1–RH_9**: Indoor humidity readings (%)  
- **Outdoor Weather**: Temperature, pressure, humidity, wind speed, visibility  
- **rv1, rv2**: Random variables for regression validation  

### Data Notes
- Data aggregated at 10-minute intervals  
- Weather data interpolated to align with sensor timestamps  
- Random variables included to validate model robustness  

---

## Machine Learning Workflow

The machine learning pipeline includes the following steps:

1. Load the dataset using Pandas  
2. Convert date column to datetime format  
3. Extract temporal features:
   - Hour
   - Day of week
   - Month
   - Week of year
4. Remove non-useful columns (`date`, `rv1`, `rv2`)
5. Define:
   - Target variable: **Appliances**
   - Feature matrix: all remaining columns
6. Split data into:
   - Training set (80%)
   - Testing set (20%)
7. Apply **StandardScaler** for feature normalization
8. Train and evaluate multiple models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
9. Evaluate models using:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
10. Select **Random Forest Regressor** as the final model
11. Save the trained scaler for deployment

---

## Google Colab Implementation

### `iot_project_colab_code.ipynb`

This notebook contains the full machine learning implementation executed in
**Google Colab**. It includes:

- Dataset loading and cleaning
- Feature engineering and preprocessing
- Model training and comparison
- Performance evaluation
- Saving the preprocessing scaler

Google Colab was used to leverage cloud computing resources, simplify setup,
and enable reproducibility and collaboration.

---

## Dashboard Implementation

### `app.py`

The Streamlit dashboard provides a user-friendly interface that allows users to:

- Enter indoor sensor values and environmental parameters
- Apply the same preprocessing steps used during model training
- Generate real-time energy consumption predictions
- View results instantly in a web browser

This dashboard acts as the deployment layer connecting machine learning
models with end users.

---

## Project File Structure

```
.
├── app.py                     # Streamlit dashboard application
├── iot_project_colab_code.ipynb # ML implementation in Google Colab
├── model_training.py          # Local model training script
├── scaler.pkl                 # Saved StandardScaler
├── IOT_PROJECT_README.md      # Project documentation
├── AWS_Deployment
├── Iot presentation
├── iot project report
├── energydata_complete.csv    # the data sheet     
```

**Note:**  
Note:
Due to file size limitations, random_forest_model.pkl is not included in this repository. The model file is uploaded manually to the AWS EC2 instance during deployment.

All preprocessing, model training, and file generation (app.py, scaler.pkl, and random_forest_model.pkl) are performed using Google Colab.
The dataset is downloaded directly from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

When running the provided Google Colab notebook, the dataset CSV file should be uploaded without manually separating columns. The notebook already handles the correct delimiter using:

df = pd.read_csv("energydata_completee.csv", sep=";")


Because Google Colab runtimes are temporary, all generated files should be downloaded locally after execution and stored in the project directory or uploaded to the deployment environment as needed.

---

## How to Run the Project

### Run Locally (Optional)

Install dependencies and run the dashboard:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## AWS Deployment

The Smart Home Energy Consumption Dashboard is deployed on **Amazon Web Services (AWS)**
using an EC2 instance, making it accessible online.

### AWS Setup
- Service: AWS EC2  
- Operating System: Ubuntu Linux  
- Instance Type: t2.micro  
- Security Group Rules:
  - Port 22 (SSH)
  - Port 8501 (Streamlit dashboard)

---

## Connecting to AWS Using PowerShell

```powershell
cd Desktop
ssh -i key.pem ubuntu@3.123.230.138
```

**Reason:**  
These commands navigate to the SSH key location and establish a secure connection
to the EC2 instance using the assigned public IP address.

---

## Server Setup on AWS

```bash
sudo apt update && sudo apt upgrade -y
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Reason:**  
These steps update the system, create an isolated Python environment, and install
all required project dependencies.

---

## Running the Dashboard on AWS

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Access the dashboard at:

```
http://3.123.230.138:8501
```

---

## Conclusion

This project demonstrates a full end-to-end IoT-based machine learning system,
from data collection and model training to cloud deployment and interactive
visualization. It highlights practical skills in Python, machine learning,
cloud computing, and dashboard development applied to smart home energy management.
