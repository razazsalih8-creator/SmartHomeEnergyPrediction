# AWS Dashboard Deployment

This section documents how the project dashboard was deployed on Amazon Web Services (AWS)
using an EC2 instance. It explains the purpose, tools, commands used, and the reason
for each step in a clear and academic manner.

---

## Overview

The dashboard is an interactive machine learning web application built using Streamlit.
It was deployed on AWS EC2 to allow remote access through a web browser without requiring
local execution of the code.

---

## Tools and Technologies

- AWS EC2 (Cloud Virtual Server)
- Ubuntu Linux (Operating System)
- Windows PowerShell (SSH Access)
- Python 3
- Virtual Environment (venv)
- Streamlit
- Scikit-learn
- SSH Key (.pem)

---

## EC2 Instance Configuration

- AMI: Ubuntu (64-bit)
- Instance Type: t2.micro
- Authentication: SSH key (.pem)
- Open Ports:
  - 22 (SSH)
  - 8501 (Streamlit dashboard)

---

## Connecting to AWS Using PowerShell

The connection to the EC2 instance was established using SSH from Windows PowerShell.

```powershell
cd Desktop
```

Reason:  
This command navigates to the directory where the SSH key file is stored.

```powershell
ssh -i key.pem ubuntu@<EC2_PUBLIC_IP>
```

Reason:  
This command creates a secure connection to the EC2 instance using the private SSH key.

---

## Server Setup on AWS

```bash
sudo apt update && sudo apt upgrade -y
```

Reason:  
Updates system packages to ensure stability and security.

```bash
python3 -m venv venv
source venv/bin/activate
```

Reason:  
Creates and activates a virtual environment to isolate project dependencies.

```bash
pip install streamlit pandas numpy scikit-learn
```

Reason:  
Installs required libraries for the dashboard and machine learning model.

---

## Running the Dashboard

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Reason:  
Starts the Streamlit application and allows access from external devices.

---

## Accessing the Dashboard

The dashboard can be accessed through a web browser using:

```
http://<EC2_PUBLIC_IP>:8501
```

---

## Outcome

The dashboard was successfully deployed on AWS EC2 and accessed remotely.
The application accepted user inputs and returned prediction results in real time,
confirming correct cloud deployment and functionality.
