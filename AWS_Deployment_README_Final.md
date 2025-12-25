# AWS Dashboard Deployment

This section documents how the project dashboard was deployed on Amazon Web Services (AWS)
using an EC2 instance. It explains the purpose, tools, exact commands used, and the reason
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
- Authentication: SSH key (key.pem)
- Public IP Address: 3.123.230.138
- Open Ports:
  - 22 (SSH)
  - 8501 (Streamlit dashboard)

---

## Connecting to AWS Using PowerShell

The connection to the EC2 instance was established using SSH from Windows PowerShell.
The SSH private key file was stored on the Desktop.

```powershell
cd Desktop
```

Reason:  
This command navigates to the Desktop directory where the SSH key file (key.pem) is stored.

```powershell
ssh -i key.pem ubuntu@3.123.230.138
```

Reason:  
This command creates a secure SSH connection to the AWS EC2 instance using the private
SSH key. The username `ubuntu` is the default for Ubuntu-based EC2 instances, and
`3.123.230.138` is the public IP address of the server.

---

## Server Setup on AWS

After successfully connecting to the EC2 instance, the following commands were executed
on the Ubuntu server.

```bash
sudo apt update && sudo apt upgrade -y
```

Reason:  
This command updates all system packages to the latest versions to ensure system
security, stability, and compatibility.

---

```bash
python3 -m venv venv
source venv/bin/activate
```

Reason:  
A Python virtual environment was created and activated to isolate the project
dependencies from the system-wide Python installation.

---

```bash
pip install streamlit pandas numpy scikit-learn
```

Reason:  
These libraries are required for building the Streamlit dashboard, processing data,
and running the machine learning prediction model.

---

## Running the Dashboard

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Reason:  
This command starts the Streamlit application on port 8501 and allows the dashboard
to be accessed from external devices over the internet.

---

## Accessing the Dashboard

The deployed dashboard was accessed through a web browser using the following URL:

```
http://3.123.230.138:8501
```

This confirmed that the application was running successfully on the AWS EC2 instance
and was publicly accessible.

---

## Outcome

The dashboard was successfully deployed on AWS EC2 and accessed remotely.
Users were able to enter numerical input values and receive prediction results
in real time, confirming correct cloud deployment and system functionality.

<img width="943" height="529" alt="image" src="https://github.com/user-attachments/assets/15473a34-b29f-4bd6-865e-c61cb7aa01af" />

Figure: The deployed Streamlit dashboard displayed in Google Chrome after successfully running the application on the AWS EC2 instance.
<img width="945" height="525" alt="image" src="https://github.com/user-attachments/assets/cae98d82-056e-4b82-aaf2-9238bdc606c8" />
Figure  : Dashboard interface showing user input values set to 0.01 for all sensor parameters and the resulting energy consumption prediction displayed on the screen.
