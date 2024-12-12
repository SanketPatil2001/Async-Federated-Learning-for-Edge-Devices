# Federated Learning with Synchronous and Asynchronous Updates
## Overview
This repository showcases Synchronous and Asynchronous Federated Learning (AFL) implementations using the Flower framework, simulating privacy-preserving edge network computing. The project demonstrates training a global machine learning model across multiple clients without sharing raw data, ensuring privacy and security.

## The Repository includes:

1. Synchronous Federated Learning: All clients send updates in sync to the server.
2. Asynchronous Federated Learning: Clients can send updates independently, without waiting for others.
3. GUI and Visualization: Real-time plotting of global and per-client accuracy/loss to understand model performance.
## Features
1. Federated Learning: Train a global model while keeping data localized to clients.
2. Synchronous and Asynchronous Updates: Implement both strategies for comparison.
3. Privacy Preservation: Sensitive data remains on client devices.
4. Edge Simulation: Docker containers represent edge devices.
5. Real-Time Visualization: Interactive GUI for monitoring performance.
## Project Structure
1. sync_federated_learning/: Contains code for synchronous federated learning.
2. async_federated_learning/: Contains code for asynchronous federated learning.
3. gui_visualization/: Contains the GUI-based real-time plotting code.
## Setup and Execution
### Prerequisites
1. Python 3.8+
2. Docker
3. Git
## Installation
### Clone this repository:
git clone https://github.com/SanketPatil2001/federated-learning.git  

cd federated-learning  
### Install Python dependencies:
pip install -r requirements.txt  
### Execution
1. Start the Server: Navigate to either sync_federated_learning or async_federated_learning
python server.py  
2. Start Clients:
Run clients in Docker containers or different machines:
python client.py  
3. Visualization:
Launch the GUI for monitoring performance:
python gui.py  
### Visual Results
1. Real-time plots showing:
2. Global Model Performance: Aggregated accuracy and loss.
3. Client-Specific Performance: Accuracy and loss for each participating client.
Demonstration of synchronous and asynchronous updates in action.
## How It Works
### Data Training:
Clients train locally on the MNIST dataset or custom datasets.
### Global Model Update:
1. Synchronous: All clients contribute together.
2. Asynchronous: Clients contribute independently.
### Edge Simulation:
Docker containers emulate edge devices.
Future Enhancements
Add encryption for enhanced security.
Extend support for heterogeneous datasets.
Deploy on cloud platforms (e.g., AWS, Azure) for large-scale testing.
## References
1. Flower Framework: Flower.dev
2. MNIST Dataset: Yann LeCun's MNIST Database
