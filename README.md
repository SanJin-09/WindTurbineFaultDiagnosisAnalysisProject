# Wind Turbine Fault Diagnosis Analysis prject
**Big Data coursework report**\
\
**Team member：WU Xinyan, SUN Runze**
## Project Description
_This project focuses on developing a big data analytics solution for wind turbine fault diagnosis by leveraging SCADA (Supervisory Control and Data Acquisition) system data. The goal is to predict and classify turbine faults (e.g., gearbox failure, bearing faults) using machine learning models trained on sensor data (temperature, vibration, rotational speed, etc.)._\
\
**Key Components:**
+ **Data Preprocessing:** Cleans and aligns SCADA data with fault timestamps.
+ **Feature Engineering:** Extracts time-domain, frequency-domain (FFT), and statistical features.
+ **Modeling:** Random Forest, Gradient Boosting, XGBoost, Neural Networks.
+ **Evaluation:** Assesses performance using accuracy, recall, and F1-score.
+ **Visualization:** Analyzes sensor trends and model results via interactive plots.
## Project Achievements
+ **Wind Turbine Fault Data Understanding and Preprocessing.**\
Parse and `clean scada_data.csv` and `fault_data.csv`.\
Align the timestamps of the datasets and extract sensor data before/after the fault.
+ **Data feature engineering.**\
Extract time domain, frequency domain or statistical features (mean, variance, maximum/minimum value, FFT, etc.).
+ **Fault classification model training.**\
Training multiple fault identification models: _Random Forest, Gradient Boosting, XGBoost, Neural Networks_.
+ **Model performance evaluation and analysis.**\
Use metrics (accuracy, recall, F1 score) to evaluate model performance.\
Analyze the impact of different sensors on fault types.
+ **Data Visualization.**\
Visualize sensor trends before/after a fault Compare model predictions to actual labels.
