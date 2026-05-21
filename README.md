# IoT Sensor Data Classification and Fault Detection

## Overview
[cite_start]This repository implements a multi-model machine learning pipeline designed to clean, scale, and classify continuous environmental sensor telemetry[cite: 884, 1186]. [cite_start]Utilizing an aggregated dataset of **37,674 records** capturing temperature and humidity values, the pipeline evaluates five supervised machine learning architectures to accurately differentiate between fault-free operations (Label 0) and hardware/environmental malfunctions (Label 1)[cite: 906, 910].

## Technical Stack
* [cite_start]**Languages & Frameworks:** Python, TensorFlow / Keras [cite: 891, 1424-1425]
* [cite_start]**Core Libraries:** Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib [cite: 908, 1187-1190, 1193-1194]
* [cite_start]**Implemented Models:** K-Nearest Neighbors (KNN), Decision Trees, Random Forests, Support Vector Machines (SVM), and a Sequential Deep Neural Network[cite: 908].

## Pipeline Architecture

### 1. Data Preprocessing & Cleaning
* [cite_start]**Feature Scaling:** Applied a standard scaling transform ($\mu = 0, \sigma = 1$) to account for divergent continuous ranges across humidity and temperature measurements [cite: 914-915].
* [cite_start]**Validation Split:** Implemented a stratified 80/20 train-test partition to ensure uniform class representation ($30,139$ rows for training, $7,535$ rows for test evaluations) [cite: 916-917, 1305].

### 2. Neural Network Configuration
[cite_start]Designed a compact, high-efficiency sequential framework comprising **2,305 trainable parameters**[cite: 936]:
* [cite_start]**Input Layer:** Dense (64 nodes, ReLU activation, Dropout = 30%) [cite: 936, 954-955]
* [cite_start]**Hidden Layer:** Dense (32 nodes, ReLU activation, Dropout = 20%) [cite: 936, 956-957]
* [cite_start]**Output Layer:** Dense (1 node, Sigmoid activation for binary classification) [cite: 936, 958]
* [cite_start]**Optimization:** Adam optimizer monitoring binary cross-entropy loss with early-stopping criteria to guarantee validation convergence [cite: 964, 1442, 1446-1447].

## Key Performance Analytics
[cite_start]The dataset exhibits a stark real-world class imbalance profile ($37,367$ fault-free vs. $307$ faulty rows), elevating the significance of minority class Recall over overall classification accuracy [cite: 1164-1166]:

| Model Architecture | Train Accuracy | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Nearest Neighbors** | 0.9986 | 0.9980 | 0.9980 | 0.9980 | 0.9979 |
| **Decision Tree** | 0.9999 | 0.9968 | 0.9966 | 0.9968 | 0.9967 |
| **Random Forest** | 0.9999 | 0.9975 | 0.9974 | 0.9975 | 0.9973 |
| **Support Vector Machine** | 0.9965 | 0.9963 | 0.9963 | 0.9963 | 0.9957 |
| **Sequential Neural Network** | 0.9970 | 0.9969 | 0.9970 | 0.9969 | 0.9966 |

[cite_start]*(Metrics sourced directly from project validation outputs [cite: 1142]).*

### Engineering Insights
* [cite_start]**The Recall Gap:** While Support Vector Machines maintained an exceptional global accuracy profile, they exhibited the weakest ability to identify actual faults, tracking a minority-class recall of only $0.54$[cite: 1160, 1168]. [cite_start]In contrast, the **Random Forest ensemble** demonstrated optimal robustness, achieving a minority recall of $0.74$ with a precision of $0.94$[cite: 1157, 1170].
* [cite_start]**Feature Importance:** Pearson correlation analysis explicitly isolated Humidity ($0.240$) as a significantly stronger indicator of sensor malfunction than Temperature ($0.086$)[cite: 1172].

## Deployment
Execute the complete classification network via:
```bash
python sensor_analysis.py
