# IoT Sensor Data Classification and Fault Detection

## Overview
This repository implements a multi-model machine learning pipeline designed to clean, scale, and classify continuous environmental sensor telemetry. Utilizing an aggregated dataset of **37,674 records** capturing temperature and humidity values, the pipeline evaluates five supervised machine learning architectures to accurately differentiate between fault-free operations (Label 0) and hardware/environmental malfunctions (Label 1).

## Technical Stack
* **Languages & Frameworks:** Python, TensorFlow / Keras 
* **Core Libraries:** Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib 
* **Implemented Models:** K-Nearest Neighbors (KNN), Decision Trees, Random Forests, Support Vector Machines (SVM), and a Sequential Deep Neural Network.

## Pipeline Architecture

### 1. Data Preprocessing & Cleaning
* **Feature Scaling:** Applied a standard scaling transform ($\mu = 0, \sigma = 1$) to account for divergent continuous ranges across humidity and temperature measurements.
* **Validation Split:** Implemented a stratified 80/20 train-test partition to ensure uniform class representation ($30,139$ rows for training, $7,535$ rows for test evaluations).

### 2. Neural Network Configuration
Designed a compact, high-efficiency sequential framework comprising **2,305 trainable parameters**:
* **Input Layer:** Dense (64 nodes, ReLU activation, Dropout = 30%) 
* **Hidden Layer:** Dense (32 nodes, ReLU activation, Dropout = 20%)
* **Output Layer:** Dense (1 node, Sigmoid activation for binary classification) 
* **Optimization:** Adam optimizer monitoring binary cross-entropy loss with early-stopping criteria to guarantee validation convergence.

## Key Performance Analytics
The dataset exhibits a stark real-world class imbalance profile ($37,367$ fault-free vs. $307$ faulty rows), elevating the significance of minority class Recall over overall classification accuracy :

| Model Architecture | Train Accuracy | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Nearest Neighbors** | 0.9986 | 0.9980 | 0.9980 | 0.9980 | 0.9979 |
| **Decision Tree** | 0.9999 | 0.9968 | 0.9966 | 0.9968 | 0.9967 |
| **Random Forest** | 0.9999 | 0.9975 | 0.9974 | 0.9975 | 0.9973 |
| **Support Vector Machine** | 0.9965 | 0.9963 | 0.9963 | 0.9963 | 0.9957 |
| **Sequential Neural Network** | 0.9970 | 0.9969 | 0.9970 | 0.9969 | 0.9966 |

*(Metrics sourced directly from project validation outputs).*

### Engineering Insights
* **The Recall Gap:** While Support Vector Machines maintained an exceptional global accuracy profile, they exhibited the weakest ability to identify actual faults, tracking a minority-class recall of only $0.54$. In contrast, the **Random Forest ensemble** demonstrated optimal robustness, achieving a minority recall of $0.74$ with a precision of $0.94$.
* **Feature Importance:** Pearson correlation analysis explicitly isolated Humidity ($0.240$) as a significantly stronger indicator of sensor malfunction than Temperature ($0.086$).

## Deployment
Execute the complete classification network via:
```bash
python sensor_analysis.py
