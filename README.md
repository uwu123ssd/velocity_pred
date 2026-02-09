# Whitebook: Using Deep Learning for Velocity Sensor Prediction


## Overview
This document provides a guide on how to utilize deep learning for predicting velocity using sensor data.

## Instructions

### 1. Modify File Paths

Before running the model, you need to update the file paths for the training and prediction data in `main.py`. Modify the following lines:

```python
parser.add_argument('--training_file', type=str, help='Path to the training Excel file',
                    default="/Users/yuanthu/Downloads/simulation.xlsx")
parser.add_argument('--prediction_file', type=str, help='Path to the Excel file for prediction',
                    default="/Users/yuanthu/Downloads/simulation_test.xlsx")
```

### 2. Run the Script
Once the file paths are set, run the script using:
```python
python main.py
```
This will execute the deep learning model for velocity prediction using the specified sensor data. Have Fun~

### 3. Results
It runs smoothly in prediction where error within 3. This can be improved by redesigning the model in the future~
```
Best regression type: deep_learning (R² = 1.0000)
Regression coefficients: (<keras.src.engine.sequential.Sequential object at 0x7fe675b74c40>, MinMaxScaler(), MinMaxScaler())
313/313 [==============================] - 1s 4ms/step
313/313 [==============================] - 1s 4ms/step
313/313 [==============================] - 1s 3ms/step

Prediction Results:

Sheet: 250
Detected Slope of Temp: 0.028956
Predicted velocity: 254.64 m/s

Sheet: 550
Detected Slope of Temp: 0.014200
Predicted velocity: 549.23 m/s

Sheet: 750
Detected Slope of Temp: 0.009983
Predicted velocity: 750.32 m/s
```
![Linear Prediction](README/linear_plt.png)

![Our methods](README/prediction_sim.png)