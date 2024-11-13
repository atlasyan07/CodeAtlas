# Battery Voltage Prediction Model

## Overview

This project aims to predict the battery voltage of a spacecraft using telemetry data and machine learning techniques. Leveraging LSTM networks and residual blocks, the model captures temporal dependencies and complex relationships between telemetry features. Accurate voltage predictions are critical for spacecraft health monitoring and operational planning.

## Table of Contents
1. [Project Motivation](#project-motivation)
2. [Data Description](#data-description)
3. [Features](#features)
4. [Model Architecture](#model-architecture)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Work](#future-work)
8. [License](#license)

---

## Project Motivation

Spacecraft batteries play a crucial role in mission success, providing power during periods when solar power is unavailable (e.g., in eclipse). Predicting battery voltage accurately can help in:
- Preventing unexpected power failures.
- Optimizing operational planning.
- Monitoring battery health and efficiency.

---

## Data Description

The model uses telemetry data from a spacecraft, including:
- **Battery Voltage** (`spacecraft_eps_pcu_batteryVoltage`)
- **Load Current** (`spacecraft_eps_pcu_loadCurrent`)
- **Solar Array Current** (`spacecraft_eps_pcu_solarArrayCurrent`)
- **Suntrack Condition** (`spacecraft_adcs_afswOut_acMode`)
- **Eclipse Condition** (`spacecraft_adcs_afswOut_eclipseCondition`)
- **Beta Angle** (`spacecraft_adcs_afswOut_betaAngle`)

The data covers a specified time period and is fetched using an Elasticsearch query.

---

## Features

1. **Data Preprocessing and Feature Engineering**:
   - Calculation of `battery_current` as the difference between `loadCurrent` and `solarArrayCurrent`.
   - Creation of `suntrack_condition` as a binary indicator based on the ADCS mode.
   - Calculation of `fractional_orbit` using eclipse exit conditions.
   - Smoothing of key features with the Savitzky-Golay filter to reduce noise.

2. **Model Architecture**:
   - **Residual CNN Blocks**: Capture local dependencies in the data.
   - **LSTM Layers**: Capture temporal patterns and long-term dependencies.
   - **Dense Layers**: Provide additional complexity with residual connections.
   - **Regularization and Normalization**: Techniques like dropout, layer normalization, and weight regularization are used to improve generalization.

3. **Training**:
   - Loss Function: Huber loss to minimize the impact of outliers.
   - Optimizer: Adam optimizer with a decaying learning rate schedule.
   - Early stopping to avoid overfitting and ensure convergence.

---

## Model Architecture

The model consists of:
- **Input Layer**: Handles input features.
- **Residual Convolutional Blocks**: Capture localized patterns with skip connections for better gradient flow.
- **LSTM Layers**: Bidirectional LSTM layers for capturing sequential patterns.
- **Dense Layers with Residual Connections**: Allow for complex nonlinear transformations.
- **Output Layer**: Predicts the battery voltage as a continuous value.

---

## Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- Pandas, NumPy, Matplotlib
- SciPy
- Elasticsearch (for fetching telemetry data)

### Running the Model
1. **Clone the repository**:
   ```bash
   git clone https://github.com/atlasyan07/Battery-Voltage-Prediction.git
   cd Battery-Voltage-Prediction
2. **Install the required dependencies**:  
  `pip install -r requirements.txt`

3. **Run the Jupyter Notebook**:  
  Open the Jupyter Notebook and follow the steps to preprocess data, train the model, and evaluate predictions.

3. **To train the model from a script**:  
  `python train_model.py`

## Results

- **Training and Validation Loss**: The training loss steadily decreased, reaching a minimum value of approximately 0.02 (Huber loss), while the validation loss fluctuated after reaching a minimum of around 0.05, indicating potential overfitting.
- **MSE and MAE**: The training MSE dropped below 0.02, while the validation MSE fluctuated between 0.03 and 0.1. The training MAE stabilized near 0.1, with the validation MAE varying between 0.1 and 0.2.
- **Predicted vs. True Voltage**: The model captured the overall trends in voltage well, with predicted values closely matching true values in most cases. However, discrepancies were noted during sharp transitions in voltage, indicating areas for potential improvement.

