# DL-Time-Series-Forecasting-INMET-Sorocaba-2006-2023

**Deep Learning Time Series Forecasting with Mixed Frequencies: Case Study Using INMET Dataset for Sorocaba (2006-2023)**

---

## 📑 Table of Contents
- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [Problem Statement](#-problem-statement)
- [Models Used](#models-used)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## 📘 Introduction

This project leverages **deep learning** models to forecast meteorological time series data with **mixed frequencies** from the **INMET Sorocaba** dataset spanning from **2006 to 2023**. The primary goal is to predict intense rainfall events and other key meteorological variables using advanced recurrent neural networks, specifically **LSTM** and **GRU** models.

---

## 🌐 Dataset

The dataset used in this project was obtained from the **INMET (Instituto Nacional de Meteorologia)** and contains hourly meteorological data from **Sorocaba**, a city in Brazil. The dataset includes the following key variables:
- **Precipitation** (mm)
- **Atmospheric Pressure** (mB)
- **Maximum and Minimum Temperature** (°C)
- **Relative Humidity** (%)

Data from **2006 to 2023** was pre-processed to handle missing values and normalize the variables for deep learning model training.

---

## 🎯 Problem Statement

The objective is to accurately **forecast meteorological events**, particularly **intense rainfall**, which is crucial for disaster management, urban planning, and public safety in regions prone to extreme weather events, such as Sorocaba.

Key challenges include:
- Handling **mixed frequency** data.
- Dealing with temporal lags between variables like humidity, pressure, and rainfall.
- Developing models that generalize well on unseen data.

---

## 🧠 Models Used

Two **Recurrent Neural Network (RNN)** architectures were employed for time series forecasting:
- **Long Short-Term Memory (LSTM)**
- **Gated Recurrent Unit (GRU)**

These models were chosen due to their ability to capture long-term dependencies in time series data, which is essential for predicting weather patterns.

---

## 🛠️ Methodology

1. **Data Preprocessing**:
   - **Normalization**: The data was normalized between 0 and 1 to enhance model convergence.
   - **Handling Missing Data**: Rows with missing values were dropped to ensure model accuracy.
   - **Sequence Generation**: The dataset was split into sequences of 24-hour intervals to predict the next hour.

2. **Model Training**:
   - The dataset was split into **80% training** and **20% testing** sets.
   - Both LSTM and GRU models were trained using **50 epochs**, with a batch size of **32**, and the **Adam optimizer**.

3. **Evaluation Metrics**:
   - **Mean Squared Error (MSE)**
   - **Root Mean Squared Error (RMSE)**
   - **Mean Absolute Error (MAE)**
   - **Coefficient of Determination (R²)**

---

## 📊 Results

The models were evaluated using the test set. The performance of the **LSTM**, **GRU**, and **SARIMA** models are summarized below:

| Model   | MSE      | RMSE    | MAE     | R²      |
|---------|----------|---------|---------|---------|
| LSTM    | 0.0075   | 0.0866  | 0.0300  | 0.0278  |
| GRU     | 0.0075   | 0.0866  | 0.0343  | 0.0267  |
| SARIMA  | 0.0109   | 0.1044  | 0.0490  | -0.4136 |

**Key Findings**:
- Both **LSTM** and **GRU** outperformed the traditional **SARIMA** model, especially in terms of RMSE and MSE.
- **LSTM** showed a slight edge in accuracy, while **GRU** was more computationally efficient.

---

## 🛠️ Installation

### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **Jupyter Notebook** (optional but recommended)
- **TensorFlow** (for deep learning models)
- **Keras**
- **Pandas**
- **NumPy**
- **Matplotlib** (for visualization)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DL-Time-Series-Forecasting-INMET-Sorocaba-2006-2023.git
   cd DL-Time-Series-Forecasting-INMET-Sorocaba-2006-2023
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook

## 🚀 Usage

### Training the Models
### Visualizing Results
### Running Predictions

## 📂 Project Structure

 ```bash
DL-Time-Series-Forecasting-INMET-Sorocaba-2006-2023/
│
├── data/                   # Directory containing the dataset
├── notebooks/               # Jupyter notebooks for training, evaluation, and prediction
├── models/                  # Saved models (LSTM, GRU)
├── src/                     # Source code for model architecture and data processing
├── results/                 # Results and evaluation metrics
├── README.md                # Project documentation
├── requirements.txt         # List of dependencies
└── LICENSE                  # License for the project
```
##🔮 Future Work

1. Explore more advanced architectures: Experiment with hybrid models combining LSTM, GRU, and CNNs for better accuracy.
2. Ensemble Methods: Test ensemble learning techniques to improve model robustness.
3. Feature Engineering: Integrate more meteorological variables to enhance prediction accuracy.
4. Real-time Prediction: Implement a real-time weather forecasting system for more practical applications.

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for more details.


