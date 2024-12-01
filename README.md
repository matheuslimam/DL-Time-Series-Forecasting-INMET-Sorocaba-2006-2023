
# DL-Time-Series-Forecasting-INMET-Sorocaba-2006-2023

**Forecasting Extreme Meteorological Events Using Deep Learning: A Case Study on INMET Sorocaba Dataset (2006–2023)**

---

## 📑 Table of Contents
- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [Problem Statement](#-problem-statement)
- [Models Used](#-models-used)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [Authors](#-authors)
- [License](#-license)

---

## 📘 Introduction

This project explores advanced **deep learning techniques** to forecast extreme meteorological events using the **INMET dataset** for Sorocaba (2006–2023). It focuses on predicting intense rainfall and other critical weather phenomena using state-of-the-art models like **LSTM**, **GRU**, and **transformers**. The study contributes to disaster management, climate resilience, and urban planning.

---

## 🌐 Dataset

The dataset includes hourly meteorological data collected by INMET for Sorocaba, with the following key variables:
- **Precipitation** (mm)
- **Atmospheric Pressure** (mB)
- **Temperature** (°C)
- **Relative Humidity** (%)

Comprehensive preprocessing ensured that:
- Temporal alignment was standardized across years.
- Missing data and outliers were handled using robust statistical techniques.
- Time series windows were created to optimize for deep learning architectures.

---

## 🎯 Problem Statement

Accurate forecasting of **rare extreme precipitation events** is critical for disaster mitigation and infrastructure planning. Challenges include:
1. **Data Imbalance**: Extreme events are underrepresented in meteorological datasets.
2. **Non-linear Dependencies**: Relationships between meteorological variables are complex and multi-scale.
3. **Temporal Misalignment**: Sensor anomalies and data inconsistencies affect prediction accuracy.

---

## 🧠 Models Used

The project employs a variety of architectures:
1. **Recurrent Neural Networks (RNNs)**:
   - Long Short-Term Memory (LSTM)
   - Gated Recurrent Unit (GRU)
2. **Attention-Based Transformers**: For capturing global and local dependencies.
3. **Stacking Ensemble**: Combines GRU, LSTM for optimal performance.

---

## 🛠️ Methodology

### 1. Preprocessing
- Temporal alignment and concatenation of yearly datasets.
- Removal of extreme outliers (e.g., erroneous sensor readings).
- Creation of sliding windows to capture sequential patterns.

### 2. Custom Loss Function
A weighted loss function penalized errors more heavily for extreme precipitation events, ensuring model sensitivity to rare events.

### 3. Evaluation Metrics
- **MSE**: Measures overall prediction error.
- **DTW (Dynamic Time Warping)**: Assesses temporal alignment between predicted and observed data.
- **Reanalysis Matrix**: Evaluates hits, misses, and false alarms for extreme event predictions.

---

## 📊 Results

**Model Performance Summary**:

| Model                 | MSE    | RMSE   | MAE    | DTW    |
|-----------------------|--------|--------|--------|--------|
| GRU                  | 0.0003 | 0.0179 | 0.0036 | 85.639 |
| LSTM                 | 0.0003 | 0.0180 | 0.0033 | 82.425 |
| Bidirectional GRU    | 0.0003 | 0.0179 | 0.0032 | 76.726 |
| Bidirectional LSTM   | 0.0003 | 0.0179 | 0.0032 | 70.472 |
| Transformer          | 0.0004 | 0.0203 | 0.0073 | 183.667 |
| **Stacking (Proposed)** | **0.0003** | **0.0199** | **0.0022** | **32.157** |

![Stacking Model Results](Graficos/stackingwithloss2.png

### Key Observations:
- The **stacking model** outperformed standalone architectures by integrating their strengths.
- Transformer-based models excelled in capturing long-term dependencies but were computationally expensive.
- Custom loss functions significantly improved sensitivity to extreme events.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- TensorFlow, Keras, Pandas, NumPy

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DL-Time-Series-Forecasting-INMET-Sorocaba.git
   cd DL-Time-Series-Forecasting-INMET-Sorocaba
   ```
2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Place the dataset in the `data/` folder.

---

## 🚀 Usage

### Train Models
Train LSTM and GRU models using scripts in the `Modelos/` directory:
```bash
python Modelos/train_lstm.py
```

### Evaluate Models
Evaluate pre-trained models stored in `Pesos/`:
```bash
python evaluate_model.py --model_path Pesos/lstm_model.h5
```

### Visualize Results
Generate performance visualizations:
```bash
python Graficos/generate_plots.py
```

---

## 📂 Project Structure

```bash
DL-Time-Series-Forecasting-INMET-Sorocaba-2006-2023/
├── data/                  # Meteorological dataset files
├── Graficos/              # Scripts and output for visualizations
├── Modelos/               # Model training and evaluation scripts
├── Pesos/                 # Pre-trained model weights
├── Tabela/                # Tables summarizing results
├── Topologias/            # Model configurations and architectures
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 🔮 Future Work

### 1. Advanced Architectures
- Explore **hybrid models** combining deep learning with physics-based approaches to improve generalization across diverse meteorological conditions.
- Integrate **graph neural networks (GNNs)** to model spatial dependencies between meteorological stations.

### 2. Real-Time Prediction
- Implement a **real-time weather forecasting system** with adaptive learning to handle streaming data.
- Incorporate **edge AI solutions** for deployment on low-power devices in remote locations.

### 3. Collaboration and Open Science
- Share the trained models and datasets via an **open repository** to encourage reproducibility and collaboration.
- Develop an API for seamless integration with operational meteorological services.

---

## 👨‍💻 Authors

- Enzo Marcondes de Andrade Pereira Esteban
- Felipe Zanardo Goldoni
- Levi De Souza Correia
- Leopoldo André Dutra Lusquino Filho
- Matheus Lima Maturano Martins de Castro
- Natan Da Silva Guedes
- William Dantas Vichete

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for more details.

