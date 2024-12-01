import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from keras import layers, Model
from keras.callbacks import EarlyStopping
from fastdtw import fastdtw
import math

import tensorflow as tf
print("GPUs detectadas:", tf.config.list_physical_devices('GPU'))

tf.keras.backend.clear_session()


# Verifique as GPUs detectadas
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detectadas: {gpus}")
else:
    print("Nenhuma GPU detectada pelo TensorFlow.")

# Execute uma operação simples para testar o dispositivo
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"Resultado da multiplicação de matrizes: {c}")


# Carregar e preparar o dataset
df = pd.read_csv('/home/matheuslimam/INMET_Sorocaba_2006-2024.csv', parse_dates=['DATAHORA'], index_col='DATAHORA')
colunas_para_converter = [
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)', 'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)', 'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)',
    'UMIDADE RELATIVA DO AR, HORARIA (%)', 'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)', 'VENTO, RAJADA MAXIMA (m/s)'
]
for coluna in colunas_para_converter:
    df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
start_date, end_date = '2009-08-08', '2009-10-30'
df_semana = df.loc[start_date:end_date]

# Normalização e criação de janelas
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df.values)
window_size = 48
target_column_index = df.columns.get_loc('PRECIPITAÇÃO TOTAL, HORÁRIO (mm)')

def create_sliding_windows(data, window_size, target_column_index):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, target_column_index])
    return np.array(X), np.array(y)

X, y = create_sliding_windows(data_normalized, window_size, target_column_index)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modelo TFT
input_layer = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

# Camada de atenção para focar em aspectos temporais específicos
attention = layers.MultiHeadAttention(num_heads=4, key_dim=X_train.shape[2])(input_layer, input_layer)
attention = layers.Add()([input_layer, attention])
attention = layers.LayerNormalization()(attention)

# LSTM com atenção para capturar dependências temporais
lstm_layer = layers.LSTM(64, return_sequences=True)(attention)
attention_lstm = layers.MultiHeadAttention(num_heads=4, key_dim=64)(lstm_layer, lstm_layer)
attention_lstm = layers.Add()([lstm_layer, attention_lstm])
attention_lstm = layers.LayerNormalization()(attention_lstm)

# Camadas Densas Finais
flattened = layers.Flatten()(attention_lstm)
output_layer = layers.Dense(1)(flattened)

model_tft = Model(inputs=input_layer, outputs=output_layer)
model_tft.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Treinamento com early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model_tft.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Previsões e métricas
y_pred_tft = model_tft.predict(X_test)

# Função para calcular métricas de erro
def calculate_metrics(y_true, y_pred):
    # Transformar y_true e y_pred em arrays 1D
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    # Calcular MSE, RMSE, MAE
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # Calcular a distância DTW
    dtw_distance, _ = fastdtw(y_true, y_pred)  # Calcular distância DTW

    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, DTW: {dtw_distance}")

calculate_metrics(y_test, y_pred_tft)

# Visualização dos resultados
n_points_to_show = 1000
plt.figure(figsize=(12, 6))
plt.plot(y_test[:n_points_to_show], label='Verdadeiro', color='blue')
plt.plot(y_pred_tft[:n_points_to_show], label='Predição TFT', color='orange')
plt.title('Previsão de Precipitação com TFT')
plt.xlabel('Tempo')
plt.ylabel('Precipitação (mm)')
plt.legend()
plt.grid(True)
plt.savefig('grafico_predicao_TFT.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
