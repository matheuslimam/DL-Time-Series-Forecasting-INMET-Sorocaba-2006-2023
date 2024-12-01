import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, LSTM, Bidirectional, Dense, BatchNormalization, LeakyReLU
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fastdtw import fastdtw

# Configurações dos modelos e do dataset
modelos_pesos = {
    'GRU': 'Trabalho_INMET/Pesos/GRU_pesos.weights.h5',
    'LSTM': 'Trabalho_INMET/Pesos/lstm_pesos.weights.h5',
    'Bidirectional_GRU': 'Trabalho_INMET/Pesos/Bidirectional_GRU_pesos.weights.h5',
    'Bidirectional_LSTM': 'Trabalho_INMET/Pesos/Bidirectional_LSTM_pesos.weights.h5'
}
df = pd.read_csv('/home/matheuslimam/INMET_Sorocaba_2006-2024.csv', parse_dates=['DATAHORA'], index_col='DATAHORA')
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df.values)

# Funções auxiliares
def create_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])  # Precipitação na primeira coluna
    return np.array(X), np.array(y)

def get_model(model_name, input_shape):
    model = Sequential()
    if model_name == 'GRU':
        model.add(GRU(256, return_sequences=True, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(GRU(128, return_sequences=True))
        model.add(BatchNormalization())
        model.add(GRU(64))
        model.add(Dense(50))
        model.add(Dense(16, kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1))
    elif model_name == 'LSTM':
        model.add(LSTM(256, return_sequences=True, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LSTM(128, return_sequences=True))
        model.add(BatchNormalization())
        model.add(LSTM(64))
        model.add(Dense(50))
        model.add(Dense(16, kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1))
    elif model_name == 'Bidirectional_GRU':
        model.add(Bidirectional(GRU(256, return_sequences=True, input_shape=input_shape)))
        model.add(BatchNormalization())
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Bidirectional(GRU(64)))
        model.add(Dense(50))
        model.add(Dense(16, kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1))
    elif model_name == 'Bidirectional_LSTM':
        model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=input_shape)))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(50))
        model.add(Dense(16, kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1))
    return model

def calcular_metricas(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    dtw_distance, _ = fastdtw(y_true, y_pred)
    return mse, rmse, mae, dtw_distance

# Preparação dos dados
window_size = 48
X, y = create_sliding_windows(data_normalized, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Embedding e previsões dos modelos base
previsoes_modelos = []
for model_name, weights_path in modelos_pesos.items():
    model = get_model(model_name, (X_train.shape[1], X_train.shape[2]))
    model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
    model.load_weights(weights_path)
    
    y_pred = model.predict(X_test).flatten()
    previsoes_modelos.append(y_pred)

embedding_array = np.column_stack(previsoes_modelos)

# Modelo de Stacking com Rede Neural
modelo_comb = Sequential([
    Dense(64, input_dim=embedding_array.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
modelo_comb.compile(optimizer='adam', loss='mean_squared_error')
modelo_comb.fit(embedding_array, y_test, epochs=100, batch_size=64, validation_split=0.2)

# Avaliação do modelo de Stacking
y_pred_nn = modelo_comb.predict(embedding_array).flatten()
mse_nn, rmse_nn, mae_nn, dtw_distance_nn = calcular_metricas(y_test, y_pred_nn)

print(f"Métricas do modelo combinado - MSE: {mse_nn}, RMSE: {rmse_nn}, MAE: {mae_nn}, DTW: {dtw_distance_nn}")

# Visualização das previsões
plt.figure(figsize=(12, 6))
plt.plot(y_test[:200], label='Valor Real', color='blue', linestyle='-', linewidth=2)
plt.plot(y_pred_nn[:200], label='Predição Combinada (Stacking)', color='red', linestyle='-')
plt.title('Comparação entre Real e Predição (Stacking)')
plt.xlabel('Tempo')
plt.ylabel('Precipitação (mm)')
plt.legend()
plt.grid(True)
plt.show()
