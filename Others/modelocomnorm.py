import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, BatchNormalization, LeakyReLU, Dropout # type: ignore
from keras.regularizers import l2 # type: ignore
from sklearn.metrics import mean_absolute_error
from fastdtw import fastdtw
import math
from keras.callbacks import EarlyStopping # type: ignore

# Carregar o dataset
df = pd.read_csv('/home/matheuslimam/INMET_Sorocaba_2006-2024.csv', parse_dates=['DATAHORA'], index_col='DATAHORA')

# Colunas a serem convertidas para numérico
colunas_para_converter = [
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)',
    'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)',
    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)',
    'UMIDADE RELATIVA DO AR, HORARIA (%)',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)',
    'VENTO, RAJADA MAXIMA (m/s)'
]

# Converter as colunas para numérico, ignorando erros
for coluna in colunas_para_converter:
    df[coluna] = pd.to_numeric(df[coluna], errors='coerce')

# Aplicar transformação logarítmica para suavizar picos de precipitação
df['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'] = np.log1p(df['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'])

# Definir o intervalo de datas
start_date = '2009-08-08'
end_date = '2009-10-30'
df_semana = df.loc[start_date:end_date]

# Plot das variáveis após a transformação
plt.figure(figsize=(15, 8))
for coluna in colunas_para_converter:
    plt.plot(df_semana.index, df_semana[coluna], label=coluna)
plt.title('Variáveis ao Longo do Tempo (' + start_date + ' - ' + end_date + ')')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend(loc='upper right')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Normalização dos dados
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df[colunas_para_converter].values)

# Função para criar janelas deslizantes
def create_sliding_windows(data, window_size, target_column_index):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, target_column_index])
    return np.array(X), np.array(y)

# Tamanho da janela deslizante ajustado
window_size = 48  # Tamanho de janela maior para capturar mais padrões temporais
target_column = 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'
target_column_index = df.columns.get_loc(target_column)
X, y = create_sliding_windows(data_normalized, window_size, target_column_index)

# Divisão em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Função para calcular métricas de erro
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    dtw_distance, _ = fastdtw(y_true, y_pred)
    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, DTW: {dtw_distance}")

# Modelo LSTM ajustado
modelo_lstm = Sequential()
modelo_lstm.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
modelo_lstm.add(BatchNormalization())
modelo_lstm.add(Dropout(0.3))
modelo_lstm.add(LSTM(64, return_sequences=True))
modelo_lstm.add(BatchNormalization())
modelo_lstm.add(Dropout(0.3))
modelo_lstm.add(LSTM(32))
modelo_lstm.add(Dense(16, kernel_regularizer=l2(0.01)))
modelo_lstm.add(LeakyReLU(alpha=0.01))
modelo_lstm.add(Dense(1))
modelo_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
modelo_lstm.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stop])

# Salvar os pesos após o treinamento
modelo_lstm.save_weights('modelo_lstm_pesos.weights.h5')

# Previsão e cálculo de métricas
y_pred_lstm = modelo_lstm.predict(X_test)

# Reverter a normalização para todas as colunas
X_test_original = scaler.inverse_transform(X_test[:, -1, :])  # Inverter todas as colunas do X_test
y_test_original = X_test_original[:, target_column_index]  # Selecionar a coluna de interesse

# Fazer a previsão
y_pred_lstm_original = np.expm1(y_pred_lstm.flatten())

# Calcular métricas
calculate_metrics(y_test_original, y_pred_lstm_original)

