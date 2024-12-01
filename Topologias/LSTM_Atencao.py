import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, BatchNormalization, LeakyReLU, Dropout, Add, Input
from keras.layers import Layer
from keras.regularizers import l2
from sklearn.metrics import mean_absolute_error
from fastdtw import fastdtw
import math
from keras.callbacks import EarlyStopping
import tensorflow as tf

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

# Definir o intervalo de datas
start_date = '2009-08-08'
end_date = '2009-10-30'

# Filtrar os dados entre essas datas
df_semana = df.loc[start_date:end_date]

# Configurar o tamanho do gráfico
plt.figure(figsize=(15, 8))

# Plotar cada variável
for coluna in colunas_para_converter:
    plt.plot(df_semana.index, df_semana[coluna], label=coluna)

# Títulos e rótulos
plt.title('Variáveis ao Longo do Tempo (' + start_date + ' - ' + end_date + ')')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend(loc='upper right')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Normalização
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df.values)

# Função para criar as janelas deslizantes
def create_sliding_windows(data, window_size, target_column_index):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, target_column_index])  # use o índice da coluna
    return np.array(X), np.array(y)

def custom_loss(y_true, y_pred):
    limiar_chuva = 0.1  
    weights = tf.where(y_true > limiar_chuva, 5.0, 1.0)  # fator de penalização
    loss = tf.reduce_mean(weights * tf.square(y_true - y_pred))
    return loss

# Defina o tamanho da janela deslizante
window_size = 48
target_column = 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'
target_column_index = df.columns.get_loc(target_column)

# Criando as janelas deslizantes
X, y = create_sliding_windows(data_normalized, window_size, target_column_index)

# Dividir em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Formato do conjunto de treinamento:", X_train.shape, y_train.shape)
print("Formato do conjunto de teste:", X_test.shape, y_test.shape)

# Função para calcular métricas de erro
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    dtw_distance, _ = fastdtw(y_true, y_pred)
    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, DTW: {dtw_distance}")

# Camada de Atenção personalizada para usar dentro do modelo sequencial
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        score = tf.matmul(inputs, inputs, transpose_b=True)
        distribution = tf.nn.softmax(score, axis=-1)
        output = tf.matmul(distribution, inputs)
        return output

# Construção do modelo usando Sequential e adicionando a atenção com um truque de compatibilidade
input_shape = (X_train.shape[1], X_train.shape[2])
modelo_sequencial = Sequential()

# Camada LSTM
modelo_sequencial.add(LSTM(256, return_sequences=True, input_shape=input_shape))
modelo_sequencial.add(BatchNormalization())

# Camada de Atenção
modelo_sequencial.add(AttentionLayer())

# Segunda camada LSTM e normalização
modelo_sequencial.add(LSTM(128, return_sequences=True))
modelo_sequencial.add(BatchNormalization())

# Terceira camada LSTM sem sequência
modelo_sequencial.add(LSTM(64))

# Camadas densas e saída
modelo_sequencial.add(Dense(50))
modelo_sequencial.add(Dense(16, kernel_regularizer=l2(0.01)))
modelo_sequencial.add(LeakyReLU(alpha=0.01))
modelo_sequencial.add(Dense(1))

# Compilar o modelo
modelo_sequencial.compile(optimizer='adam', loss=custom_loss)

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Treinamento com validação e early stopping
modelo_sequencial.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stop])

# Salvar os pesos após o treinamento
modelo_sequencial.save_weights('LSTM_sequencial_atenção_pesos.weights.h5')

# Previsão e cálculo de métricas
y_pred_sequencial = modelo_sequencial.predict(X_test)
calculate_metrics(y_test, y_pred_sequencial)

# Definir uma janela menor para visualização
n_points_to_show = 1000

# Visualização dos resultados
plt.figure(figsize=(12, 6))
plt.plot(y_test[:n_points_to_show], label='Verdadeiro', color='blue')
plt.plot(y_pred_sequencial[:n_points_to_show], label='Predição LSTM Sequencial c/ Atenção', color='orange')
plt.title('Previsão de Precipitação com LSTM e Mecanismo de Atenção')
plt.xlabel('Tempo')
plt.ylabel('Precipitação (mm)')
plt.legend()
plt.grid(True)

# Salvar o gráfico automaticamente
plt.savefig('grafico_predicao_LSTM_Atencao.png', format='png', dpi=300, bbox_inches='tight')

# Mostrar o gráfico
plt.show()
