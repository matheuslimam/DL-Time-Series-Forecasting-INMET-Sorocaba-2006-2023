import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, LSTM, Bidirectional, Dense, BatchNormalization, LeakyReLU
from keras.regularizers import l2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from fastdtw import fastdtw
from keras import backend as K

# Caminho dos pesos dos modelos
modelos_pesos = {
    'GRU': 'Trabalho_INMET/Pesos/GRU_pesos.weights.h5',
    'LSTM': 'Trabalho_INMET/Pesos/lstm_pesos.weights.h5',
    'Bidirectional_GRU': 'Trabalho_INMET/Pesos/Bidirectional_GRU_pesos.weights.h5',
    'Bidirectional_LSTM': 'Trabalho_INMET/Pesos/Bidirectional_LSTM_pesos.weights.h5'
}

# Carregar o dataset
df = pd.read_csv('/home/matheuslimam/INMET_Sorocaba_2006-2024.csv', parse_dates=['DATAHORA'], index_col='DATAHORA')

# Normalização
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df.values)

# Função para criar as janelas deslizantes
def create_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])  # Usando a primeira coluna para a precipitação
    return np.array(X), np.array(y)

# Definindo o tamanho da janela deslizante e dividindo os dados
window_size = 48
X, y = create_sliding_windows(data_normalized, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Função para carregar pesos e retornar o modelo desejado
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

# Defina os limites de precipitação para cada nível de intensidade
chuva_limites = {
    "leve": (0.03, 0.08),     
    "moderada": (0.08, 0.2),   
    "intensa": (0.2, 0.3),    
    "muito_intensa":  (0.3, 0.4), 
    "extrema": (0.4, float("inf"))  
}

# Função para categorizar intensidade de chuva
def categorizar_chuva(valor):
    for categoria, (min_val, max_val) in chuva_limites.items():
        if min_val <= valor < max_val:
            return categoria
    return "sem_chuva"

# Função para calcular RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Função para calcular a tabela de métricas e de contingência com categorias
def calcular_metricas_e_contingencia(y_test, y_pred, model_name):
    # Calcule MSE, RMSE, MAE
    mse = mean_squared_error(y_test, y_pred)
    rmse_val = rmse(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calcule DTW
    dtw_distance, _ = fastdtw(y_test, y_pred)
    
    # Inicialize uma tabela de contingência para cada categoria de chuva
    categorias = ["leve", "moderada", "intensa", "muito_intensa","extrema"]
    contingencia = {categoria: {"Hit": 0, "Miss": 0, "Fault": 0} for categoria in categorias}
    contingencia["sem_chuva"] = {"Hit": 0, "Miss": 0, "Fault": 0}  # Inclui a categoria "sem_chuva" para controle
    
    # Preencha a tabela de contingência
    for i in range(len(y_test)):
        categoria_real = categorizar_chuva(y_test[i])
        categoria_pred = categorizar_chuva(y_pred[i])

        # Adicionar para depuração
        print(f"Real: {categoria_real}, Predito: {categoria_pred}")
        
        if categoria_real == categoria_pred and categoria_real != "sem_chuva":
            contingencia[categoria_real]["Hit"] += 1
        elif categoria_real != "sem_chuva" and categoria_pred == "sem_chuva":
            contingencia[categoria_real]["Miss"] += 1
        elif categoria_real == "sem_chuva" and categoria_pred != "sem_chuva":
            contingencia[categoria_pred]["Fault"] += 1  # Falha para previsão errada de chuva
        elif categoria_real == "sem_chuva" and categoria_pred == "sem_chuva":
            contingencia["sem_chuva"]["Hit"] += 1  # Caso onde ambos são "sem chuva"
        elif categoria_real != categoria_pred:  # Qualquer outra discrepância
            contingencia[categoria_pred]["Fault"] += 1  # Falha para previsão errada
        else:
            print(f"Caso inesperado - Real: {categoria_real}, Predito: {categoria_pred}")

    # Converta a tabela de contingência para um DataFrame
    contingencia_df = pd.DataFrame(contingencia).T
    contingencia_df.index.name = "Categoria"

    # Retorne as métricas e a tabela de contingência
    return pd.DataFrame({
        "Model": [model_name],
        "MSE": [mse],
        "RMSE": [rmse_val],
        "MAE": [mae],
        "DTW": [dtw_distance]
    }), contingencia_df
# Loop para carregar pesos, gerar predições e calcular métricas e contingências
metricas = []
contingencias = []

for model_name, weights_path in modelos_pesos.items():
    # Instancia o modelo e carrega os pesos
    model = get_model(model_name, (X_train.shape[1], X_train.shape[2]))
    model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
    model.load_weights(weights_path)
    
    # Previsão
    y_pred = model.predict(X_test).flatten()  # Garantir que y_pred seja uma lista unidimensional
    
    # Calcule métricas e contingência
    tabela_metricas, tabela_contingencia = calcular_metricas_e_contingencia(y_test, y_pred, model_name)
    metricas.append(tabela_metricas)
    contingencias.append(tabela_contingencia)

# Combine as tabelas em uma única para visualização
tabela_metricas_final = pd.concat(metricas).reset_index(drop=True)
tabela_contingencia_final = pd.concat(contingencias).reset_index(drop=True)

# Exiba ou salve as tabelas
print("Tabela de Métricas:\n", tabela_metricas_final)
print("\nTabela de Contingência:\n", tabela_contingencia_final)

# Opcional: Salvar as tabelas como CSV
tabela_metricas_final.to_csv('tabela_metricas.csv', index=False)
tabela_contingencia_final.to_csv('tabela_contingencia.csv', index=False)

# Plotar os dados e previsões
n_points_to_show = 20000  # Quantidade de pontos a mostrar
plt.figure(figsize=(12, 6))

# Plotar os dados verdadeiros uma vez
plt.plot(y_test[:n_points_to_show], label='Verdadeiro', color='blue')

# Para cada modelo, fazer a previsão e plotar no mesmo gráfico
for model_name, weights_path in modelos_pesos.items():
    # Instancia o modelo e carrega os pesos
    model = get_model(model_name, (X_train.shape[1], X_train.shape[2]))
    model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
    model.load_weights(weights_path)
    
    # Previsão
    y_pred = model.predict(X_test).flatten()  # Garantir que y_pred seja uma lista unidimensional

    # Plotar a previsão do modelo no mesmo gráfico
    plt.plot(y_pred[:n_points_to_show], label=f'Predição {model_name}')

# Configurações do gráfico
plt.title('Comparação de Previsões de Precipitação')
plt.xlabel('Tempo')
plt.ylabel('Precipitação (mm)')
plt.legend()
plt.grid(True)

# Salvar o gráfico comparativo
plt.savefig('grafico_comparativo_predicao.png', format='png', dpi=300, bbox_inches='tight')
plt.close()  # Fechar a figura após salvar para liberar memória

print("Benchmark concluído. Gráfico comparativo e tabelas salvos.")

# Concatenando as previsões de todos os modelos para criar um vetor de embedding
embeddings = []

for model_name, weights_path in modelos_pesos.items():
    # Instancia o modelo e carrega os pesos
    model = get_model(model_name, (X_train.shape[1], X_train.shape[2]))
    model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
    model.load_weights(weights_path)
    
    # Previsão
    y_pred = model.predict(X_test).flatten()  # Garantir que y_pred seja uma lista unidimensional
    
    # Adicionar a previsão ao vetor de embeddings
    embeddings.append(y_pred)

# Convertendo a lista de previsões em um array numpy
embedding_array = np.column_stack(embeddings)  # Cada coluna é a predição de um modelo

# Agora, podemos usar a média das predições de todos os modelos para gerar um único vetor
embedding_avg = embedding_array.mean(axis=1)

# Verificando se o tamanho é compatível
print(f"Tamanho de y_test: {y_test.shape}")
print(f"Tamanho do embedding_avg: {embedding_avg.shape}")

# Calcule as métricas para o embedding
mse = mean_squared_error(y_test, embedding_avg)  # Usando a média das predições
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, embedding_avg)
dtw_distance, _ = fastdtw(y_test, embedding_avg)

# Exibindo as métricas
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"DTW: {dtw_distance}")

# Plotando os valores reais de precipitação (y_test)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:n_points_to_show], label='Valor Real', color='blue', linestyle='-', linewidth=2)

# Plotando as previsões geradas pelo embedding (embedding_avg)
plt.plot(embedding_avg[:n_points_to_show], label='Predição (Embedding)', color='red', linestyle='-', linewidth=2)

# Adicionando título e rótulos aos eixos
plt.title('Comparação entre Real e Predição (Embedding)', fontsize=14)
plt.xlabel('Tempo', fontsize=12)
plt.ylabel('Precipitação (mm)', fontsize=12)

# Adicionando a legenda
plt.legend()

# Mostrando o gráfico
plt.grid(True)
plt.tight_layout()
plt.show()

# Limites de precipitação para cada nível de intensidade
chuva_limites = {
    "leve": (0.03, 0.08),     
    "moderada": (0.08, 0.2),   
    "intensa": (0.2, 0.3),    
    "muito_intensa":  (0.3, 0.4), 
    "extrema": (0.4, float("inf"))  
}

def categorizar_chuva(valor):
    for categoria, (min_val, max_val) in chuva_limites.items():
        if min_val <= valor < max_val:
            return categoria
    return "sem_chuva"  # Retorna "sem_chuva" se não se enquadrar em nenhuma categoria


# Inicialize a tabela de contingência para cada categoria de chuva
categorias = ["leve", "moderada", "intensa", "muito_intensa", "extrema"]
contingencia = {categoria: {"Hit": 0, "Miss": 0, "Fault": 0} for categoria in categorias}
contingencia["sem_chuva"] = {"Hit": 0, "Miss": 0, "Fault": 0}  # Inclui a categoria "sem_chuva" para controle

# Preencha a tabela de contingência
for i in range(len(y_test)):
    categoria_real = categorizar_chuva(y_test[i])
    categoria_pred = categorizar_chuva(embedding_avg[i])
    
    # Contagem na tabela de contingência com lógica revisada
    if categoria_real == categoria_pred and categoria_real != "sem_chuva":
        contingencia[categoria_real]["Hit"] += 1
    elif categoria_real != "sem_chuva" and categoria_pred == "sem_chuva":
        contingencia[categoria_real]["Miss"] += 1
    elif categoria_real == "sem_chuva" and categoria_pred != "sem_chuva":
        contingencia[categoria_pred]["Fault"] += 1
    elif categoria_real == "sem_chuva" and categoria_pred == "sem_chuva":
        contingencia["sem_chuva"]["Hit"] += 1
    elif categoria_real != categoria_pred:
        contingencia[categoria_pred]["Fault"] += 1

# Contagem de dias "sem chuva"
dias_sem_chuva_real = contingencia["sem_chuva"]["Hit"] + contingencia["sem_chuva"]["Miss"]
dias_sem_chuva_pred = contingencia["sem_chuva"]["Hit"] + sum(contingencia[categoria]["Fault"] for categoria in categorias)

print(f"Total de dias sem chuva (Real): {dias_sem_chuva_real}")
print(f"Total de dias sem chuva (Predito): {dias_sem_chuva_pred}")

# Adicione porcentagem de acertos para `Hit`, `Miss`, e `Fault`
for categoria in categorias:
    total = sum(contingencia[categoria].values())
    for key in ["Hit", "Miss", "Fault"]:
        contingencia[categoria][f"{key}_percentual"] = (contingencia[categoria][key] / total) * 100 if total > 0 else 0

# Converta a tabela de contingência atualizada para DataFrame
contingencia_df = pd.DataFrame(contingencia).T
contingencia_df.index.name = "Categoria"

print("Tabela de Contingência para Hits, Misses e Faults:")
print(contingencia_df)


#-------------------------

previsoes_modelos = []
for model_name, weights_path in modelos_pesos.items():
    model = get_model(model_name, (X_train.shape[1], X_train.shape[2]))
    model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
    model.load_weights(weights_path)
    
    # Previsão para o conjunto de teste
    y_pred = model.predict(X_test).flatten()  # Garantir que y_pred seja unidimensional
    previsoes_modelos.append(y_pred)

# Transformar a lista de previsões em um array numpy para treinamento da rede neural
embedding_array = np.column_stack(previsoes_modelos)

# Agora, 'embedding_array' tem o formato (n_samples, n_models), onde cada coluna contém as previsões de um modelo

# Definir o conjunto de treinamento para a rede neural
X_train_nn = embedding_array  # Previsões dos modelos base
y_train_nn = y_test  # Valores reais

def rbf_kernel_loss(y_true, y_pred, gamma):
    # Calcula a diferença quadrada entre previsões e valores reais
    diff = tf.square(y_true - y_pred)
    # Calcula a distância de kernel RBF
    rbf_distance = tf.exp(-gamma * diff)
    # Para transformar em função de custo, queremos minimizar 1 - similaridade
    return tf.reduce_mean(1 - rbf_distance)

def weighted_loss(y_true, y_pred, threshold=0.38, alpha=5.0):
    """
    Função de perda ponderada para penalizar mais fortemente os erros em chuvas intensas.
    
    Parâmetros:
    - y_true: Valores reais (verdadeiros) de precipitação.
    - y_pred: Valores previstos de precipitação.
    - threshold: Limite para considerar como chuva intensa (em mm).
    - alpha: Fator de ponderação para chuvas intensas.
    
    Retorna:
    - A perda calculada.
    """
    
    # Calcular a diferença entre valores reais e previstos
    diff = tf.abs(y_true - y_pred)
    
    # Criar uma máscara para identificar quando a chuva é intensa (acima do limiar)
    intense_mask = tf.cast(tf.greater(y_true, threshold), dtype=K.floatx())
    
    # A perda será maior para eventos de chuva intensa
    loss = diff * (1 + alpha * intense_mask)  # Aumenta o peso quando a chuva é intensa
    
    return tf.reduce_mean(loss)  # Média da perda

# Criação do modelo de rede neural simples para combinar as previsões
modelo_comb = Sequential()
modelo_comb.add(Dense(64, input_dim=X_train_nn.shape[1], activation='relu'))
modelo_comb.add(Dense(32, activation='relu'))
modelo_comb.add(Dense(1))

# Compilação do modelo usando a função de custo RBF personalizada
modelo_comb.compile(optimizer='adam', loss=weighted_loss )

# Treinamento do modelo
modelo_comb.fit(X_train_nn, y_train_nn, epochs=100, batch_size=64, validation_split=0.2)

# Previsões e avaliação do modelo
X_test_nn = embedding_array
y_pred_nn = modelo_comb.predict(X_test_nn).flatten()

# Calcular a performance do modelo combinado
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
dtw_distance_nn, _ = fastdtw(y_test, y_pred_nn)

print(f"MSE: {mse_nn}")
print(f"RMSE: {rmse_nn}")
print(f"MAE: {mae_nn}")
print(f"DTW: {dtw_distance_nn}")

# Plotagem dos resultados
plt.figure(figsize=(12, 6))
plt.plot(y_test[:n_points_to_show], label='Verdadeiro', color='blue')
plt.plot(y_pred_nn[:n_points_to_show], label='Predição Combinada (Rede Neural)', color='red', linestyle='-')
plt.title('Comparação de Previsões Reais e Combinadas')
plt.xlabel('Tempo')
plt.ylabel('Precipitação (mm)')
plt.legend()
plt.grid(True)
plt.show()


# Limites de precipitação para cada nível de intensidade
chuva_limites = {
    "leve": (0.03, 0.08),     
    "moderada": (0.08, 0.2),   
    "intensa": (0.2, 0.3),    
    "muito_intensa":  (0.3, 0.4), 
    "extrema": (0.4, float("inf"))  
}

def categorizar_chuva(valor):
    """Retorna a categoria da precipitação com base no valor"""
    for categoria, (min_val, max_val) in chuva_limites.items():
        if min_val <= valor < max_val:
            return categoria
    return "sem_chuva"  # Retorna "sem_chuva" se não se enquadrar em nenhuma categoria

# Inicialize a tabela de contingência para cada categoria de chuva
categorias = ["leve", "moderada", "intensa", "muito_intensa", "extrema"]
contingencia = {categoria: {"Hit": 0, "Miss": 0, "Fault": 0} for categoria in categorias}
contingencia["sem_chuva"] = {"Hit": 0, "Miss": 0, "Fault": 0}  # Inclui a categoria "sem_chuva" para controle

# Preencha a tabela de contingência
for i in range(len(y_test)):
    categoria_real = categorizar_chuva(y_test[i])
    categoria_pred = categorizar_chuva(y_pred_nn[i])
    
    # Contagem na tabela de contingência
    if categoria_real == categoria_pred and categoria_real != "sem_chuva":
        contingencia[categoria_real]["Hit"] += 1
    elif categoria_real != "sem_chuva" and categoria_pred == "sem_chuva":
        contingencia[categoria_real]["Miss"] += 1
    elif categoria_real == "sem_chuva" and categoria_pred != "sem_chuva":
        contingencia[categoria_pred]["Fault"] += 1
    elif categoria_real == "sem_chuva" and categoria_pred == "sem_chuva":
        contingencia["sem_chuva"]["Hit"] += 1
    elif categoria_real != categoria_pred:
        contingencia[categoria_pred]["Fault"] += 1

# Contagem de dias "sem chuva"
dias_sem_chuva_real = contingencia["sem_chuva"]["Hit"] + contingencia["sem_chuva"]["Miss"]
dias_sem_chuva_pred = contingencia["sem_chuva"]["Hit"] + sum(contingencia[categoria]["Fault"] for categoria in categorias)

print(f"Total de dias sem chuva (Real): {dias_sem_chuva_real}")
print(f"Total de dias sem chuva (Predito): {dias_sem_chuva_pred}")

# Adicione porcentagem de acertos para `Hit`, `Miss`, e `Fault` para cada categoria
for categoria in categorias + ["sem_chuva"]:
    total = sum(contingencia[categoria].values())
    if total > 0:
        for key in ["Hit", "Miss", "Fault"]:
            contingencia[categoria][f"{key}_percentual"] = (contingencia[categoria][key] / total) * 100
    else:
        # Caso não haja ocorrências, setar percentuais como 0
        for key in ["Hit", "Miss", "Fault"]:
            contingencia[categoria][f"{key}_percentual"] = 0

# Converta a tabela de contingência atualizada para DataFrame
contingencia_df = pd.DataFrame(contingencia).T
contingencia_df.index.name = "Categoria"

# Exibir a tabela de contingência
print("Tabela de Contingência para Hits, Misses e Faults:")
print(contingencia_df)



