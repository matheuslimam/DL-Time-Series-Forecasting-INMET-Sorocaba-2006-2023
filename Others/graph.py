import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Criando DataFrame para a Tabela de Contingência
data_contingency = {
    'Topologia': [
        'GRU', 'GRU', 'GRU', 'GRU', 'LSTM', 'LSTM', 'LSTM', 'LSTM', 
        'Bidirectional_GRU', 'Bidirectional_GRU', 'Bidirectional_GRU', 'Bidirectional_GRU', 
        'Bidirectional_LSTM', 'Bidirectional_LSTM', 'Bidirectional_LSTM', 'Bidirectional_LSTM'
    ],
    'Tipo de chuva': [
        'leve', 'moderada', 'intensa', 'muito intensa', 
        'leve', 'moderada', 'intensa', 'muito intensa', 
        'leve', 'moderada', 'intensa', 'muito intensa', 
        'leve', 'moderada', 'intensa', 'muito intensa'
    ],
    'Hit': [
        158, 55, 0, 0, 
        71, 12, 0, 0, 
        226, 254, 4, 0, 
        18, 13, 11, 0
    ],
    'Miss': [
        240, 91, 72, 16, 
        341, 91, 74, 17, 
        64, 83, 69, 15, 
        387, 99, 81, 21
    ],
    'Fault': [
        85, 97, 0, 0, 
        92, 41, 0, 0, 
        169, 96, 7, 0, 
        50, 43, 22, 0
    ]
}

df_contingency = pd.DataFrame(data_contingency)

# Criando DataFrame para a Tabela de Parâmetros
data_params = {
    'Model': [
        'GRU', 'LSTM', 'Bidirectional_GRU', 
        'Bidirectional_LSTM', 'LSTM_atencao', 'Tranformer'
    ],
    'MSE': [
        0.0003191118899622368, 0.00032493365454397996, 
        0.0003202892922301957, 0.00032109222568096316, 
        0.0003901427087350962, 0.0004111794907028363
    ],
    'RMSE': [
        0.017863703142468438, 0.018025916191527686, 
        0.017896627956969873, 0.017919046450103397, 
        0.019752030496510888, 0.020277561261227553
    ],
    'MAE': [
        0.0035951634980488376, 0.003284033330255859, 
        0.0031965150677622737, 0.003181782291763421, 
        0.006991154075390483, 0.007345677809803076
    ],
    'DTW': [
        85.6392262724711, 82.42516645660841, 76.72583608592556, 
        70.47227826357395, 209.99673402297233, 183.66691300150006
    ]
}

df_params = pd.DataFrame(data_params)

# Configurando o estilo de visualização
sns.set(style="whitegrid")

# Gráfico de Tabela de Contingência
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

sns.barplot(
    data=df_contingency, x='Tipo de chuva', y='Hit', hue='Topologia', ax=ax[0]
).set(title="Hits por Topologia e Tipo de Chuva")

sns.barplot(
    data=df_contingency, x='Tipo de chuva', y='Miss', hue='Topologia', ax=ax[1]
).set(title="Misses por Topologia e Tipo de Chuva")

plt.tight_layout()

# Gráfico de Tabela de Parâmetros
fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    df_params.set_index('Model').T, annot=True, cmap="YlGnBu", fmt=".6f"
).set(title="Tabela de Parâmetros dos Modelos")

plt.show()

# Exibindo as tabelas formatadas
df_contingency_styled = df_contingency.style.set_caption("Tabela de Contingência").background_gradient(subset=["Hit", "Miss", "Fault"], cmap="YlOrRd")
df_params_styled = df_params.style.set_caption("Tabela de Parâmetros dos Modelos").background_gradient(cmap="YlGnBu")

df_contingency_styled, df_params_styled

