import pandas as pd

# Dados fornecidos para cada topologia e tipo de chuva
data = {
    'Topologia': ['GRU', 'GRU', 'GRU', 'GRU', 'LSTM', 'LSTM', 'LSTM', 'LSTM', 
                  'Bidirectional_GRU', 'Bidirectional_GRU', 'Bidirectional_GRU', 'Bidirectional_GRU',
                  'Bidirectional_LSTM', 'Bidirectional_LSTM', 'Bidirectional_LSTM', 'Bidirectional_LSTM'],
    'Tipo de Chuva': ['leve', 'moderada', 'intensa', 'muito intensa', 
                      'leve', 'moderada', 'intensa', 'muito intensa',
                      'leve', 'moderada', 'intensa', 'muito intensa',
                      'leve', 'moderada', 'intensa', 'muito intensa'],
    'Hit': [158, 55, 0, 0, 71, 12, 0, 0, 226, 254, 4, 0, 18, 13, 11, 0],
    'Miss': [240, 91, 72, 16, 341, 91, 74, 17, 64, 83, 69, 15, 387, 99, 81, 21],
    'Fault': [85, 97, 0, 0, 92, 41, 0, 0, 169, 96, 7, 0, 50, 43, 22, 0]
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Mostrando a tabela de contingência
df_contingencia = df.pivot_table(index=["Topologia", "Tipo de Chuva"], values=["Hit", "Miss", "Fault"], aggfunc="sum")
df_contingencia = df_contingencia.reset_index()  # Organizando para exibição

# Exibindo a tabela
print("Tabela de Contingência - Reanálise")
print(df_contingencia)
