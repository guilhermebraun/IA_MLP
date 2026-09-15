import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

dados = pd.read_excel("ENB2012_data.xlsx")

print(dados.shape)
print(dados.head())

X = dados.iloc[:, 0:8] #Pega as 8 primeiras colunas
Y = dados.iloc[:, 8:10] #Pega as 2 últimas colunas
print("\nEntradas X:")
print(X.head())

print("\nSaídas Y:")
print(Y.head())

#Divisão de treino e depois validação e teste
X_treino, X_temp, y_treino, y_temp = train_test_split(X, Y, test_size=0.20, random_state=42)
X_validacao, X_teste, y_validacao, y_teste = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

#Normalizar entradas
normalizador_X = StandardScaler()
X_treino = normalizador_X.fit_transform(X_treino)
X_validacao = normalizador_X.transform(X_validacao)
X_teste = normalizador_X.transform(X_teste)

#Normalizar saídas
normalizador_y = StandardScaler()
y_treino = normalizador_y.fit_transform(y_treino)
y_validacao = normalizador_y.transform(y_validacao)
y_teste = normalizador_y.transform(y_teste)

#Criar o modelo de rede neural MLP
mlp = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='logistic',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42
)
#Treinar o modelo
mlp.fit(X_treino, y_treino)

#Fazer as previsões
y_pred = mlp.predict(X_teste)
print("\nPrimeiras previsões:")
print(y_pred[:5])

# Voltar as previsões para a escala original (desnormalizar)
y_pred = normalizador_y.inverse_transform(y_pred)
y_real = normalizador_y.inverse_transform(y_teste)

#Calcular os resultados do modelo
mse = mean_squared_error(y_real, y_pred)
r2 = r2_score(y_real, y_pred)

print("\nResultados gerais do teste:")
print("MSE:", mse)
print("R²:", r2)

#Calcular os resultados de cada saída separadamente
mse_y1 = mean_squared_error(y_real[:, 0], y_pred[:, 0])
mse_y2 = mean_squared_error(y_real[:, 1], y_pred[:, 1])

r2_y1 = r2_score(y_real[:, 0], y_pred[:, 0])
r2_y2 = r2_score(y_real[:, 1], y_pred[:, 1])

print("\nResultados por saída:")

print("\nHeating Load (Y1)")
print("MSE:", mse_y1)
print("R²:", r2_y1)

print("\nCooling Load (Y2)")
print("MSE:", mse_y2)
print("R²:", r2_y2)

#Mostrar as informações da rede neural
print("\nArquitetura da rede neural:")
print("Camada oculta:", mlp.hidden_layer_sizes)
print("Número de camadas:", len(mlp.coefs_) + 1)