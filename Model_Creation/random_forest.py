import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score

# Lo primero que hago es cargar los datos que previamente limpié y escalé. Esto es 
# lo mismo que hago con mi otro modelo.
df_scaled = pd.read_csv('automobile_cleaned_scaled.csv')
X = df_scaled.drop('Price', axis=1).values  # Aquí, elimino la columna de precios para que las demás características sean mis variables independientes.
y = df_scaled['Price'].values  # Esta es la columna que quiero predecir, es decir, mi variable dependiente.

# A continuación, divido mis datos en tres conjuntos: entrenamiento, validación y prueba.
# El conjunto de entrenamiento lo utilizaré para ajustar mi modelo,
# el de validación me ayudará a ajustar los hiperparámetros y evitar el overfitting,
# y el de prueba será para evaluar el rendimiento final del modelo.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Ahora creo el modelo de Random Forest, utilizando 100 árboles en el bosque.
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entreno mi modelo utilizando el conjunto de entrenamiento. Aquí es donde el modelo aprende
# las relaciones entre las características de los automóviles y el precio.
model.fit(X_train, y_train)

# Después del entrenamiento, hago predicciones en los conjuntos de entrenamiento, validación y prueba.
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculo el error cuadrático medio (MSE) y el coeficiente de determinación (R^2) para cada conjunto.
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Cross-Validation MSE:", -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean())
print("Error final (entrenamiento):", train_mse)
print("Error en validación:", val_mse)
print("Error final (prueba):", test_mse)
print("R^2 en conjunto de entrenamiento:", r2_train)
print("R^2 en conjunto de validación:", r2_val)
print("R^2 en conjunto de prueba:", r2_test)

# Para entender cómo el tamaño del conjunto de entrenamiento afecta el rendimiento del modelo,
# trazo la curva de aprendizaje. Esto me muestra el MSE en función del tamaño del conjunto de entrenamiento.
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)

train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = -np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, val_scores_mean, label='Validation error')

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.2)

plt.title('Curva de Aprendizaje')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# Aquí, analizo la importancia de cada característica en mi modelo.
# Esto me ayuda a entender qué variables son las más influyentes en la predicción del precio.
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Importancia de las Características')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), df_scaled.drop('Price', axis=1).columns[indices], rotation=90)
plt.xlabel('Características')
plt.ylabel('Importancia')
plt.show()


def grafica_costo(mse_history):
    """
    Grafico el historial de costo durante el entrenamiento.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mse_history)), mse_history)
    plt.title('Función de costo')
    plt.xlabel('Numero de iteraciones')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def grafica_predicted_vs_actual(y_true, y_pred, set_name):
    """
    Graficar los valores predichos vs los valores reales.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores predichos')
    plt.title(f'Predecidos vs Reales - {set_name}')
    plt.grid(True)
    plt.show()

# Finalmente, visualizo cómo las predicciones del modelo se comparan con los valores reales.
# Hago esto para los conjuntos de entrenamiento, validación y prueba.
grafica_predicted_vs_actual(y_train, y_train_pred, 'Training Set')
grafica_predicted_vs_actual(y_val, y_val_pred, 'Validation Set')
grafica_predicted_vs_actual(y_test, y_test_pred, 'Test Set')
