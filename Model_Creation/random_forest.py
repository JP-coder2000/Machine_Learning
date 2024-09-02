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

# Lista de hiperparámetros a comparar: max_depth, max_leaf_nodes, min_samples_leaf
parametros = [
    {'max_depth': 10, 'max_leaf_nodes': 10, 'min_samples_leaf': 5},
    {'max_depth': 20, 'max_leaf_nodes': 20, 'min_samples_leaf': 5},
    {'max_depth': 5, 'max_leaf_nodes': 5, 'min_samples_leaf': 5},
]

# Tabla para guardar los resultados
resultados = []

# Recorro cada conjunto de hiperparámetros
for params in parametros:
    # Ahora creo el modelo de Random Forest con los hiperparámetros específicos.
    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        max_depth=params['max_depth'],
        max_leaf_nodes=params['max_leaf_nodes'],
        min_samples_leaf=params['min_samples_leaf']
    )

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

    # Cross-validation MSE
    cross_val_mse = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()

    # Guardo los resultados
    resultados.append({
        'max_depth': params['max_depth'],
        'max_leaf_nodes': params['max_leaf_nodes'],
        'min_samples_leaf': params['min_samples_leaf'],
        'train_mse': train_mse,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'r2_train': r2_train,
        'r2_val': r2_val,
        'r2_test': r2_test,
        'cross_val_mse': cross_val_mse
    })

# Mostrar los resultados en forma de tabla
resultados_df = pd.DataFrame(resultados)
print(resultados_df)

# Visualización de los resultados
plt.figure(figsize=(10, 6))
for i, params in enumerate(parametros):
    plt.plot(['Train', 'Validation', 'Test'], 
             [resultados[i]['train_mse'], resultados[i]['val_mse'], resultados[i]['test_mse']],
             marker='o', label=f"max_depth={params['max_depth']}, max_leaf_nodes={params['max_leaf_nodes']}, min_samples_leaf={params['min_samples_leaf']}")

plt.title('Comparación de MSE con Diferentes Hiperparámetros')
plt.xlabel('Dataset')
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
