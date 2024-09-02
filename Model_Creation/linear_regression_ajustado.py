"""
En este archivo estará mi modelo de regresión lineal ajustado, es decir, con regularización L2.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def hypothesis(params, sample, bias):
    """
    Calcula la hipótesis de una ecuación lineal. Sumo el bias al final de la ecuación.
    Me parece mejor de esta forma en vez de hacer el arreglo con 1 al principio.
    """
    return np.dot(params, sample) + bias


def cost_function(params, bias, samples, y):
    """
    Calculo la función del MSE, es la que hemos visto en clase, nada fancy. Me basé en este código de stack overflow:
    https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy, pero al platicarlo con benji, me dijo que para que 
    no fallara, tenía que tener mi y_prima ya calulada.
    """
    y_prima = np.dot(samples, params) + bias
    mse = np.mean(np.square(y_prima - y)) / 2
    
    return mse


def gradient_descent(params, bias, samples, y, alpha, iterations):
    """
    Implemento el algoritmo de gradiente descendiente, me base en el código que Benji nos dio de regresion lineal.
    Solo que en vez de hacerlo con listas, lo hago con arrays de numpy directamente.
    """
    m = len(y)
    mse_history = []

    for _ in range(iterations):
        # Le sumo el bias a la hipótesis, esto lo platiqué con Benji para no tener que tener un arreglo con 1 al principio.
        predictions = np.dot(samples.T, params) + bias
        errors = predictions - y
        
        mse = np.mean(np.square(errors)) / 2
        mse_history.append(mse)
        
        # Calcular los gradientes
        gradient_params = (1/m) * np.dot(samples, errors)
        gradient_bias = (1/m) * np.sum(errors)
        
        # Actualizar los parámetros y el bias
        params -= alpha * gradient_params
        bias -= alpha * gradient_bias

    return params, bias, mse_history

def cost_function_l2(params, bias, samples, y, lambda_):
    """
    Calculo el MSE con regularización L2, pero ahora con la regularización L2.
    """
    m = len(y)
    y_pred = np.dot(samples, params) + bias
    mse = np.mean(np.square(y_pred - y)) / 2
    l2_regularization = (lambda_ / (2 * m)) * np.sum(np.square(params))
    cost = mse + l2_regularization
    return cost

def gradient_descent_l2(params, bias, samples, y, alpha, iterations, lambda_):
    """
    Implemento gradiente descendente con regularización L2.
    """
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        y_pred = np.dot(samples, params) + bias
        errors = y_pred - y

        # Calcular los gradientes con regularización
        gradient_params = (1/m) * (np.dot(samples.T, errors) + lambda_ * params)
        gradient_bias = (1/m) * np.sum(errors)
        
        # Actualizar los parámetros y el bias
        params -= alpha * gradient_params
        bias -= alpha * gradient_bias
        
        # Guardar el historial de costo
        cost = cost_function_l2(params, bias, samples, y, lambda_)
        cost_history.append(cost)

    return params, bias, cost_history

def train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=None):
    """
    Esta es una función muy sencilla para separar los datos de entrenamiento y prueba.
    Update: Ahora también separo los datos que me van a ayudar para hacer de validación.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val
    
    indices = np.random.permutation(n_samples)
    
    X_train = X[indices[:n_train]]
    y_train = y[indices[:n_train]]
    
    X_val = X[indices[n_train:n_train + n_val]]
    y_val = y[indices[n_train:n_train + n_val]]
    
    X_test = X[indices[n_train + n_val:]]
    y_test = y[indices[n_train + n_val:]]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def r2_score(y_true, y_pred):
    """
    Calculo el coeficiente de determinación R^2.
    """
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2


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
    Grafico los valores predecidos vs los valores reales.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores predecidos')
    plt.title(f'Predecidos vs Reales - {set_name}')
    plt.grid(True)
    plt.show()

def plot_residuals(y_true, y_pred, set_name):
    """
    Grafico los residuales.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
    plt.xlabel('Valores predecidos')
    plt.ylabel('Residuales')
    plt.title(f'Grafica de Residuales- {set_name}')
    plt.grid(True)
    plt.show()
    

# Empiezo con el import de los datos limpios y escalados
df_scaled = pd.read_csv('automobile_cleaned_scaled.csv')

# Dropeo la columna de precios para tener mis datos de entrenamiento
X = df_scaled.drop('Price', axis=1).values
y = df_scaled['Price'].values

alpha = 0.01 
# Numero de iteraciones máximas
iterations = 1000000

# Separo los datos en entrenamiento, validación y prueba
# Le pongo 42 porque 42 es el resultado para todo
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42)

params = np.zeros(X_train.shape[1])
bias = 0.0 
lambda_ = 0.1  # Factor de regularización

params, bias, mse_history = gradient_descent_l2(params, bias, X_train, y_train, alpha, iterations, lambda_)

# Evaluar en conjunto de validación
y_val_pred = np.dot(X_val, params) + bias
val_mse = cost_function_l2(params, bias, X_val, y_val, lambda_)
val_r2 = r2_score(y_val, y_val_pred)

# Calcular el error final en los datos de entrenamiento y prueba
final_cost_train = cost_function_l2(params, bias, X_train, y_train, lambda_)
final_cost_test = cost_function_l2(params, bias, X_test, y_test, lambda_)

#print("Parámetros finales:", params)
print("Bias final:", bias)
print("Error final (entrenamiento):", final_cost_train)
print("Error en validación:", val_mse)
print("Error final (prueba):", final_cost_test)

# Calcular R^2
y_train_pred = np.dot(X_train, params) + bias
y_test_pred = np.dot(X_test, params) + bias

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R^2 en conjunto de entrenamiento:", r2_train)
print("R^2 en conjunto de validación:", val_r2)
print("R^2 en conjunto de prueba:", r2_test)

# Determinar si el modelo está fitting, underfitting o overfitting
if abs(r2_train - r2_test) < 0.05:
    print("El modelo está correctamente ajustado (fitting).")
elif r2_train > r2_test:
    print("El modelo está sobreajustado (overfitting).")
else:
    print("El modelo está subajustado (underfitting).")
    
    
# Calcular el grado de bias basado en R^2 en el conjunto de entrenamiento
if r2_train > 0.8:
    print("Bias bajo.")
elif 0.5 <= r2_train <= 0.8:
    print("Bias medio.")
else:
    print("Bias alto.")

# Gráfica del historial de costo durante el entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(range(len(mse_history)), mse_history, label='Training Cost (L2)')
plt.title('Cost Function with L2 Regularization')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.yscale('log')  # Usando escala logarítmica para mejor visualización
plt.legend()
plt.show()

# Comparación de datos de entrenamiento, validación y prueba con los valores reales
plt.figure(figsize=(14, 8))

plt.scatter(range(len(y_train)), y_train, color='blue', label='Actual Training Data', alpha=0.6)
plt.scatter(range(len(y_val)), y_val, color='purple', label='Actual Validation Data', alpha=0.6)
plt.scatter(range(len(y_test)), y_test, color='green', label='Actual Test Data', alpha=0.6)

plt.scatter(range(len(y_train)), np.dot(X_train, params) + bias, color='orange', label='Predicted Training Data', alpha=0.6)
plt.scatter(range(len(y_val)), y_val_pred, color='pink', label='Predicted Validation Data', alpha=0.6)
plt.scatter(range(len(y_test)), y_test_pred, color='red', label='Predicted Test Data', alpha=0.6)

plt.title('Comparison of Actual vs Predicted Data for Training, Validation, and Test Sets')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()

grafica_costo(mse_history)

# Graficar los valores predecidos vs los valores reales en el conjunto de entrenamiento
y_train_pred = np.dot(X_train, params) + bias
grafica_predicted_vs_actual(y_train, y_train_pred, 'Training Set')

# Graficar los valores predecidos vs los valores reales en el conjunto de prueba
y_test_pred = np.dot(X_test, params) + bias
grafica_predicted_vs_actual(y_test, y_test_pred, 'Test Set')

# Graficar los residuales en el conjunto de entrenamiento
plot_residuals(y_train, y_train_pred, 'Training Set')

# Graficar los residuales en el conjunto de prueba
plot_residuals(y_test, y_test_pred, 'Test Set')