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

    for _ in range(iterations):
        # Le sumo el bias a la hipótesis, esto lo platiqué con Benji para no tener que tener un arreglo con 1 al principio.
        predictions = np.dot(samples.T, params) + bias
        errors = predictions - y
        
        # Calcular los gradientes
        gradient_params = (1/m) * np.dot(samples, errors)
        gradient_bias = (1/m) * np.sum(errors)
        
        # Actualizar los parámetros y el bias
        params -= alpha * gradient_params
        bias -= alpha * gradient_bias

    return params, bias

#samples = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
#y = np.array([2, 4, 6, 8, 10])

# Coeficiente de aprendizaje 
alpha = 0.01 
# Numero de iteraciones máximas
iterations = 1000000


# Después de hacer el ETL, ahora si voy a usar mi modelo lineal para empezar a predecir los precios de los carros.
# Pero primero tengo que hacer lo que vimos en clase de separar los datos en entrenamiento y prueba.

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Esta es una función muy sencilla para separar los datos de entrenamiento y prueba.
    """
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[-n_test:]
    train_indices = indices[:-n_test]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Update, tengo que hacer la función para sacer el r^2
def r2_score(y_true, y_pred):
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Empiezo con el import de los datos limpios y escalados
df_scaled = pd.read_csv('automobile_cleaned_scaled.csv')

# Dropeo la columna de precios para tener mis datos de entrenamiento
X = df_scaled.drop('Price', axis=1).values
y = df_scaled['Price'].values

# Separo los datos en entrenamiento y prueba
# Le pongo 42 porque 42 es el resultado para todo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = np.zeros(X_train.shape[1])
bias = 0.0 


params, bias = gradient_descent(params, bias, X_train.T, y_train, alpha, iterations)

# Calcular el error final en los datos de entrenamiento y prueba
final_cost_train = cost_function(params, bias, X_train, y_train)
final_cost_test = cost_function(params, bias, X_test, y_test)

#print("Parámetros finales:", params)
print("Bias final:", bias)
print("Error final (entrenamiento):", final_cost_train)
print("Error final (prueba):", final_cost_test)

# Calcular R^2
y_train_pred = np.dot(X_train, params) + bias
y_test_pred = np.dot(X_test, params) + bias

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R^2 en conjunto de entrenamiento:", r2_train)
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

# Generar predicciones
y_train_pred = np.dot(X_train, params) + bias
y_test_pred = np.dot(X_test, params) + bias


plt.figure(figsize=(14, 8))


plt.subplot(3, 1, 1)
plt.scatter(range(len(y_train)), y_train, color='blue', label='Actual Training Data')
plt.scatter(range(len(y_train)), y_train_pred, color='orange', label='Predicted Training Data')
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()


plt.subplot(3, 1, 2)
plt.scatter(range(len(y_test)), y_test, color='green', label='Actual Test Data')
plt.scatter(range(len(y_test)), y_test_pred, color='red', label='Predicted Test Data')
plt.title('Test Data: Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()

plt.figure(figsize=(14, 8))

# Comparación de datos de entrenamiento y prueba con los valores reales
plt.scatter(range(len(y_train)), y_train, color='blue', label='Actual Training Data', alpha=0.6)
plt.scatter(range(len(y_test)), y_test, color='green', label='Actual Test Data', alpha=0.6)
plt.scatter(range(len(y_train_pred)), y_train_pred, color='orange', label='Predicted Training Data', alpha=0.6)
plt.scatter(range(len(y_test_pred)), y_test_pred, color='red', label='Predicted Test Data', alpha=0.6)

plt.title('Comparison of Actual vs Predicted Data for Training and Test Sets')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()
