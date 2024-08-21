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
iterations = 1000 


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

# Calcular el costo final en los datos de entrenamiento
final_cost_train = cost_function(params, bias, X_train, y_train)

# Calcular el costo final en los datos de prueba
final_cost_test = cost_function(params, bias, X_test, y_test)

print("Parámetros finales:", params)
print("Bias final:", bias)
print("Costo final (entrenamiento):", final_cost_train)
print("Costo final (prueba):", final_cost_test)

# Intento hacer una predicción
y_pred = np.dot(X_test, params) + bias
print(y_pred)

# Ahora quiero organizar todo lo que veo, lo voy a plotear en un scatter plot para ver si se ve bien.
# Esto lo hago porque como Benji nos dijo en clase, tenemos que ver que haga sentido lo que estamos haciendo.
#Plot de train
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_train)), y_train, color='blue', label='Actual Training Data')
plt.scatter(range(len(y_train)), np.dot(X_train, params) + bias, color='orange', label='Predicted Training Data')
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()
#Plot de test
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Test Data')
plt.scatter(range(len(y_test)), y_pred, color='orange', label='Predicted Test Data')
plt.title('Test Data: Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()
