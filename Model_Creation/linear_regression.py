import numpy as np

def hypothesis(params, sample, intercept):
    """
    Calcula la hipótesis de una ecuación lineal. Sumo el bias al final de la ecuación.
    Me parece mejor de esta forma en vez de hacer el arreglo con 1 al principio.
    """
    return np.dot(params, sample) + intercept

def cost_function(params, intercept, samples, y):
    """
    Calculo la función del MSE, es la que hemos visto en clase, nada fancy.
    """
    m = len(y)
    total_error = 0
    for i in range(m):
        prediction = hypothesis(params, samples[i], intercept)
        total_error += (prediction - y[i]) ** 2
    return total_error / (2 * m)


def gradient_descent(params, intercept, samples, y, alpha, iterations):
    """
    Implementa el algoritmo de gradiente descendiente, 
    """
    m = len(y)

    for _ in range(iterations):
        #Le sumo la intercepión a la hipótesis, esto lo platiqué con Benji para no tener que tener un arreglo con 1 al principio.
        predictions = np.dot(samples, params) + intercept        
        errors = predictions - y
        
        # Calcular los gradientes
        gradient_params = (1/m) * np.dot(samples.T, errors)
        gradient_intercept = (1/m) * np.sum(errors)
        
        # Actualizar los parámetros y el intercepto
        params -= alpha * gradient_params
        intercept -= alpha * gradient_intercept

    return params, intercept


params = np.array([0.0, 0.0]) 
intercept = 0.0 


samples = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
y = np.array([2, 4, 6, 8, 10])

# Coeficiente de aprendizaje 
alpha = 0.01 
# Numero de iteraciones máximas
iterations = 1000 


params, intercept = gradient_descent(params, intercept, samples, y, alpha, iterations)


print("Parámetros finales:", params)
print("Intercepto final:", intercept)


final_cost = cost_function(params, intercept, samples, y)
print("Costo final:", final_cost)
