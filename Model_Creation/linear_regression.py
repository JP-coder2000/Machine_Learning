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
        prediction = h(params, samples[i], intercept)
        total_error += (prediction - y[i]) ** 2
    return total_error / (2 * m)

