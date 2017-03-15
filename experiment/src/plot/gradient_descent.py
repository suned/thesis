from sklearn.linear_model import SGDRegressor
import math
import numpy as np
import random
from matplotlib import pyplot as plt


def cost(x, y, h):
    squared_error = (h.predict(x) - y)**2
    return np.mean(squared_error)


def plot_gradient_steps():
    def target_function(x):
        return x + 10

    target_function = np.vectorize(target_function)
    x = np.linspace(-15, 15, 100)
    y = target_function(x)
    x = x.reshape((-1, 1))
    data = list(zip(x, y))
    n = 100
    d_train = [random.choice(data) for _ in range(n)]
    x_train, y_train = zip(*d_train)
    h = SGDRegressor(n_iter=1, warm_start=True, penalty="none")
    w1s = np.linspace(0, 20, 100)
    w2s = np.linspace(0, 2, 100)
    W1, W2 = np.meshgrid(w1s, w2s)
    costs = np.zeros(W1.shape)
    for index, _ in np.ndenumerate(W1):
        w1 = W1[index]
        w2 = W2[index]
        h.intercept_ = w1
        h.coef_ = np.array([w2]).reshape((1, 1))
        c = cost(x, y, h)
        costs[index] = c
    plt.figure()
    values = np.arange(0.1, 10, .5)
    contour = plt.contour(W1, W2, costs, values)
