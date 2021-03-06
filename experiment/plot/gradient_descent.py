from sklearn.linear_model import SGDRegressor
import math
import numpy as np
import random
from matplotlib import pyplot as plt
from util import figure_size, format_ticks, move_spines_to_zero, remove_first_ylabel, remove_first_xlabel
import os
from scipy import misc


def cost(x, y, h):
    squared_error = (h.predict(x) - y)**2
    return np.mean(squared_error)


def plot_gradient_steps(path):
    def target_function(x):
        return 5 * x + np.random.normal(scale=3, size=len(x))

    x_min = 0
    x_max = 10
    x = np.linspace(x_min, x_max, 100)
    y = target_function(x)
    x = x.reshape((-1, 1))
    data = list(zip(x, y))
    n = 20
    d_train = random.sample(data, n)
    x_train, y_train = zip(*d_train)
    x_train = np.array(x_train).reshape((-1, 1))
    y_train = np.array(y_train)
    h = SGDRegressor(n_iter=1, warm_start=True, penalty="none")
    w1s = np.linspace(-30, 30, 100)
    w2s = np.linspace(0, 10, 100)
    W1, W2 = np.meshgrid(w1s, w2s)
    costs = np.zeros(W1.shape)
    for index, _ in np.ndenumerate(W1):
        w1 = W1[index]
        w2 = W2[index]
        h.intercept_ = w1
        h.coef_ = np.array([w2]).reshape((1, 1))
        c = cost(x, y, h)
        costs[index] = c

    plt.figure(figsize=figure_size())
    values = np.arange(1, 300, 30)**1.4
    plt.contour(W1, W2, costs, values)
    plt.xlabel(r"$w_0$")
    #plt.title(r"$\hat{E}(\mathbf{w}, \mathcal{D}_{train})$")

    steps = 6
    h = SGDRegressor(
        n_iter=1,
        warm_start=True,
        penalty="none",
        eta0=0.037,
        learning_rate="constant",
        shuffle=False
    )
    h.intercept_ = np.array([-25.0])
    h.coef_ = np.array([1.0])
    weights = []
    colors = ["red", "green", "blue", "magenta", "cyan", "black"]
    plot_weights(colors, 0, h.intercept_[0], h.coef_[0], weights)
    for step in range(1, steps):
        h.fit(x_train, y_train)
        w0 = h.intercept_[0]
        w1 = h.coef_[0]
        plot_weights(colors, step, w0, w1, weights)
    format_ticks()
    file_path = os.path.join(path, "cost_function.pgf")
    move_spines_to_zero()
    plt.ylabel(r"$w_1$", labelpad=-30)
    remove_first_ylabel()
    plt.savefig(file_path)

    plt.figure(figsize=figure_size())
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.plot(x_train, y_train, "o", label=r"$\mathcal{D}$")
    step = 0
    for w0, w1, color in weights:
        label = r"$y = \mathbf{w}_" + str(step) + r"^T\tilde{\mathbf{x}}$"
        y_pred = [
            w1 * x_min + w0,
            w1 * x_max + w0
            ]
        plt.plot(
            [x_min, x_max],
            y_pred,
            c=color
        )
        step += 1
    plt.legend(borderaxespad=2.)
    format_ticks()
    file_name = os.path.join(path, "d_train.pgf")
    move_spines_to_zero()
    remove_first_xlabel()
    plt.savefig(file_name)


def plot_weights(colors, step, w0, w1, weights):
    color = colors[step]
    weights.append((w0, w1, color))
    plt.plot(w0, w1, "o", c=color)
    plt.text(w0 + 1, w1, r"$\mathbf{w}_" + str(step) + r"$")


def plot_early_stopping(path="data"):
    e_train = lambda i: 1 / (1.7 ** i) + .9
    e = lambda i: -(i * .5) / (1.25 ** i) + 2
    plt.figure(figsize=figure_size())
    iss = np.arange(0, 10, .1)
    plt.plot(
        iss,
        e_train(iss),
        label=r"$\hat{E}(\mathbf{w}_i, \mathcal{D})$"
    )
    plt.plot(
        iss,
        e(iss),
        label=r"$E(\mathbf{w}_i)$"
    )
    plt.tick_params(
        axis="y",
        which="both",
        left="off",
        labelleft="off"
    )
    plt.xticks([])
    plt.xticks([4.5], [r'$i^*$'])
    plt.tick_params(
        axis="x",
        which="both",
        bottom="off",
        labelsize="12"
    )
    plt.axvline(x=4.5, linestyle="--", ymax=.27, color="black", linewidth=1)
    plt.xlabel(r"Iterations $i$", horizontalalignment="right", position=(1,25))
    plt.legend()
    filename = os.path.join(path, "early_stopping.pgf")
    plt.savefig(filename)
