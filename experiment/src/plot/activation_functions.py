import numpy as np
from matplotlib import pyplot as plt
from util import figure_size, format_ticks
import os


def plot_sigmoid(output_dir):
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))

    def sigmoid_prime(a):
        return sigmoid(a) * (1 - sigmoid(a))

    xs = np.arange(-10, 10, .1)
    y_sigmoid = sigmoid(xs)
    y_sigmoid_prime = sigmoid_prime(xs)
    file_name = os.path.join(output_dir, "sigmoid.pgf")
    plt.figure(figsize=figure_size())
    plt.plot(xs, y_sigmoid, label=r"$\sigma(a) = \frac{1}{1 + e^{-a}}$")
    plt.plot(xs, y_sigmoid_prime, label=r"$\frac{d\sigma}{da}$")
    y_ticks = np.arange(0, 1.1, .25)
    plt.yticks(y_ticks)
    plt.ylim((0, 1.1))
    plt.legend(loc="center left")
    plt.xlabel(r"$a$")
    format_ticks()
    axis = plt.gca()
    axis.spines["left"].set_position("center")
    axis.yaxis.set_tick_params(direction="in", pad=-25)
    plt.savefig(file_name)


def plot_relu(output_dir):
    def relu(a):
        return np.maximum(0, a)

    relu = np.vectorize(relu)

    def relu_prime(a):
        return a > 0

    xs = np.arange(-4, 5, 1)
    y_relu = relu(xs)
    x_relu_prime = (-4, 0, 0, 4)
    y_relu_prime = (0, 0, 1, 1)

    file_name = os.path.join(output_dir, "relu.pgf")
    plt.figure(figsize=figure_size())
    plt.plot(xs, y_relu, label=r"$\sigma(a) = \max(0, a)$")
    plt.plot(x_relu_prime, y_relu_prime, label=r"$\frac{d\sigma}{da}$")
    y_ticks = np.arange(0, 5, 1)
    plt.yticks(y_ticks)
    plt.ylim((0, 4))
    plt.legend(loc="center left")
    plt.xlabel(r"$a$")
    format_ticks()
    axis = plt.gca()
    axis.spines["left"].set_position("center")
    axis.yaxis.set_tick_params(direction="in")
    plt.savefig(file_name)
