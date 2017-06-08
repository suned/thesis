import numpy as np
import scipy
from scipy import signal
from matplotlib import pyplot
import os


def plot_gaussian(output_dir):
    pyplot.figure()
    xs = np.arange(0, 21, .1)
    ys = scipy.sin(xs) + np.random.normal(scale=.1, size=len(xs))
    kernel = signal.gaussian(len(xs), std=2.0, sym=True)
    kernel = kernel / sum(kernel)
    convolution = np.convolve(ys, kernel, mode='same')
    pyplot.plot(xs, ys, label=r"$f(x)$")
    pyplot.plot(xs, kernel, label=r"$k(x)$")
    pyplot.plot(xs, convolution, label=r"$(f * k)(x)$")
    axis = pyplot.gca()
    axis.spines["bottom"].set_position("zero")
    pyplot.legend()
    filename = os.path.join(output_dir, "gaussian.pgf")
    pyplot.savefig(filename)


def plot_feature_detector(output_dir):
    pyplot.figure()
    xs = np.arange(0, 21, .5)
    ys = signal.square(xs) + np.random.normal(
        scale=.06,
        size=len(xs)
    )
    kernel = np.zeros(ys.shape)
    half = len(kernel) // 2
    kernel[half - 3:half + 3] = .5
    kernel = kernel / sum(kernel)
    convolution = np.convolve(ys, kernel, mode='same')
    pyplot.plot(xs, ys, label=r"$f(x)$")
    pyplot.plot(xs, kernel, label=r"$k(x)$")
    pyplot.plot(xs, convolution, label=r"$(f * k)(x)$")
    axis = pyplot.gca()
    axis.spines["bottom"].set_position("zero")
    pyplot.legend()
    filename = os.path.join(output_dir, "detector.pgf")
    pyplot.savefig(filename)
