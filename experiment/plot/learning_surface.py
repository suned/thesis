import pandas
import numpy
from .util import figure_size
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_learning_surface(metrics_path, path, task_name):
    metrics = pandas.read_csv(metrics_path)
    metrics.auxFraction = metrics.auxFraction.astype(str)
    metrics.targetFraction = metrics.targetFraction.astype(str)
    mean = metrics.groupby(["auxFraction", "targetFraction"]).mean()["f1"]
    xs, ys = numpy.meshgrid(mean.reset_index()["auxFraction"],
                            mean.reset_index()["targetFraction"])
    z = numpy.array([mean[x][y] for x, y in zip(xs.ravel(), ys.ravel())])
    zs = z.reshape(xs.shape)
    width, height = figure_size(1.3)
    width += .5
    fig = pyplot.figure(figsize=(width, height))
    ax = fig.add_subplot(111, projection='3d')
    xs = xs.astype(float)
    ys = ys.astype(float)
    ax.plot_surface(xs, ys, zs, cmap="coolwarm", antialiased=False)
    ax.set_xlabel("\nfraction of\n" + task_name)
    ax.set_ylabel("\nfraction of\nSemEval data")
    ax.set_zlabel("\nmean F1")
    metrics_name = metrics_path.split("/")[-2]
    out_path = os.path.join(path, metrics_name + "_learningSurface.pgf")
    ax.dist = 12
    fig.savefig(out_path)
