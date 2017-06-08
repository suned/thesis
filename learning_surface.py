# coding: utf-8
get_ipython().magic('matplotlib')
import pandas
metrics = pandas.read_csv("results/SemEval+ACE/metrics.csv") 
mean = metrics.gropby(["auxFraction", "targetFraction"]).mean()
mean = metrics.groupby(["auxFraction", "targetFraction"]).mean()
mean
mean.values()
mean.values
mean.index
mean.index.values
mean.reset_index()
mean["f1"].reset_index()
mean["f1"].reset_index().values
from matplotlib import pyplot
mean = mean["f1"].reset_index()
mean
import numpy
numpy.meshgrid(mean["auxFraction"], mean["targetFraction"])
X, Y = numpy.meshgrid(mean["auxFraction"], mean["targetFraction"])
X
mean = metrics.groupby(["auxFraction", "targetFraction"]).mean()["f1"]
mean
mean.ix[1.0]
mean.ix[1.0][0.0]
mean.index
mean.index.levels
metrics
metrics["auxFraction"]
xs, ys = numpy.meshgrid(metrics["auxFraction"], metrics["targetFraction"])
xs
ys
xs.ravel()
mean
z = numpy.array([mean[x][y] for x, y in zip(xs.ravel(), ys.ravel())])
xz.shape
xs.shape
xs, ys = numpy.meshgrid(mean.reset_index()["auxFraction"], mean.reset_index()["targetFraction"])
xs.shape
mean
z = numpy.array([mean[x][y] for x, y in zip(xs.ravel(), ys.ravel())])
z.shape
z.reshape(xs.shape)
zs = z.reshape(xs.shape)
zs.shape
fig = pyplot.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, zs)
ax.xaxis.set_label("aux fraction")
fig = pyplot.figure()
ax.plot_surface(xs, ys, zs, cmap="coolwarm")
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, zs, cmap="coolwarm")
ax.plot_surface(xs, ys, zs, cmap="coolwarm", antialiased=False)
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, zs, cmap="coolwarm", antialiased=False)
fig.colorbar()
ax.xaxis.label
ax.xaxis.set_label("percentage of ACE data")
ax.zaxis.set_label("mean F1")
