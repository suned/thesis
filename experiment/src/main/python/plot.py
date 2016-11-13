import matplotlib as mpl
from cycler import cycler
import numpy as np
mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#268bd2', '#2aa198', '#859900'])
from matplotlib import pyplot as plt
font_size = 10

plt.figure(figsize=(5,3))
plt.plot(range(5), label="$\\Delta(a)$")
plt.plot(range(5), np.sin(range(5)), label="$\\beta(a)$")
plt.xlabel(u"µ is not $\\mu$")
plt.ylabel("this is the y label")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().xaxis.set_ticks_position("bottom")
plt.gca().yaxis.set_ticks_position("left")
labels = plt.gca().yaxis.get_major_ticks()
labels[0].label1.set_visible(False)
plt.legend(loc='upper left', frameon=False, prop={'size': font_size})
plt.tight_layout()
plt.savefig("img/test.pgf")
