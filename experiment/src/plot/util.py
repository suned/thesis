import numpy as np
from matplotlib import pyplot as plt


def format_ticks():
    axis = plt.gca()
    axis.xaxis.set_ticks_position("bottom")
    axis.yaxis.set_ticks_position("left")
    return axis


def remove_first_ylabel():
    axis = plt.gca()
    axis.yaxis.majorTicks[0].set_visible(False)


def figure_size(scale=1):
    fig_width_pt = 418.25368
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.0
    fig_width = fig_width_pt*inches_per_pt*scale
    fig_height = fig_width*golden_mean
    fig_size = [fig_width, fig_height]
    return fig_size
