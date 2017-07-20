from matplotlib import pyplot
from util import figure_size, format_ticks
import os

def plot_semeval_relations(path):
    xs = [
        "Cause-Effect",
        "Component-Whole",
        "Entity-Destination",
        "Entity-Origin",
        "Product-Producer",
        "Member-Collection",
        "Message-Topic",
        "Content-Container",
        "Instrument-Agency",
        "Other"
    ]
    ys = [
        .124,
        .117,
        .106,
        .091,
        .088,
        .086,
        .084,
        .068,
        .062,
        .174
    ]
    pyplot.figure(figsize=figure_size())
    pyplot.bar(range(len(ys)), ys)
    pyplot.xticks(range(len(xs)), xs, rotation="vertical")
    pyplot.ylim((0, .2))
    format_ticks()
    file_name = os.path.join(path, "semeval_dist.pgf")
    pyplot.savefig(file_name)
