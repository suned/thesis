import os
from matplotlib import pyplot
import pandas
from .util import figure_size


labels = {
    "ACE": "+ACE 2005",
    "Conll2000POS": "+CONLL2000 POS",
    "Conll2000Chunk": "+CONLL2000 Chunking",
    "GMB-NER": "+GMB NER",
    "ACE_share_all_filters": "+ACE 2005"
}


def plot_learning_curves(results, path, name):
    semeval_path = os.path.join("results", "SemEval", "metrics.csv")
    semeval = pandas.read_csv(semeval_path)
    if "auxFraction" in semeval:
        semeval = semeval[semeval.auxFraction > .9]
    for root, _, filenames in os.walk(results):
        if root.endswith("SemEval"):
            continue
        for file in filenames:
            if file == "metrics.csv":
                aux_path = os.path.join(root, file)
                aux_name = os.path.split(root)[-1].split("+")[-1]
                aux_data = pandas.read_csv(aux_path)
                if "auxFraction" in aux_data:
                    aux_data = aux_data[aux_data.auxFraction > .9]
                pyplot.figure(figsize=figure_size())
                semeval.groupby("targetFraction").mean().f1.plot(
                    label="SemEval 2010 Task 8"
                )
                aux_data.groupby("targetFraction").mean().f1.plot(
                    label=labels[aux_name],
                    linestyle="--"
                )
                xs = semeval.groupby("targetFraction").mean().index.values
                x_labels = [
                    r"$0$",
                    r"$20$",
                    r"$40$",
                    r"$60$",
                    r"$80$",
                    r"$100$"]
                pyplot.xticks(xs, x_labels)
                pyplot.xlabel(r"percentage of $\mathcal{D}_t$")
                pyplot.ylabel("Mean macro F1")
                pyplot.legend()
                fig_path = os.path.join(
                    path,
                    name + "_" + aux_name + "_learning_curve.pgf"
                )
                pyplot.savefig(fig_path)
