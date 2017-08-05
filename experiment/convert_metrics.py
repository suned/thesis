import os
import pandas


def convert(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file == "metrics.csv":
                filepath = os.path.join(root, file)
                metrics = pandas.read_csv(filepath)
                if "auxFraction" in metrics.columns:
                    no_aux = metrics.drop("auxFraction", axis=1)
                    no_aux.to_csv(filepath, index=False)
