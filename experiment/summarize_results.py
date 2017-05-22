import argparse
import pandas
from scipy.stats import ttest_ind

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Summarize a metrics file from an experiment"
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to results file"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline results file",
        default="results/SemEval/metrics.csv"
    )
    args = parser.parse_args()

    metrics = pandas.read_csv(args.file)
    baseline_metrics = pandas.read_csv(args.baseline)
    metrics_header = "{0:<12} {1:>10} {2:>10}".format("", "Mean", "STD")
    metrics_line = "{0:<10} : {1:>10.4f} {2:>10.4f}"
    mean = metrics.mean()
    std = metrics.std()
    t_value, p_value = ttest_ind(
        metrics["f1"],
        baseline_metrics["f1"]
    )
    f1_line = metrics_line.format(
        "F1",
        mean["f1"],
        std["f1"]
    )
    precision_line = metrics_line.format(
        "Precision",
        mean["precision"],
        std["precision"]
    )
    recall_line = metrics_line.format(
        "Recall",
        mean["recall"],
        std["recall"]
    )
    t_line = "t value : {:<0.4f}".format(t_value)
    p_line = "p value : {:<0.4f}".format(p_value / 2)
    print()
    print("Metrics")
    print("=======")
    print(metrics_header)
    print(f1_line)
    print(precision_line)
    print(recall_line)
    print()
    print("Comparison With Baseline")
    print("========================")
    print(t_line)
    print(p_line)
    print()
