import argparse
import os
import csv
from io import StringIO

field_names = [
    "id",
    "e1",
    "relation",
    "e2",
    "e1_normalized",
    "relation_normalized",
    "e2_normalized",
    "occurrences",
    "confidence",
    "urls"
]


def parse_line(line):
    return next(
        csv.DictReader(StringIO(line), fieldnames=field_names)
    )


def run():
    parser = argparse.ArgumentParser(
        "Download sentences for reverb dataset"
    )
    parser.add_argument(
        "path",
        help="path to reverb data",
        type=str
    )
    parser.add_argument(
        "--chunk-size",
        help="number of chunks to read to memory at once",
        type=int,
        default=100
    )
    args = parser.parse_args()
    train_path = os.path.join(args.path, "train.tsv")
    sentences_path = os.path.join(args.path, "sentences.tsv")
    train_file = open(train_path)
    sentences_file = open(sentences_path, "w")
    for line in train_file:
        row = parse_line(line)

if __name__ == "__main__":
    run()
