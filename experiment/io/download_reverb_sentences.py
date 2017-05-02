import argparse
import os
import csv
from bs4 import BeautifulSoup
from urllib.error import HTTPError, URLError

from io import StringIO
from nltk import sent_tokenize
from ..mtl_relation_extraction import log
from urllib.request import urlopen

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

header = "\t".join([
    "id",
    "sentence",
    "relation",
    "relation_normalized",
    "e1_start",
    "e1_end",
    "e2_start",
    "e2_end"
])


def parse_line(line):
    return next(
        csv.DictReader(
            StringIO(line),
            fieldnames=field_names,
            delimiter="\t"
        )
    )


class SentenceNotFoundException(Exception):
    pass


def find_indices(e1, e2, sentence):
    e1_start = sentence.find(e1)
    e1_end = e1_start + len(e1)
    e2_start = sentence.find(e2)
    e2_end = e2_start + len(e2)
    if e1_start == -1 or e2_start == -1:
        return None, None
    return (e1_start, e1_end), (e2_start, e2_end)


def find_sentence(text, row):
    relation = row["relation"]
    e1 = row["e1"]
    e2 = row["e2"]
    for sentence in sent_tokenize(text):
        if (relation in sentence
            and e1 in sentence
            and e2 in sentence):
            e1_indices, e2_indices = find_indices(e1, e2, sentence)
            if e1_indices is not None and e2_indices is not None:
                return sentence, e1_indices, e2_indices
    raise SentenceNotFoundException


def run(path):
    train_path = os.path.join(path, "train.tsv")
    sentences_path = os.path.join(path, "sentences.tsv")
    train_file = open(train_path)
    sentences_file = open(sentences_path, "w")
    sentences_file.write(header)
    for line in train_file:
        row = parse_line(line)
        urls = row["urls"].split("|")
        sentence_id = row["id"]
        for url in urls:
            try:
                page = urlopen(url, timeout=10).read()
                text = BeautifulSoup(page, "lxml").text
                log.info("Page found: %s", url)
                sentence, e1, e2 = find_sentence(text, row)
                relation = row["relation"]
                relation_normalized = row["relation_normalized"]
                row = "\t".join([
                    sentence_id,
                    sentence,
                    relation,
                    relation_normalized,
                    e1[0],
                    e1[1],
                    e2[0],
                    e2[1]]) + "\n"
                sentences_file.write(row)
            except HTTPError:
                log.error("Page not found: %s", url)
            except URLError:
                log.error("Bad url: %s", url)
            except SentenceNotFoundException:
                log.error("No sentence found for %s", sentence_id)
    train_file.close()
    sentences_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Download sentences for reverb dataset"
    )
    parser.add_argument(
        "path",
        help="path to reverb data",
        type=str
    )
    args = parser.parse_args()
    run(args.path)
