import os
import json
from json import JSONDecodeError

import numpy
from ..mtl_relation_extraction.ground_truth import Relation


def process(text):
    try:
        annotations = json.loads(text)
    except JSONDecodeError:
        import ipdb
        ipdb.sset_trace()
    if annotations["relLabels"][0] == "NO_RELATION":
        return []
    entities = json.loads(annotations["nePairs"])[0]
    e1_start = entities["m1"]["start"]
    e1_end = entities["m1"]["end"]
    e2_start = entities["m2"]["start"]
    e2_end = entities["m2"]["end"]
    label = annotations["relLabels"][0]
    relation = Relation(
        None,
        annotations["words"],
        (e1_start, e1_end),
        (e2_start, e2_end),
        label,
        ("", ""),
        tokenized=True
    )
    return [relation]


def get_relations(file):
    relations = []
    with open(file) as json_file:
        text = json_file.read()
    texts = text.split("\n\n")
    texts = [text for text in texts if text != ""]
    for text in texts:
        relations.extend(process(text))
    return relations


def read_files(path):
    relations = []
    for path, _, filenames in os.walk(path):
        for file in filenames:
            if file.endswith("json"):
                file_name = os.path.join(path, file)
                relations.extend(get_relations(file_name))
    return numpy.array(relations)
