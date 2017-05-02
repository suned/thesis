import os
import unicodedata

from .. import config
from ..io import arguments


def first_entity_index(relation):
    return relation.first_entity()[0]


def last_entity_index(relation):
    return relation.last_entity()[1]


def entity_distance(e1, e2):
    return e2[1] - e1[0]


def max_distance(relations):
    return max(
        last_entity_index(relation) - first_entity_index(relation)
        for relation in relations
    )


def max_sentence_length(relations):
    if arguments.dynamic_max_len:
        return max_distance(relations) + config.max_len_buffer
    else:
        return arguments.max_len


def longest_sentence(relations):
    return max(len(relation.sentence) for relation in relations)


def clean_glove_vectors(vector_path, length=300):
    path, extension = os.path.splitext(vector_path)
    clean_path = path + ".clean" + extension
    line_count = 0
    with open(vector_path) as vector_file, \
         open(clean_path, "w") as clean_file:
        for vector_string in vector_file:
            if line_count == 77292:
                continue
            decoded = unicodedata.normalize("NFKC", vector_string)
            splat = decoded.split(" ")
            if len(splat) == length + 1:
                clean_file.write(decoded)
            line_count += 1
