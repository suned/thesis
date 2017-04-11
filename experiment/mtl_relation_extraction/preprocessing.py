from . import config
from .io import arguments


def first_entity_index(relation):
    return relation.e1[0]


def last_entity_index(relation):
    return relation.e2[1]


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
