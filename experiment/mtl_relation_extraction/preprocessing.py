from . import config


def first_entity_index(relation):
    return relation.e1[0]


def last_entity_index(relation):
    return relation.e2[1]


def max_distance(relations):
    return max(
        last_entity_index(relation) - first_entity_index(relation)
        for relation in relations
    )


def max_sentence_length(relations):
    return max_distance(relations) + config.max_len_buffer


def longest_sentence(relations):
    return max(len(relation.sentence) for relation in relations)
