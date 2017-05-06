import numpy

from ..mtl_relation_extraction.ground_truth import Sequence


def get_words(sentence):
    return [token[0] for token in sentence]


def get_pos(sentence):
    return [token[1] for token in sentence]


def conll2000():
    sequences = []
    from nltk.corpus import conll2000
    for sentence in conll2000.tagged_sents(tagset="universal"):
        words = get_words(sentence)
        tags = get_pos(sentence)
        sequence = Sequence(
            words,
            tags
        )
        sequences.append(sequence)
    return numpy.array(sequences)




