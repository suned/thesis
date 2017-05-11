import numpy

from ..io import arguments
from . import log

out_of_vocabulary = "#none#"


class Token:
    def __init__(self, word, rank, vector):
        self.word = word
        self.rank = rank
        self.vector = vector


class Vocabulary:
    def __init__(self):
        self.words = {}
        self.length = 0
        self.add(
            out_of_vocabulary,
            numpy.random.rand(arguments.word_embedding_dimension)
        )

    def add(self, word, vector):
        if word not in self.words:
            self.length += 1
            self.words[word] = Token(word, self.length, vector)

    def __getitem__(self, word):
        if word in self.words:
            return self.words[word].rank
        else:
            return self.words[out_of_vocabulary].rank

    def __contains__(self, word):
        return word in self.words

    def __iter__(self):
        return iter(self.words.values())

    def get_embeddings(self):
        vectors = numpy.random.rand(
            # indices:
            # padding: 0
            # out of vocab: 1
            self.length + 1,
            arguments.word_embedding_dimension
        ) / 100
        for lex in self:
            vectors[lex.rank] = lex.vector
        log.info("Word embedding shape: %s", vectors.shape)
        return vectors
