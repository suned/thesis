import numpy

from ..io import arguments

out_of_vocabulary = "#none#"


class Token:
    def __init__(self, word, rank, vector):
        self.word = word
        self.rank = rank
        self.vector = vector


class Vocabulary:
    def __init__(self):
        self.words = {}
        self.length = 1
        self.add(
            out_of_vocabulary,
            numpy.random.rand(arguments.word_embedding_dimension)
        )

    def add(self, word, vector):
        if word not in self.words:
            self.words[word] = Token(word, self.length, vector)
            self.length += 1

    def __getitem__(self, word):
        if word in self.words:
            return self.words[word].rank
        else:
            return self.words[out_of_vocabulary].rank

    def __contains__(self, word):
        return word in self.words

    def __iter__(self):
        return iter(self.words.values())
