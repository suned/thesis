from unittest import TestCase
import numpy
from numpy import testing

from ..io import arguments
from ..mtl_relation_extraction.vocabulary import (
    Vocabulary,
    out_of_vocabulary
)

arguments.word_embedding_dimension = 300


class TestVocabulary(TestCase):
    def test_ranks(self):
        vocabulary = Vocabulary()
        self.assertEqual(vocabulary.length, 1)
        self.assertEqual(vocabulary[out_of_vocabulary], 1)
        vector = numpy.zeros(arguments.word_embedding_dimension) + 1
        vocabulary.add("first", vector)
        self.assertEqual(vocabulary.length, 2)
        self.assertEqual(vocabulary["first"], 2)
        testing.assert_equal(
            vector,
            vocabulary.words["first"].vector
        )
        vector = numpy.zeros(arguments.word_embedding_dimension) + 2
        vocabulary.add("second", vector)
        self.assertEqual(vocabulary.length, 3)
        self.assertEqual(vocabulary["second"], 3)
        testing.assert_equal(
            vector,
            vocabulary.words["second"].vector
        )
        self.assertEqual(
            vocabulary["new"],
            1
        )

    def test_embeddings(self):
        vocabulary = Vocabulary()
        vector1 = numpy.zeros(arguments.word_embedding_dimension) + 1
        vocabulary.add("first", vector1)
        vector2 = numpy.zeros(arguments.word_embedding_dimension) + 2
        vocabulary.add("second", vector2)
        embeddings = vocabulary.get_embeddings()
        self.assertEqual(
            embeddings.shape,
            (4, arguments.word_embedding_dimension)
        )
        first_rank = vocabulary["first"]
        testing.assert_equal(
            vector1,
            embeddings[first_rank]
        )
        second_rank = vocabulary["second"]
        testing.assert_equal(
            vector2,
            embeddings[second_rank]
        )
        out_of_vocabulary_rank = vocabulary["new"]
        testing.assert_equal(
            embeddings[out_of_vocabulary_rank],
            vocabulary.words[out_of_vocabulary].vector
        )
