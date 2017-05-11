from unittest import TestCase
import numpy
from numpy import testing

from ..io import arguments
arguments.word_embedding_dimension = 300
from ..mtl_relation_extraction.ground_truth import Relation
from ..mtl_relation_extraction.nlp import vocabulary


class TestGroundTruth(TestCase):
    def setUp(self):
        odd_sentence = ("This is a sentence "
                        "with an odd number of tokens. Plumbus")
        self.odd_relation = Relation(
            sentence_id="odd",
            relation="test",
            sentence=odd_sentence,
            e1_offset=(10, 18),
            e2_offset=(27, 30),
            relation_args=("e1", "e2")
        )
        vector = numpy.random.rand(arguments.word_embedding_dimension)
        vocabulary.add("This", vector)
        vocabulary.add("is", vector)
        vocabulary.add("a", vector)
        vocabulary.add("sentence", vector)
        vocabulary.add("with", vector)
        vocabulary.add("an", vector)
        vocabulary.add("odd", vector)
        vocabulary.add("number", vector)
        vocabulary.add("of", vector)
        vocabulary.add("tokens", vector)
        vocabulary.add(".", vector)

    def test_get_features(self):
        features = self.odd_relation.feature_vector()
        expected_features = numpy.array(
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]
        )
        testing.assert_equal(
            features,
            expected_features
        )
