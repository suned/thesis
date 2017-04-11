import unittest

from numpy import testing, zeros

import arguments
from io.arguments import config
from mtl_relation_extraction.ground_truth import GroundTruth

odd_sentence = "This is a sentence with an odd number of tokens."
even_sentence = "This is not a sentence with an odd number of tokens."
odd_relation = GroundTruth(
    sentence_id="odd",
    relation="test",
    sentence=odd_sentence,
    e1_offset=(0, 4),
    e2_offset=(10, 17),
    relation_args=("e1", "e2")
)
even_relation = GroundTruth(
    sentence_id="even",
    relation="test",
    sentence=even_sentence,
    e1_offset=(0, 4),
    e2_offset=(14, 21),
    relation_args=("e1", "e2")
)
arguments.max_len = len(odd_relation.sentence)
odd_full_vector = odd_relation.feature_vector()
arguments.max_len = len(even_relation.sentence)
even_full_vector = even_relation.feature_vector()


class TestGroundTruth(unittest.TestCase):

    def test_no_trim(self):
        self.assertEqual(
            len(odd_full_vector),
            len(odd_relation.sentence)
        )
        self.assertEqual(
            len(even_full_vector),
            len(even_relation.sentence)
        )

    def check_trim(self, relation, full_vector, left, right):
        feature_vector = relation.feature_vector()
        self.assertEqual(len(feature_vector), arguments.max_len)
        testing.assert_equal(
            feature_vector,
            full_vector[left:len(full_vector) - right]
        )

    def check_pad(self, relation, full_vector, left, right):
        feature_vector = relation.feature_vector()
        self.assertEqual(len(feature_vector), arguments.max_len)
        testing.assert_equal(
            feature_vector[left:len(feature_vector) - right],
            full_vector
        )
        left_zeros = zeros((left,))
        right_zeros = zeros((right,))
        testing.assert_equal(
            feature_vector[:left],
            left_zeros
        )
        testing.assert_equal(
            feature_vector[left + len(full_vector):],
            right_zeros
        )

    def test_trim_odd_sentence_odd_max_len(self):
        arguments.max_len = 7
        self.check_trim(odd_relation, odd_full_vector, 2, 2)

    def test_trim_odd_sentence_even_max_len(self):
        arguments.max_len = 8
        self.check_trim(odd_relation, odd_full_vector, 2, 1)

    def test_trim_even_sentence_odd_max_len(self):
        arguments.max_len = 9
        self.check_trim(even_relation, even_full_vector, 2, 1)

    def test_trim_even_sentence_even_max_len(self):
        arguments.max_len = 10
        self.check_trim(even_relation, even_full_vector, 1, 1)

    def test_pad_odd_sentence_odd_max_len(self):
        arguments.max_len = 13
        self.check_pad(odd_relation, odd_full_vector, 1, 1)

    def test_pad_odd_sentence_even_max_len(self):
        arguments.max_len = 14
        self.check_pad(odd_relation, odd_full_vector, 2, 1)

    def test_pad_even_sentence_odd_max_len(self):
        arguments.max_len = 17
        self.check_pad(even_relation, even_full_vector, 3, 2)

    def test_pad_even_sentence_even_max_len(self):
        arguments.max_len = 14
        self.check_pad(even_relation, even_full_vector, 1, 1)
