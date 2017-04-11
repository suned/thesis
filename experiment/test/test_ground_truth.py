import unittest

from numpy import testing, zeros, array

from mtl_relation_extraction.io import arguments
from mtl_relation_extraction.ground_truth import GroundTruth

odd_sentence = "This is a sentence with an odd number of tokens."
even_sentence = "This is not a sentence with an odd number of tokens."
odd_relation = GroundTruth(
    sentence_id="odd",
    relation="test",
    sentence=odd_sentence,
    e1_offset=(10, 18),
    e2_offset=(27, 30),
    relation_args=("e1", "e2")
)
even_relation = GroundTruth(
    sentence_id="even",
    relation="test",
    sentence=even_sentence,
    e1_offset=(14, 22),
    e2_offset=(31, 33),
    relation_args=("e1", "e2")
)
arguments.max_len = len(odd_relation.sentence)
odd_full_feature_vector = odd_relation.feature_vector()
arguments.max_len = len(even_relation.sentence)
even_full_feature_vector = even_relation.feature_vector()


class TestGroundTruth(unittest.TestCase):

    def test_no_trim(self):
        self.assertEqual(
            len(odd_full_feature_vector),
            len(odd_relation.sentence)
        )
        self.assertEqual(
            len(even_full_feature_vector),
            len(even_relation.sentence)
        )

    def check_trim(self, relation, full_vector, left, right):
        feature_vector = relation.feature_vector()
        self.check_length(feature_vector)
        testing.assert_equal(
            feature_vector,
            full_vector[left:len(full_vector) - right]
        )

    def check_length(self, vector):
        self.assertEqual(len(vector), arguments.max_len)

    def check_pad(self, relation, full_vector, left, right):
        feature_vector = relation.feature_vector()
        self.check_length(feature_vector)
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
        self.check_trim(odd_relation, odd_full_feature_vector, 3, 1)
        e1_positions, e2_positions = odd_relation.position_vectors()
        self.check_length(e1_positions)
        self.check_length(e2_positions)
        expected_e1 = array([0, 1, 2, 3, 4, 5, 6]) + 7
        expected_e2 = array([-3, -2, -1, 0, 1, 2, 3]) + 7
        testing.assert_equal(
            e1_positions,
            expected_e1
        )
        testing.assert_equal(
            e2_positions,
            expected_e2
        )

    def test_trim_odd_sentence_even_max_len(self):
        arguments.max_len = 8
        self.check_trim(odd_relation, odd_full_feature_vector, 3, 0)
        e1_positions, e2_positions = odd_relation.position_vectors()
        self.check_length(e1_positions)
        self.check_length(e2_positions)
        expected_e1 = array([0, 1, 2, 3, 4, 5, 6, 7]) + 8
        expected_e2 = array([])

    def test_trim_even_sentence_odd_max_len(self):
        arguments.max_len = 9
        self.check_trim(even_relation, even_full_feature_vector, 3, 0)

    def test_trim_even_sentence_even_max_len(self):
        arguments.max_len = 10
        self.check_trim(even_relation, even_full_feature_vector, 2, 0)

    def test_pad_odd_sentence_odd_max_len(self):
        arguments.max_len = 13
        self.check_pad(odd_relation, odd_full_feature_vector, 1, 1)

    def test_pad_odd_sentence_even_max_len(self):
        arguments.max_len = 14
        self.check_pad(odd_relation, odd_full_feature_vector, 2, 1)

    def test_pad_even_sentence_odd_max_len(self):
        arguments.max_len = 17
        self.check_pad(even_relation, even_full_feature_vector, 3, 2)

    def test_pad_even_sentence_even_max_len(self):
        arguments.max_len = 14
        self.check_pad(even_relation, even_full_feature_vector, 1, 1)
