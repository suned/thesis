import unittest

from numpy import testing, zeros, array

from ..io import arguments
from ..mtl_relation_extraction.ground_truth import (
    GroundTruth
)


class TestGroundTruth(unittest.TestCase):

    def setUp(self):
        odd_sentence = ("This is a sentence "
                        "with an odd number of tokens.")
        even_sentence = ("This is not a sentence with "
                         "an odd number of tokens.")
        self.odd_relation = GroundTruth(
            sentence_id="odd",
            relation="test",
            sentence=odd_sentence,
            e1_offset=(10, 18),
            e2_offset=(27, 30),
            relation_args=("e1", "e2")
        )
        self.even_relation = GroundTruth(
            sentence_id="even",
            relation="test",
            sentence=even_sentence,
            e1_offset=(14, 22),
            e2_offset=(31, 34),
            relation_args=("e1", "e2")
        )
        arguments.max_len = len(self.odd_relation.sentence)
        self.odd_full_feature_vector = (self
            .odd_relation.feature_vector())
        (self.odd_full_position1_vector,
         self.odd_full_position2_vector) = (self
            .odd_relation
            .position_vectors()
        )
        arguments.max_len = len(self.even_relation.sentence)
        self.even_full_feature_vector = (self
            .even_relation.feature_vector())
        (self.even_full_position1_vector,
         self.even_full_position2_vector) = (self
            .even_relation
            .position_vectors()
        )

    def test_no_trim(self):
        self.assertEqual(
            len(self.odd_full_feature_vector),
            len(self.odd_relation.sentence)
        )
        expected_odd_position1_vector = array(
            [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
        ) + 11
        expected_odd_position2_vector = array(
            [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        ) + 11
        testing.assert_equal(
            expected_odd_position1_vector,
            self.odd_full_position1_vector
        )
        testing.assert_equal(
            expected_odd_position2_vector,
            self.odd_full_position2_vector
        )
        self.assertEqual(
            len(self.even_full_feature_vector),
            len(self.even_relation.sentence)
        )
        expected_even_position1_vector = array(
            [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
        ) + 12
        expected_even_position2_vector = array(
            [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        ) + 12
        testing.assert_equal(
            expected_even_position1_vector,
            self.even_full_position1_vector
        )
        testing.assert_equal(
            expected_even_position2_vector,
            self.even_full_position2_vector
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
        self.check_trim(
            self.odd_relation,
            self.odd_full_feature_vector,
            3,
            1
        )
        e1_positions, e2_positions = (self
            .odd_relation
            .position_vectors())
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
        self.check_trim(
            self.odd_relation,
            self.odd_full_feature_vector,
            3,
            0
        )
        e1_positions, e2_positions = (self
            .odd_relation
            .position_vectors())
        self.check_length(e1_positions)
        self.check_length(e2_positions)
        expected_e1 = array([0, 1, 2, 3, 4, 5, 6, 7]) + 8
        expected_e2 = array([-3, -2, -1, 0, 1, 2, 3, 4]) + 8
        testing.assert_equal(
            e1_positions,
            expected_e1
        )
        testing.assert_equal(
            e2_positions,
            expected_e2
        )

    def test_trim_even_sentence_odd_max_len(self):
        arguments.max_len = 9
        self.check_trim(
            self.even_relation,
            self.even_full_feature_vector, 3, 0
        )
        e1_positions, e2_positions = (self
            .even_relation
            .position_vectors())
        self.check_length(e1_positions)
        self.check_length(e2_positions)
        expected_e1 = array([-1, 0, 1, 2, 3, 4, 5, 6, 7]) + 9
        expected_e2 = array([-4, -3, -2, -1, 0, 1, 2, 3, 4]) + 9
        testing.assert_equal(
            e1_positions,
            expected_e1
        )
        testing.assert_equal(
            e2_positions,
            expected_e2
        )

    def test_trim_even_sentence_even_max_len(self):
        arguments.max_len = 10
        self.check_trim(
            self.even_relation,
            self.even_full_feature_vector,
            2,
            0
        )
        e1_positions, e2_positions = (self
            .even_relation.position_vectors())
        self.check_length(e1_positions)
        self.check_length(e2_positions)
        expected_e1 = array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) + 10
        expected_e2 = array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]) + 10
        testing.assert_equal(
            e1_positions,
            expected_e1
        )
        testing.assert_equal(
            e2_positions,
            expected_e2
        )

    def test_pad_odd_sentence_odd_max_len(self):
        arguments.max_len = 13
        self.check_pad(
            self.odd_relation,
            self.odd_full_feature_vector,
            1,
            1
        )

    def test_pad_odd_sentence_even_max_len(self):
        arguments.max_len = 14
        self.check_pad(
            self.odd_relation,
            self.odd_full_feature_vector,
            2,
            1
        )

    def test_pad_even_sentence_odd_max_len(self):
        arguments.max_len = 17
        self.check_pad(
            self.even_relation,
            self.even_full_feature_vector,
            3,
            2
        )

    def test_pad_even_sentence_even_max_len(self):
        arguments.max_len = 14
        self.check_pad(
            self.even_relation,
            self.even_full_feature_vector,
            1,
            1
        )

    def test_no_negatives(self):
        arguments.max_len = 11
        self.odd_relation.e1 = (0, 1)
        self.odd_relation.e2 = (10, 11)
        e1_positions, e2_positions = (self
            .odd_relation
            .position_vectors())
        self.check_length(e1_positions)
        self.check_length(e2_positions)
        expected_e1 = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 11
        expected_e2 = array(
            [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]
        ) + 11
        testing.assert_equal(
            e1_positions,
            expected_e1
        )
        testing.assert_equal(
            e2_positions,
            expected_e2
        )

    def test_overlapping_entities(self):
        arguments.max_len = 7
        self.odd_relation.e1 = (0, 9)
        self.odd_relation.e2 = (4, 5)
        (e1_position,
         e2_position) = self.odd_relation.position_vectors()
        expected_e1 = array([0, 0, 0, 0, 0, 0, 0]) + 7
        expected_e2 = array([-4, -3, -2, -1, 0, 1, 2]) + 7
        self.check_length(e1_position)
        self.check_length(e2_position)
        testing.assert_equal(
            e1_position,
            expected_e1
        )
        testing.assert_equal(
            e2_position,
            expected_e2
        )

    def test_long_entities(self):
        arguments.max_len = 7
        self.odd_relation.e1 = (1, 3)
        self.odd_relation.e2 = (4, 6)
        (e1_position,
         e2_position) = self.odd_relation.position_vectors()
        expected_e1 = array([0, 0, 1, 2, 3, 4, 5]) + 7
        expected_e2 = array([-3, -2, -1, 0, 0, 1, 2]) + 7
        self.check_length(e1_position)
        self.check_length(e2_position)
        testing.assert_equal(
            e1_position,
            expected_e1
        )
        testing.assert_equal(
            e2_position,
            expected_e2
        )

    def test_e2_is_first_entity(self):
        arguments.max_len = 7
        e1 = self.odd_relation.e1
        self.odd_relation.e1 = self.odd_relation.e2
        self.odd_relation.e2 = e1
        self.check_trim(
            self.odd_relation,
            self.odd_full_feature_vector,
            3,
            1
        )
        e1_positions, e2_positions = (self
                                      .odd_relation
                                      .position_vectors())
        self.check_length(e1_positions)
        self.check_length(e2_positions)
        expected_e1 = array([-3, -2, -1, 0, 1, 2, 3]) + 7
        expected_e2 = array([0, 1, 2, 3, 4, 5, 6]) + 7
        testing.assert_equal(
            e1_positions,
            expected_e1
        )
        testing.assert_equal(
            e2_positions,
            expected_e2
        )

