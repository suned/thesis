import unittest

from numpy import testing, zeros, array

from ..io import arguments
from ..mtl_relation_extraction.ground_truth import (
    Relation,
    nlp
)
from .. import config


class TestGroundTruth(unittest.TestCase):

    def setUp(self):
        config.max_distance = 10
        arguments.model = "cnn"
        odd_sentence = ("This is a sentence "
                        "with an odd number of tokens.")
        even_sentence = ("This is not a sentence with "
                         "an odd number of tokens.")
        self.odd_relation = Relation(
            sentence_id="odd",
            relation="test",
            sentence=odd_sentence,
            e1_offset=(10, 18),
            e2_offset=(27, 30),
            relation_args=("e1", "e2")
        )
        self.even_relation = Relation(
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
            [-3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]
        ) + config.max_distance
        expected_odd_position2_vector = array(
            [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        ) + config.max_distance
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
            [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]
        ) + config.max_distance
        expected_even_position2_vector = array(
            [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        ) + config.max_distance
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
        expected_e1 = array([1, 2, 3, 4, 5, 6, 7]) + config.max_distance
        expected_e2 = array([-3, -2, -1, 1, 2, 3, 4]) + config.max_distance
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
        expected_e1 = array([1, 2, 3, 4, 5, 6, 7, 8]) + config.max_distance
        expected_e2 = array([-3, -2, -1, 1, 2, 3, 4, 5]) + config.max_distance
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
        expected_e1 = array([-1, 1, 2, 3, 4, 5, 6, 7, 8]) + config.max_distance
        expected_e2 = array([-4, -3, -2, -1, 1, 2, 3, 4, 5]) + config.max_distance
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
        expected_e1 = array([-2, -1, 1, 2, 3, 4, 5, 6, 7, 8]) + config.max_distance
        expected_e2 = array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) + config.max_distance
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
        expected_e1 = array(
            [0, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 0]
        ) + config.max_distance
        expected_e2 = array(
            [0, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 0]
        ) + config.max_distance
        e1, e2 = self.odd_relation.position_vectors()
        testing.assert_equal(
            e1,
            expected_e1
        )
        testing.assert_equal(
            e2,
            expected_e2
        )

    def test_pad_odd_sentence_even_max_len(self):
        arguments.max_len = 14
        self.check_pad(
            self.odd_relation,
            self.odd_full_feature_vector,
            2,
            1
        )
        expected_e1 = array(
            [0, 0, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 0]
        ) + config.max_distance
        expected_e2 = array(
            [0, 0, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 0]
        ) + config.max_distance
        e1, e2 = self.odd_relation.position_vectors()
        testing.assert_equal(
            e1,
            expected_e1
        )
        testing.assert_equal(
            e2, expected_e2
        )

    def test_pad_even_sentence_odd_max_len(self):
        arguments.max_len = 17
        self.check_pad(
            self.even_relation,
            self.even_full_feature_vector,
            3,
            2
        )
        expected_e1 = array(
            [0, 0, 0, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0]
        ) + config.max_distance
        expected_e2 = array(
            [0, 0, 0, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 0, 0]
        ) + config.max_distance
        e1, e2 = self.even_relation.position_vectors()
        testing.assert_equal(
            e1,
            expected_e1
        )
        testing.assert_equal(
            e2,
            expected_e2
        )

    def test_pad_even_sentence_even_max_len(self):
        arguments.max_len = 14
        self.check_pad(
            self.even_relation,
            self.even_full_feature_vector,
            1,
            1
        )
        expected_e1 = array(
            [0, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 0]
        ) + config.max_distance
        expected_e2 = array(
            [0, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 0]
        ) + config.max_distance
        e1, e2 = self.even_relation.position_vectors()
        testing.assert_equal(
            e1,
            expected_e1
        )
        testing.assert_equal(
            e2,
            expected_e2
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
        expected_e1 = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) + config.max_distance
        expected_e2 = array(
            [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1]
        ) + config.max_distance
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
        expected_e1 = array([1, 1, 1, 1, 1, 1, 1]) + config.max_distance
        expected_e2 = array([-4, -3, -2, -1, 1, 2, 3]) + config.max_distance
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
        expected_e1 = array([1, 1, 2, 3, 4, 5, 6]) + config.max_distance
        expected_e2 = array([-3, -2, -1, 1, 1, 2, 3]) + config.max_distance
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
        expected_e1 = array([-3, -2, -1, 1, 2, 3, 4]) + config.max_distance
        expected_e2 = array([1, 2, 3, 4, 5, 6, 7]) + config.max_distance
        testing.assert_equal(
            e1_positions,
            expected_e1
        )
        testing.assert_equal(
            e2_positions,
            expected_e2
        )

    def test_long_sentence(self):
        arguments.max_len = 31
        s = ("In 1998 Go-Ahead Group replaced Northwest and Delta as "
            "GHI's owner.FundingUniverse The Go-Ahead Group Plc : "
            "Company History In 2000 Go-ahead merged GHI with its "
            "other UK aircraft ground handling operations Midland "
            "Airport Services British Midland Handling Services and "
            "Reed Aviation under the Aviance UK brand ..")
        test_relation = Relation(
            sentence_id="test",
            relation="test",
            sentence=s,
            e1_offset=(8, 22),
            e2_offset=(167, 169),
            relation_args=("e1", "e2")
        )
        features = test_relation.feature_vector()
        self.check_length(features)
        expected_e1 = array(
            [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        ) + config.max_distance
        expected_e2 = array(
            [-33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22,
             -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10,
             -9, -8, -7, -6, -5, -4, -3]
        ) + config.max_distance
        e1, e2 = test_relation.position_vectors()
        testing.assert_equal(
            e1,
            expected_e1
        )
        testing.assert_equal(
            e2,
            expected_e2
        )

    def test_rnn_features(self):
        entity_start_rank = nlp.vocab.length + 2
        entity_end_rank = nlp.vocab.length + 3
        arguments.model = "rnn"
        arguments.max_len = 15
        features = self.odd_relation.feature_vector()
        expected_features = [
            118,
            11,
            6,
            entity_start_rank,
            1905,
            entity_end_rank,
            26,
            60,
            entity_start_rank,
            1879,
            entity_end_rank,
            482,
            8,
            10540,
            1
        ]
        testing.assert_equal(
            features,
            expected_features
        )



