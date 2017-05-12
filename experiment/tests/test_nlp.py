from unittest import TestCase, mock
import numpy
import logging
from numpy import testing

from ..io import arguments

arguments.word_embedding_dimension = 300
from ..mtl_relation_extraction import nlp
from ..mtl_relation_extraction.vocabulary import out_of_vocabulary

vector = numpy.zeros(arguments.word_embedding_dimension)


class NLPTest(TestCase):
    def setUp(self):
        nlp.glove_vectors = {
            "This": vector + 1,
            "is": vector + 2,
            "a": vector + 3,
            "sentence": vector + 4,
            "with": vector + 5,
            "an": vector + 6,
            "odd": vector + 7,
            "number": vector + 8,
            "of": vector + 9,
            "tokens": vector + 10,
            "some": vector + 11,
        }
        nlp.vocabulary = nlp.Vocabulary()

    @mock.patch("numpy.random.rand")
    def test_add_vocabulary(self, rand_mock):
        mock_task = mock.Mock()
        rand_mock.return_value = vector
        mock_task.get_train_vocabulary.return_value = {
            "This",
            "is",
            "a",
            "sentence",
            "with",
            "an",
            "odd",
            "number",
            "of",
            "tokens",
            "."
        }
        mock_task.get_validation_vocabulary.return_value = {
            "This",
            "is",
            "a",
            "sentence",
            "with",
            "some",
            "new",
            "tokens",
            "."
        }
        train_vocabulary = mock_task.get_train_vocabulary()
        validation_vocabulary = mock_task.get_validation_vocabulary()
        nlp.add_vocabularies([mock_task])
        vocabulary = set([token.word for token in nlp.vocabulary])
        self.assertTrue(
            train_vocabulary.issubset(vocabulary)
        )
        self.assertTrue(
            validation_vocabulary.difference(vocabulary) == {"new"}
        )
        testing.assert_equal(
            nlp.glove_vectors["This"],
            nlp.vocabulary.words["This"].vector
        )
        testing.assert_equal(
            nlp.glove_vectors["is"],
            nlp.vocabulary.words["is"].vector
        )
        testing.assert_equal(
            nlp.glove_vectors["a"],
            nlp.vocabulary.words["a"].vector
        )
        testing.assert_equal(
            nlp.glove_vectors["sentence"],
            nlp.vocabulary.words["sentence"].vector
        )
        testing.assert_equal(
            nlp.glove_vectors["with"],
            nlp.vocabulary.words["with"].vector
        )
        testing.assert_equal(
            nlp.glove_vectors["an"],
            nlp.vocabulary.words["an"].vector
        )
        testing.assert_equal(
            nlp.glove_vectors["odd"],
            nlp.vocabulary.words["odd"].vector
        )
        testing.assert_equal(
            nlp.glove_vectors["number"],
            nlp.vocabulary.words["number"].vector
        )
        testing.assert_equal(
            nlp.glove_vectors["number"],
            nlp.vocabulary.words["number"].vector
        )
        testing.assert_equal(
            nlp.glove_vectors["of"],
            nlp.vocabulary.words["of"].vector
        )
        testing.assert_equal(
            nlp.glove_vectors["tokens"],
            nlp.vocabulary.words["tokens"].vector
        )
        testing.assert_equal(
            nlp.vocabulary.words["."].vector,
            vector
        )
        self.assertEqual(
            nlp.vocabulary["new"],
            nlp.vocabulary[out_of_vocabulary]
        )

