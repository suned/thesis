import numpy
from sklearn import model_selection

from ..io import arguments
from . import inputs


def split(data, test_ratio):
    iterator = model_selection.ShuffleSplit(
            n_splits=2,
            test_size=test_ratio
    )
    train_indices, test_indices = next(
        iterator.split(data)
    )
    return data[train_indices], data[test_indices]


def get_labels(relations):
    return numpy.array(
        [train_relation.relation for train_relation in relations]
    )


def make_input(features,
               position1_vectors,
               position2_vectors):
    return {
        inputs.word_input: features,
        inputs.position1_input: position1_vectors,
        inputs.position2_input: position2_vectors
    }


class Task:
    def __init__(self, name, is_target, loss="categorical_crossentropy"):
        self.is_target = is_target
        self.name = name
        self.encoder = None
        self.num_classes = None
        self.model = None
        self.output_name = name + "_output"
        self.loss = loss

    def __repr__(self):
        return self.name

    def load(self):
        raise NotImplementedError()

    def reduce_train_data(self, fraction):
        raise NotImplementedError()

    def init_weights(self):
        del self.model
        self.compile_model()

    def get_batch(self, size=arguments.batch_size):
        raise NotImplementedError()

    def compile_model(self):
        raise NotImplementedError()

    def init_encoder(self):
        raise NotImplementedError()

    def get_vocabulary(self):
        raise NotImplementedError()

    def get_validation_vocabulary(self):
        return set()

    def longest_sentence(self):
        raise NotImplementedError()


def at_beginning(left):
    return left == 0


def beyond_edge(right, vector):
    return right > len(vector)
