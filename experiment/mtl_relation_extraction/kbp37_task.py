import numpy

from ..io import arguments, kbp37_parser
from .. import config
from .task import get_labels
from .cnn import CNN
from . import preprocessing
from .relation_task import get_vocabulary

import os


class KBP37Task(CNN):
    def get_validation_vocabulary(self):
        return get_vocabulary(self.early_stopping_relations)

    def load(self):
        train_path = os.path.join(
            arguments.data_path,
            config.kbp37_train
        )
        test_path = os.path.join(
            arguments.data_path,
            config.kbp37_test
        )
        dev_path = os.path.join(
            arguments.data_path,
            config.kbp37_dev
        )
        train_relations = kbp37_parser.read_file(train_path)
        test_relations = kbp37_parser.read_file(test_path)
        self.train_relations = numpy.append(
            train_relations,
            test_relations
        )
        self.input_length = preprocessing.max_distance(train_relations)
        self.early_stopping_relations = kbp37_parser.read_file(dev_path)
        self.early_stopping_labels = get_labels(
            self.early_stopping_relations
        )
        self.train_labels = get_labels(
            self.train_relations
        )
        self.init_encoder()

    def __init__(self):
        super().__init__(
            name="KBP37",
            is_target=False
        )
