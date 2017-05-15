import numpy

from ..io import arguments, kbp37_parser
from .. import config
from .task import get_labels
from .cnn import CNN
from . import preprocessing

import os


class KBP37Task(CNN):

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
        dev_relations = kbp37_parser.read_file(dev_path)
        self.relations = numpy.concatenate(
            (
                train_relations,
                test_relations,
                dev_relations
            )
        )
        self.input_length = preprocessing.max_distance(self.relations)
        self.labels = get_labels(
            self.relations
        )
        self.init_encoder()

    def __init__(self):
        super().__init__(
            name="KBP37",
            is_target=False
        )
