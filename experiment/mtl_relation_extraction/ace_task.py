import os

from ..io import arguments
from .. import config
from ..io import ace_parser
from .task import (
    get_labels,
    split
)
from .cnn import CNN
from . import preprocessing
from .relation_task import get_vocabulary


class ACE(CNN):
    def get_validation_vocabulary(self):
        return (get_vocabulary(self.early_stopping_relations)
                if self.early_stopping_relations is not None
                else set())

    def __init__(self):
        super().__init__(name="ACE", is_target=False)

    def load(self):
        ace_path = os.path.join(arguments.data_path, config.ace_path)
        train_relations = ace_parser.read_files(ace_path)
        if arguments.fit_sequential:
            train_relations, early_stopping_relations = split(
                train_relations,
                arguments.early_stopping_ratio
            )
            self.early_stopping_relations = early_stopping_relations
            self.early_stopping_labels = get_labels(
                early_stopping_relations
            )
        self.input_length = preprocessing.max_distance(train_relations)
        self.train_relations = train_relations
        self.train_labels = get_labels(self.train_relations)
        self.init_encoder()
