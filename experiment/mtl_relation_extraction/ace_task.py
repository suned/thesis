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


class ACE(CNN):
    def get_validation_vocabulary(self):
        return set()

    def __init__(self):
        super().__init__(name="ACE", is_target=False)

    def load(self):
        ace_path = os.path.join(arguments.data_path, config.ace_path)
        train_relations = ace_parser.read_files(ace_path)
        self.input_length = preprocessing.max_distance(train_relations)
        self.relations = train_relations
        self.train_relations = self.relations
        self.labels = get_labels(self.relations)
        self.train_labels = self.labels
        self.init_encoder()
