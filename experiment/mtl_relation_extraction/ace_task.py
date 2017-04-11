import os
from . import config
from mtl_relation_extraction.io import arguments, ace_parser
from mtl_relation_extraction.task import Task, get_features, get_labels, \
    get_positions


class ACETask(Task):

    def __init__(self):
        super().__init__()
        self.name = "ACE"

    def load(self):
        ace_path = os.path.join(arguments.data_path, config.ace_path)
        self.train_relations = ace_parser.read_files(ace_path)
        self.train_features = get_features(
            self.train_relations
        )
        self.train_labels = get_labels(self.train_relations)
        (self.train_position1_vectors,
         self.train_position2_vectors) = get_positions(
            self.train_relations
        )
        self.init_encoder()
        self.init_num_positions()
