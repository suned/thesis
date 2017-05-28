import os

from .sequence_cnn import SequenceCNN
from ..io import arguments, gmb_parser
from .. import config
from .task import split


class GMBNERTask(SequenceCNN):
    def __init__(self):
        super().__init__(name="GMB-NER", is_target=False)

    def load(self):
        corpus_root = os.path.join(arguments.data_path, config.gmb_root)
        train_sequences = gmb_parser.gmb_named_entities(corpus_root)
        self.sequences = train_sequences
        self.init_encoder()

