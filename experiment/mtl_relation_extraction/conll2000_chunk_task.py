from ..io import conll2000_parser
from .sequence_cnn import SequenceCNN


class Conll2000ChunkTask(SequenceCNN):
    def __init__(self):
        super().__init__(name="Conll2000Chunk", is_target=False)

    def load(self):
        train_sequences = conll2000_parser.conll2000chunk()
        self.sequences = train_sequences
        self.init_encoder()
