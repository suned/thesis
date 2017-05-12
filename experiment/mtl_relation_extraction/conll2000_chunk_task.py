from ..io import conll2000_parser, arguments
from .rnn import RNN
from .task import split


class Conll2000ChunkTask(RNN):
    def __init__(self):
        super().__init__(name="Conll2000Chunk", is_target=False)

    def load(self):
        train_sequences = conll2000_parser.conll2000chunk()
        train_sequences, early_stopping_sequences = split(
            train_sequences,
            test_ratio=arguments.early_stopping_ratio
        )
        self.train_sequences = train_sequences
        self.early_stopping_sequences = early_stopping_sequences
        self.init_encoder()
