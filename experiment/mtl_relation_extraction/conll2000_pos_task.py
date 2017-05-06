from .rnn import RNN
from ..io import conll_pos_parser, arguments
from .task import split


class Conll2000PosTask(RNN):
    def __init__(self):
        super().__init__(
            name="Conll2000POS",
            is_target=False
        )

    def load(self):
        train_sequences = conll_pos_parser.conll2000()
        train_sequences, early_stopping_sequences = split(
            train_sequences,
            test_ratio=arguments.early_stopping_ratio
        )
        self.train_sequences = train_sequences
        self.early_stopping_sequences = early_stopping_sequences
        self.init_encoder()
