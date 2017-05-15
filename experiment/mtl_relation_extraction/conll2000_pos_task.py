from .rnn import RNN
from ..io import conll2000_parser, arguments
from .task import split


class Conll2000PosTask(RNN):
    def __init__(self):
        super().__init__(
            name="Conll2000POS",
            is_target=False
        )

    def load(self):
        train_sequences = conll2000_parser.conll2000pos()
        self.sequences = train_sequences
        self.init_encoder()
