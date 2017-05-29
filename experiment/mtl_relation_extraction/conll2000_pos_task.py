from .rnn import RNN
from .sequence_cnn import SequenceCNN
from ..io import conll2000_parser


class Conll2000PosTask(SequenceCNN):
    def __init__(self):
        super().__init__(
            name="Conll2000POS",
            is_target=False
        )

    def load(self):
        train_sequences = conll2000_parser.conll2000pos()
        self.sequences = train_sequences
        self.train_sequences = train_sequences
        self.init_encoder()
