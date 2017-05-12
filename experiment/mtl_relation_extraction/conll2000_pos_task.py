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
        if arguments.fit_sequential:
            train_sequences, early_stopping_sequences = split(
                train_sequences,
                test_ratio=arguments.early_stopping_ratio
            )
            self.early_stopping_sequences = early_stopping_sequences
        self.train_sequences = train_sequences
        self.init_encoder()
