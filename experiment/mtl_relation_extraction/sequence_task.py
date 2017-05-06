import numpy
from sklearn.preprocessing.label import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from spacy.tokens import Doc

from ..io import arguments
from .. import config
from .task import Task
from . import nlp, inputs


def get_features(sentence):
    if type(sentence) == Doc:
        return numpy.array(
            [token.rank
             if token.has_vector
             else nlp.vocab.length + 1
             for token in sentence]
        )
    if type(sentence) == list:
        return numpy.array(
            [nlp.vocab[token]
             if token in nlp.vocab
             else nlp.vocab.length + 1
             for token in sentence]
        )
    raise ValueError(
        "Unsupported sentence type: " + str(type(sentence))
    )


class SequenceTask(Task):
    def init_encoder(self):
        all_tags = numpy.concatenate(
            [sequence.tags for sequence in self.train_sequences] +
            [config.pad_rank]
        )
        classes = numpy.unique(all_tags)
        self.encoder = LabelEncoder()
        self.encoder.fit(classes)
        self.num_classes = len(self.encoder.classes_)

    def early_stopping_set(self):
        return self.format_set(self.early_stopping_sequences)

    def init_longest_sentence(self):
        self.longest_sentence = max(
            len(sequence.sentence)
            for sequence in self.train_sequences
        )

    def __init__(self, name, is_target):
        super().__init__(name, is_target)
        self.train_sequences = None
        self.early_stopping_sequences = None
        self.longest_sentence = None

    def get_batch(self, size=arguments.batch_size):
        n = len(self.train_sequences)

        batch_indices = numpy.random.randint(
            0,
            high=n,
            size=size
        )
        batch_sequences = self.train_sequences[batch_indices]
        return self.format_set(batch_sequences)

    def compile_model(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def format_set(self, sequences):
        tags = [sequence.tags for sequence in sequences]
        features = [get_features(sequence.sentence)
                    for sequence in sequences]
        padded_tags = pad_sequences(
            tags,
            maxlen=self.longest_sentence,
            value=config.pad_rank
        )
        encoded_tags = self.encode_tags(padded_tags)
        one_hot_tags = self.to_one_hot(encoded_tags)
        padded_features = pad_sequences(
            features,
            maxlen=self.longest_sentence,
            value=config.pad_rank
        )
        input = {
            inputs.word_input: padded_features
        }
        output = {
            self.output_name: one_hot_tags
        }
        return input, output

    def encode_tags(self, padded_tags):
        encoded_tags = []
        for tags in padded_tags:
            encoded = []
            for tag in tags:
                encoded.append(self.encoder.transform(tag))
            encoded_tags.append(encoded)
        return encoded_tags

    def to_one_hot(self, encoded_tags):
        one_hot_tags = []
        for tags in encoded_tags:
            one_hot = []
            for tag in tags:
                one_hot.append(
                    to_categorical(tag, num_classes=self.num_classes)
                )
            one_hot_tags.append(one_hot)
        return one_hot_tags
