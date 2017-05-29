import numpy
from sklearn.preprocessing.label import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from ..io import arguments
from .task import Task
from . import inputs, nlp


def get_features(sequences):
    return [sequence.feature_vector() for sequence in sequences]


class SequenceTask(Task):

    def longest_sentence(self):
        return max(len(sequence.sentence)
                   for sequence in self.sequences)

    def get_validation_vocabulary(self):
        return set()

    def get_vocabulary(self):
        return set(
            [word for sequence in self.sequences for word
             in sequence.sentence]
        )

    def reduce_train_data(self, fraction):
        n = len(self.sequences)
        size = int(n / fraction) if fraction != 0. else 0

        batch_indices = numpy.random.randint(
            0,
            high=n,
            size=size
        )
        self.train_sequences = self.sequences[batch_indices]

    def init_encoder(self):
        classes = numpy.unique(
            [tag for sequence in self.sequences
             for tag in sequence.tags] + [nlp.pad_token]
        )
        self.encoder = LabelEncoder()
        self.encoder.fit(classes)
        encoded_pad = self.encoder.transform([nlp.pad_token])[0]
        assert encoded_pad == nlp.pad_rank
        self.num_classes = len(self.encoder.classes_)

    def __init__(self, name, is_target):
        super().__init__(name, is_target)
        self.sequences = None
        self.train_sequences = None

    def get_batch(self, size=arguments.batch_size):
        n = len(self.train_sequences)
        if n == 0:
            return self.format_set(numpy.array([]))
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
        encoded_tags = self.encode_tags(tags)
        padded_tags = pad_sequences(
            encoded_tags,
            maxlen=self.longest_sentence(),
            dtype=int,
            value=nlp.pad_rank
        )
        one_hot_tags = self.to_one_hot(padded_tags)
        features = get_features(sequences)
        padded_features = pad_sequences(
            features,
            value=nlp.pad_rank,
            dtype=int,
            maxlen=self.longest_sentence()
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
            encoded = self.encoder.transform(tags)
            encoded_tags.append(encoded)
        return encoded_tags

    def to_one_hot(self, encoded_tags):
        one_hot_tags = []
        for tags in encoded_tags:
            one_hot = to_categorical(
                tags,
                num_classes=self.num_classes
            )
            one_hot_tags.append(one_hot)
        return numpy.array(one_hot_tags)
