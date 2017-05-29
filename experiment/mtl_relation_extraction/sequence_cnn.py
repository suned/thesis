import numpy
from keras import layers, models
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from . import inputs, nlp, convolutions
from . import embeddings
from .sequence_task import SequenceTask
from ..io import arguments
from .. import config


class SequenceCNN(SequenceTask):
    def __init__(self,
                 name,
                 is_target,
                 window_size=arguments.window_size):
        super().__init__(name, is_target)
        self.window_size = window_size

    def load(self):
        raise NotImplementedError()

    def to_one_hot(self):
        pass

    def init_encoder(self):
        classes = numpy.unique(
            [tag for sequence in self.sequences
             for tag in sequence.tags]
        )
        self.encoder = LabelEncoder()
        self.encoder.fit(classes)
        self.num_classes = len(self.encoder.classes_)

    def format_set(self, sequences):
        if not sequences:
            windows = numpy.array([]).reshape((0, self.window_size))
            labels = numpy.array([]).reshape((0, self.num_classes))
            return self.make_in_out_pair(labels, windows)
        windows = []
        tags = []
        half_window_size = self.window_size // 2
        for sequence in sequences:
            length = len(sequence.sentence)
            for index in range(length):
                start = index - half_window_size
                if start < 0:
                    left_padding = abs(start)
                    start = 0
                else:
                    left_padding = 0
                end = index + half_window_size + 1
                if end > length:
                    right_padding = end - length
                    end = length
                else:
                    right_padding = 0
                window = sequence.sentence[start:end]
                window_features = [nlp.vocabulary[token]
                                   for token in window]
                padded_window = numpy.pad(
                    window_features,
                    (left_padding, right_padding),
                    "constant"
                )
                tag = sequence.tags[index]
                windows.append(padded_window)
                tags.append(tag)
        windows = numpy.array(windows)
        encoded_tags = self.encoder.transform(tags)
        one_hot_tags = to_categorical(
            encoded_tags,
            num_classes=self.num_classes
        )
        return self.make_in_out_pair(one_hot_tags, windows)

    def make_in_out_pair(self, one_hot_tags, windows):
        input = {
            inputs.word_input: windows
        }
        output = {
            self.output_name: one_hot_tags
        }
        return input, output

    def compile_model(self):
        word_input = inputs.make_word_input(
            input_length=self.window_size
        )
        word_embedding = embeddings.shared_word_embedding(
            word_input
        )

        pooling_layers = []
        if arguments.share_filters:
            convolution_layers = convolutions.shared_convolutions
        else:
            convolution_layers = convolutions.make_convolution_layers(
                prefix=self.name + "_"
            )
        for convolution in convolution_layers:
            convolution_layer = convolution(
                word_embedding
            )
            pooling_layer = layers.GlobalMaxPool1D()(
                convolution_layer
            )
            pooling_layers.append(pooling_layer)
        pooling_concat = layers.concatenate(
            pooling_layers
        )
        output = layers.Dense(
            units=self.num_classes,
            activation="relu",
            name=self.name + "_output"
        )(pooling_concat)
        model = models.Model(
            inputs=word_input,
            outputs=output
        )
        model.compile(
            optimizer=config.optimizer,
            loss=self.loss
        )
        self.model = model
