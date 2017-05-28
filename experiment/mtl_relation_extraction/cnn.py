from keras import layers, models

from .. import config
from ..io import arguments
from . import inputs, embeddings, convolutions
from .relation_task import RelationTask


class CNN(RelationTask):
    def load(self):
        raise NotImplementedError

    def compile_model(self):
        word_input = inputs.make_word_input(
            input_length=self.input_length
        )
        position1_input, position2_input = (inputs
            .make_position_inputs(input_length=self.input_length)
        )
        word_embedding = embeddings.shared_word_embedding(word_input)
        position1_embedding = embeddings.shared_position_embedding(
            position1_input
        )
        position2_embedding = embeddings.shared_position_embedding(
            position2_input
        )
        pooling_layers = []
        if arguments.share_filters:
            word_convolutions = convolutions.shared_word_convolutions
            position_convolutions = (convolutions
                .shared_position_convolutions
            )
        else:
            (word_convolutions, position_convolutions) = (convolutions
                .make_convolution_layers(prefix=self.name + "_")
            )

        for convolution in word_convolutions:
            convolution_layer = convolution(
                word_embedding
            )
            pooling_layer = layers.GlobalMaxPool1D()(
                convolution_layer
            )
            pooling_layers.append(pooling_layer)
        for convolution in position_convolutions:
            convolution_layer = convolution(
                position1_embedding
            )
            pooling_layer = layers.GlobalMaxPool1D()(
                convolution_layer
            )
            pooling_layers.append(pooling_layer)
            convolution_layer = convolution(
                position2_embedding
            )
            pooling_layer = layers.GlobalMaxPool1D()(
                convolution_layer
            )
            pooling_layers.append(pooling_layer)
        pooling_layers_concatenation = layers.concatenate(
                pooling_layers
        )
        if arguments.dropout:
            drop_out = layers.Dropout(rate=.5)(
                pooling_layers_concatenation
            )
            output = self.get_output()(drop_out)
        else:
            output = self.get_output()(pooling_layers_concatenation)
        input_layers = [
            word_input,
            position1_input,
            position2_input
        ]
        model = models.Model(
            inputs=input_layers,
            outputs=output
        )
        model.compile(
            optimizer=config.optimizer,
            loss="categorical_crossentropy"
        )
        self.model = model
