from keras import layers, models

from . import inputs, nlp
from . import embeddings
from .sequence_task import SequenceTask
from ..io import arguments
from .. import config


class RNN(SequenceTask):
    def load(self):
        raise NotImplementedError()

    def compile_model(self):
        word_input = inputs.make_word_input(
            input_length=self.longest_sentence()
        )
        mask = layers.Masking(
            mask_value=nlp.pad_rank,
            dtype=int
        )(word_input)
        word_embedding = embeddings.shared_word_embedding(
            mask
        )
        bi_lstm = layers.Bidirectional(
            layers.LSTM(
                units=arguments.hidden_layer_dimension,
                return_sequences=True
            )
        )(word_embedding)
        output = layers.TimeDistributed(
            layers.Dense(self.num_classes, activation="softmax"),
            name=self.output_name
        )(bi_lstm)
        model = models.Model(
            inputs=word_input,
            outputs=output
        )
        model.compile(
            optimizer=config.optimizer,
            loss=self.loss
        )
        self.model = model
