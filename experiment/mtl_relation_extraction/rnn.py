from keras import layers, models
from ..io import arguments
from . import tasks, log, embedding, input_layers
from .. import config


def compile_models():
    log.info("Compiling RNN models")
    (word_input,
     position1_input,
     position2_input,
     entity_marker_input) = input_layers.make_inputs()
    position_embedding = embedding.make_position_embedding(
        "position_embedding"
    )
    position1_embedding = position_embedding(position1_input)
    position2_embedding = position_embedding(position2_input)
    word_embedding = embedding.make_word_embedding()(word_input)
    embedding_layer = layers.concatenate(
        [word_embedding, position1_embedding, position2_embedding],
        name="embedding_layer"
    ) if not arguments.entity_markers else layers.concatenate(
        [word_embedding, entity_marker_input]
    )
    bi_lstm = layers.Bidirectional(
        layers.LSTM(
            arguments.hidden_layer_dimension,
            return_sequences=True,
            activation="relu",
        ),
        name="bi_lstm"
    )(embedding_layer)
    lstm = layers.LSTM(
        arguments.hidden_layer_dimension,
        activation="relu",
        name="lstm"
    )(bi_lstm)
    output = tasks.target_task.get_output(lstm)
    inputs = [
        word_input,
        position1_input,
        position2_input
    ] if not arguments.entity_markers else [
        word_input,
        entity_marker_input
    ]
    model = models.Model(
        inputs=inputs,
        outputs=output
    )
    model.compile(
        optimizer=config.optimizer,
        loss="categorical_crossentropy"
    )
    tasks.target_task.model = model
    model.summary()
