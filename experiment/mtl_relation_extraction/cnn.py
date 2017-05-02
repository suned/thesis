from keras import layers, models


from . import log
from .embedding import make_word_embedding, make_position_embedding
from .input_layers import make_inputs
from .tasks import experiment_tasks
from .. import config
from ..io import arguments


def compile_models():
    log.info("Compiling cnn models")
    (word_input,
     position1_input,
     position2_input,
     entity_marker_input) = make_inputs()
    shared_layers = make_shared_layers(
        word_input,
        position1_input,
        position2_input,
        entity_marker_input
    )
    inputs = [
        word_input,
        position1_input,
        position2_input
    ] if not arguments.entity_markers else [
        word_input,
        entity_marker_input
    ]

    for task in experiment_tasks:
        output = task.get_output(
            shared_layers=shared_layers
        )
        model = models.Model(
            inputs=inputs,
            outputs=output
        )
        model.compile(
            optimizer=config.optimizer,
            loss="categorical_crossentropy"
        )
        task.model = model


def get_num_positions():
    return max(task.num_positions for task in experiment_tasks)


def make_shared_layers(word_input,
                       position1_input,
                       position2_input,
                       entity_marker_input):
    log.info("Building position embeddings")
    position_embedding = make_position_embedding(
        "shared_position_embedding"
    )
    position1_embedding = position_embedding(position1_input)
    position2_embedding = position_embedding(position2_input)
    word_embedding = make_word_embedding()(word_input)
    embedding_merge_layer = layers.concatenate(
        [word_embedding, position1_embedding, position2_embedding],
        name="embedding_merge"
    ) if not arguments.entity_markers else word_embedding
    log.info("Building convolution layers")
    convolution_layers = make_convolution_layers(embedding_merge_layer)
    if arguments.entity_markers:
        log.info("Using entity markers instead of position embeddings")
        return layers.concatenate(
            [convolution_layers, entity_marker_input],
            name="entity_marker_merge"
        )
    else:
        return convolution_layers


def make_convolution_layers(embedding_merge_layer):
    convolution_layers = []
    for n_gram in arguments.n_grams:
        convolution_layer = layers.Conv1D(
            kernel_size=n_gram,
            filters=arguments.filters,
            activation="relu",
            name="shared_convolution_" + str(n_gram) + "_gram"
        )(embedding_merge_layer)
        pooling_layer = layers.GlobalMaxPooling1D(
            name="pooling_" + str(n_gram) + "_gram",
        )(convolution_layer)
        convolution_layers.append(pooling_layer)
    convolution_merge_layer = layers.concatenate(
        convolution_layers,
        name="convolution_merge"
    )
    if arguments.dropout:
        log.info("Adding dropout layer")
        convolution_merge_layer = layers.Dropout(
            rate=.5
        )(convolution_merge_layer)
    for dense_count in range(1, arguments.shared_layer_depth + 1):
        convolution_merge_layer = layers.Dense(
            units=arguments.hidden_layer_dimension,
            activation="relu",
            name="shared_dense_" + str(dense_count)
        )(convolution_merge_layer)
    return convolution_merge_layer


