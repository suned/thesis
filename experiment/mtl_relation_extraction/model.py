import numpy
import spacy
from keras import layers
from mtl_relation_extraction import log, hyperparameters, config
from .tasks import load_tasks, target_task

nlp = spacy.load("en")


def make_word_embedding():
    log.info("Building word embedding layer")
    embeddings = get_embeddings(nlp.vocab)
    return layers.Embedding(
        input_dim=embeddings.shape[0],
        output_dim=embeddings.shape[1],
        trainable=True,
        weights=[embeddings],
        name="word_embedding"
    )


def compile_models():
    position1_input, position2_input, word_input = make_inputs()
    shared_layers = make_shared_layers(
        position1_input,
        position2_input,
        word_input
    )
    log.info("Compiling models")
    target_task.compile_model(
        shared_layers=shared_layers,
        inputs=[
            word_input,
            position1_input,
            position2_input
        ]
    )


def make_shared_layers(position1_input, position2_input, word_input):
    log.info("Building position embeddings")
    position1_embedding = make_position_embedding(
        target_task.num_positions,
        "position1_embedding"
    )(position1_input)
    position2_embedding = make_position_embedding(
        target_task.num_positions,
        "position2_embedding"
    )(position2_input)
    word_embedding = make_word_embedding()(word_input)
    embedding_merge_layer = layers.concatenate(
        [word_embedding, position1_embedding, position2_embedding]
    )
    log.info("Building convolution layers")
    return make_convolution_layers(embedding_merge_layer)


def make_convolution_layers(embedding_merge_layer):
    convolution_layers = []
    for n_gram in hyperparameters.n_grams:
        convolution_layer = layers.Conv1D(
            kernel_size=n_gram,
            filters=hyperparameters.filters,
            activation="relu",
            name="convolution_" + str(n_gram)
        )(embedding_merge_layer)
        pooling_layer = layers.GlobalMaxPooling1D(
            name="pooling_" + str(n_gram),
        )(convolution_layer)
        convolution_layers.append(pooling_layer)
    convolution_merge_layer = layers.concatenate(convolution_layers)
    return convolution_merge_layer


def make_inputs():
    word_input = layers.Input(
        (target_task.max_length,),
        dtype="int32",
        name="word_input"
    )
    position1_input = layers.Input(
        (target_task.max_length,),
        dtype="int32",
        name="position1_input"
    )
    position2_input = layers.Input(
        (target_task.max_length,),
        dtype="int32",
        name="position2_input"
    )
    return position1_input, position2_input, word_input


def make_position_embedding(num_positions, name):
    return layers.Embedding(
        input_dim=num_positions,
        output_dim=hyperparameters.position_embedding_dimension,
        trainable=True,
        name=name
    )


def get_embeddings(vocab):
    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    vectors = numpy.ndarray(
        (max_rank + 1, vocab.vectors_length),
        dtype='float32'
    )
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors


def fit():
    best_validation_f1 = float("-inf")
    best_weights = None
    log.info(
        "Training model with %i training samples, %i validation samples",
        len(target_task.train_relations),
        len(target_task.validation_relations)
    )
    for epoch in range(1, config.epochs):
        batch_input, batch_labels = target_task.get_batch()
        epoch_stats = target_task.pipeline.batch_fit(
            batch_input,
            batch_labels,
            validation_data=target_task.validation_set()
        )
        training_f1 = epoch_stats.history["loss"][0]
        validation_f1 = epoch_stats.history["val_loss"][0]

        if validation_f1 > best_validation_f1:
            optimum = "*"
            best_validation_f1 = validation_f1
            best_weights = target_task.model.get_weights()
        else:
            optimum = ""
        log.info(
            "Epoch %i \t: Training F1: %f | Validation F1: %f %s",
            epoch,
            training_f1,
            validation_f1,
            optimum
        )
        if training_f1 >.99:
            log.info("Training F1 maximised. Stopping")
            break
    log.info("Finished training with best f1: %i", best_validation_f1)
    target_task.model.set_weights(best_weights)


def train():
    load_tasks()
    compile_models()
    fit()
