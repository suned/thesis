import numpy
from keras import layers, models

from .. import config
from ..io import arguments
from . import log
from . import tokenization
from .tasks import load_tasks, target_task, experiment_tasks

log_header = """Epoch\t\tTask\t\tTraining Loss\t\tEarly Stopping Loss
==========================================================================="""
log_line = """%i\t\t%s\t\t%f\t\t%f %s"""


def make_word_embedding():
    log.info(
        "Building %s word embedding layer",
        "trainable" if not arguments.freeze_embeddings
        else "un-trainable"
    )
    embeddings = get_embeddings(tokenization.nlp.vocab)
    return layers.Embedding(
        input_dim=embeddings.shape[0],
        output_dim=embeddings.shape[1],
        trainable=not arguments.freeze_embeddings,
        weights=[embeddings],
        name="shared_word_embedding"
    )


def compile_models():
    log.info("Compiling models")
    word_input, position1_input, position2_input = make_inputs()
    shared_layers = make_shared_layers(
        word_input,
        position1_input,
        position2_input
    )
    inputs = [
        word_input,
        position1_input,
        position2_input
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


def make_shared_layers(position1_input, position2_input, word_input):
    log.info("Building position embeddings")
    position_embedding = make_position_embedding(
        "shared_position_embedding"
    )
    position1_embedding = position_embedding(position1_input)
    position2_embedding = position_embedding(position2_input)
    word_embedding = make_word_embedding()(word_input)
    embedding_merge_layer = layers.concatenate(
        [word_embedding, position1_embedding, position2_embedding]
    )
    log.info("Building convolution layers")
    return make_convolution_layers(embedding_merge_layer)


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
    convolution_merge_layer = layers.concatenate(convolution_layers)
    if arguments.dropout:
        log.info("Adding dropout layer")
        convolution_merge_layer = layers.Dropout(
            rate=.5
        )(convolution_merge_layer)
    return convolution_merge_layer


def make_inputs():
    word_input = layers.Input(
        (arguments.max_len,),
        dtype="int32",
        name="word_input"
    )
    position1_input = layers.Input(
        (arguments.max_len,),
        dtype="int32",
        name="position1_input"
    )
    position2_input = layers.Input(
        (arguments.max_len,),
        dtype="int32",
        name="position2_input"
    )
    return position1_input, position2_input, word_input


def make_position_embedding(name):
    return layers.Embedding(
        input_dim=2 * arguments.max_len,
        output_dim=arguments.position_embedding_dimension,
        trainable=True,
        name=name
    )


def get_embeddings(vocab):
    vectors = numpy.random.rand(
        tokenization.max_rank + 2,
        vocab.vectors_length
    ) / 100
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors


def fit():
    best_validation_loss = float("inf")
    best_weights = None
    log.info(
        "Training model with %i training samples, "
        "%i early stopping samples, sentence length %i",
        len(target_task.train_relations),
        len(target_task.early_stopping_relations),
        arguments.max_len
    )
    epochs_without_improvement = 0
    early_stopping_set = target_task.early_stopping_set()
    task_count = len(experiment_tasks)
    log.info(log_header)
    for epoch in range(1, arguments.epochs + 1):
        task = experiment_tasks[epoch % task_count]
        batch_input, batch_labels = task.get_batch()
        epoch_stats = task.model.fit(
            batch_input,
            batch_labels,
            verbose=config.keras_verbosity,
            validation_data=early_stopping_set if task.is_target else None
        )
        training_loss = epoch_stats.history["loss"][0]
        validation_loss = (epoch_stats.history["val_loss"][0]
                           if task.is_target else float("nan"))
        if task.is_target and validation_loss < best_validation_loss:
            optimum = "*"
            best_validation_loss = validation_loss
            best_weights = target_task.model.get_weights()
            epochs_without_improvement = 0
        else:
            optimum = ""
            epochs_without_improvement += 1 if task.is_target else 0
        log.info(
            log_line,
            epoch,
            task.name,
            training_loss,
            validation_loss,
            optimum
        )
        if task.is_target and training_loss < .01:
            log.info("Training F1 maximised. Stopping")
            break
        if epochs_without_improvement > arguments.patience:
            log.info("Patience exceeded. Stopping")
            break
    log.info(
        "Finished training with best loss: %f",
        best_validation_loss
    )
    target_task.model.set_weights(best_weights)
    log.info("Validation F1: %f", target_task.validation_f1())


def train():
    load_tasks()
    compile_models()
    fit()
