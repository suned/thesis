import numpy
from keras import layers

from ..io import arguments
from . import log, nlp


def make_word_embedding():
    log.info(
        "Building %s word embedding layer",
        "trainable" if not arguments.freeze_embeddings
        else "un-trainable"
    )
    embeddings = get_embeddings(nlp.vocab)
    return layers.Embedding(
        input_dim=embeddings.shape[0],
        output_dim=embeddings.shape[1],
            trainable=not arguments.freeze_embeddings,
        weights=[embeddings],
        name="shared_word_embedding"
    )


def get_embeddings(vocab):
    vectors = numpy.random.rand(
        nlp.vocab.length + 2,
        vocab.vectors_length
    ) / 100
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    log.info("Word embedding shape: %s", vectors.shape)
    return vectors


def make_position_embedding(name):
    return layers.Embedding(
        input_dim=2 * arguments.max_len,
        output_dim=arguments.position_embedding_dimension,
        trainable=True,
        name=name
    )