import numpy
from keras import layers

from ..io import arguments
from . import log, nlp


def make_word_embedding():
    log.info(
        "Building word embedding layer"
    )
    embeddings = nlp.vocabulary.get_embeddings()
    return layers.Embedding(
        input_dim=embeddings.shape[0],
        output_dim=embeddings.shape[1],
        trainable=True,
        weights=[embeddings],
        name="shared_word_embedding"
    )


def make_position_embedding():
    return layers.Embedding(
        input_dim=nlp.longest_sentence * 2,
        output_dim=arguments.position_embedding_dimension,
        trainable=True,
        name="shared_position_embedding"
    )


shared_word_embedding = None
shared_position_embedding = None


def make_shared_embeddings():
    global shared_word_embedding
    global shared_position_embedding
    shared_word_embedding = make_word_embedding()
    shared_position_embedding = make_position_embedding()
