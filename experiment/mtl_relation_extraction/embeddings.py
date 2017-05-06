import numpy
from keras import layers

from ..io import arguments
from . import log, nlp, tasks


def make_word_embedding():
    log.info(
        "Building word embedding layer"
    )
    embeddings = get_embeddings(nlp.vocab)
    return layers.Embedding(
        input_dim=embeddings.shape[0],
        output_dim=embeddings.shape[1],
        trainable=True,
        weights=[embeddings],
        name="shared_word_embedding"
    )


def get_embeddings(vocab):
    vectors = numpy.random.rand(
        # indices:
        # padding: 0
        # out of vocab: nlp.vocab.length + 1
        # entity start: nlp.vocab.length + 2
        # entity end: nlp.vocab.length + 3
        nlp.vocab.length + 2,
        vocab.vectors_length
    ) / 100
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    log.info("Word embedding shape: %s", vectors.shape)
    return vectors


def make_position_embedding():
    return layers.Embedding(
        input_dim=tasks.longest_sentence * 2,
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
