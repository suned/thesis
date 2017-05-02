import numpy
import spacy
from . import log
from ..io import word_vector_parser

_nlp = spacy.load(
    "en_core_web_md",
    parser=False,
    tagger=False,
    entity=False
)

vocab = _nlp.vocab


def tokenize(s):
    return _nlp.tokenizer(s)


def add_all(train_relations):
    log.info("Adding relation words to vocabulary")

    for relation in train_relations:
        for token in relation.sentence:
            if token.string not in _nlp.vocab:
                _ = _nlp.vocab[token.string]
    log.info("Vocabulary length after: %i", _nlp.vocab.length)


def add_test_vocabulary(glove_vectors, task):
    glove_or_nothing(glove_vectors, task.validation_relations)
    glove_or_nothing(glove_vectors, task.early_stopping_relations)
    glove_or_nothing(glove_vectors, task.test_relations)


def glove_or_nothing(glove_vectors, relations):
    for relation in relations:
        for token in relation.sentence:
            if token.string not in _nlp.vocab:
                lex = _nlp.vocab[token.string]
                if token.string in glove_vectors:
                    lex.vector = glove_vectors[token.string]
                else:
                    assert not lex.has_vector


def add_vocabularies(tasks, vector_length=300):
    log.info("Loading glove vectors")
    glove_vectors = word_vector_parser.from_pickle()
    for task in tasks:
        log.info("Adding vocabulary from task: %s", task.name)
        log.info("Vocabulary length before: %i", _nlp.vocab.length)
        add_train_vocabulary(glove_vectors, task, vector_length)
        if task.is_target:
            add_test_vocabulary(glove_vectors, task)
        log.info("Vocabulary after: %i", _nlp.vocab.length)


def add_train_vocabulary(glove_vectors, task, vector_length):
    for relation in task.train_relations:
        for token in relation.sentence:
            if token.string not in _nlp.vocab:
                lex = _nlp.vocab[token.string]
                vector = (glove_vectors[token.string]
                          if token.string in glove_vectors
                          else numpy.random.rand(vector_length))
                lex.vector = vector
