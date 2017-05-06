import numpy
import spacy
from . import log
from ..io import word_vector_parser
from .. import config

_nlp = spacy.load(
    "en_core_web_md",
    parser=False,
    tagger=False,
    entity=False
)

vocab = _nlp.vocab
glove_vectors = None


def load_glove_vectors():
    global glove_vectors
    log.info("Loading glove vectors")
    glove_vectors = word_vector_parser.from_pickle()


def tokenize(s):
    return _nlp.tokenizer(s)


def add_all(train_relations):
    log.info("Adding relation words to vocabulary")

    for relation in train_relations:
        for token in relation.sentence:
            if token.string not in _nlp.vocab:
                _ = _nlp.vocab[token.string]
    log.info("Vocabulary length after: %i", _nlp.vocab.length)


def add_test_vocabulary(task):
    glove_or_nothing(task.validation_relations)
    glove_or_nothing(task.early_stopping_relations)
    glove_or_nothing(task.test_relations)


def glove_or_nothing(relations):
    for relation in relations:
        for token in relation.sentence:
            if token.string not in _nlp.vocab:
                lex = _nlp.vocab[token.string]
                if token.string in glove_vectors:
                    lex.vector = glove_vectors[token.string]
                else:
                    assert not lex.has_vector


def add_vocabularies(tasks, vector_length=300):
    for task in tasks:
        log.info("Adding vocabulary from task: %s", task.name)
        log.info("Vocabulary length before: %i", _nlp.vocab.length)
        add_train_vocabulary(task, vector_length)
        if task.is_target:
            add_test_vocabulary(task)
        log.info("Vocabulary after: %i", _nlp.vocab.length)


def add_train_vocabulary(task, vector_length):
    _ = vocab[config.pad_token]
    for relation in task.train_relations:
        for token in relation.sentence:
            if token.string not in _nlp.vocab:
                lex = _nlp.vocab[token.string]
                vector = (glove_vectors[token.string]
                          if token.string in glove_vectors
                          else numpy.random.rand(vector_length))
                lex.vector = vector
