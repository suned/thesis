import numpy
import spacy

from .vocabulary import Vocabulary
from . import log
from ..io import word_vector_parser, arguments

_nlp = spacy.load(
    "en_core_web_md",
    parser=False,
    tagger=False,
    entity=False
)

pad_token = "#pad#"
vocabulary = Vocabulary()
pad_rank = 0
glove_vectors = None
longest_sentence = None


def load_glove_vectors():
    global glove_vectors
    log.info("Loading glove vectors")
    glove_vectors = word_vector_parser.from_pickle()


def tokenize(s):
    return _nlp.tokenizer(s)


def add_validation_vocabulary(task):
    count = 0
    for word in task.get_validation_vocabulary():
        if word in glove_vectors:
            vector = glove_vectors[word]
            vocabulary.add(word, vector)
            count += 1
    log.info("Added %i glove vectors", count)


def add_vocabularies(tasks):
    for task in tasks:
        log.info("Adding vocabulary from task: %s", task.name)
        log.info("Vocabulary length before: %i", vocabulary.length)
        add_train_vocabulary(task)
        add_validation_vocabulary(task)
        log.info("Vocabulary after: %i", vocabulary.length)


def add_train_vocabulary(task):
    glove_vector_count = 0
    for word in task.get_vocabulary():
        vector = (
            glove_vectors[word] if word in glove_vectors
            else numpy.random.rand(
                arguments.word_embedding_dimension
            )
        )
        if word in glove_vectors:
            glove_vector_count += 1
        vocabulary.add(word, vector)
    log.info(
        "Added %i glove vectors, %i random vectors",
        glove_vector_count,
        glove_vector_count - len(task.get_validation_vocabulary())
    )
