from funcparserlib import lexer, parser
import numpy
import unicodedata
import pickle

from .macos_file import pickle_dump, pickle_load
from .. import config


def tokenize(s):
    def choice(choices):
        return r"|".join(choices)

    specs = [
        ("Word", (r"^.+?(?= (\d|\-))",)),
        ("Space", (r' ',)),
        ("Newline", (r"\n",)),
        ("Number", (r"\-?\d+((\.\d+(e\-?\d+)?)|(e\-?\d+))?",))
    ]
    useless = ["Space"]
    tokenizer = lexer.make_tokenizer(specs)
    return [x for x in tokenizer(s)
            if x.type not in useless]


class InconsistentVectorFileError(Exception):
    pass


def parse(s, vector_length=300):
    def tokval(x):
        return x.value

    def toktype(t):
        return parser.some(lambda x: x.type == t) >> tokval

    def make_vector(numbers):
        v = numpy.array(numbers)
        if len(v) != vector_length:
            raise InconsistentVectorFileError()
        return v

    def decode(w):
        return unicodedata.normalize("NFKC", w)

    number = toktype("Number") >> float
    word = toktype("Word") >> decode
    newline = parser.skip(toktype("Newline"))
    vector = parser.many(number) >> make_vector

    line = word + vector + newline >> tuple
    lines = parser.many(line)
    tokens = tokenize(s)
    return lines.parse(tokens)


def parse_file(path, vector_length=300):
    vectors = []
    with open(path) as vector_file:
        for line_number, line in enumerate(vector_file):
            try:
                vector = parse(line, vector_length)
                if len(vector) == 0:
                    raise parser.NoParseError(
                        "In line: %s" % line_number
                    )
                vectors.extend(vector)
            except InconsistentVectorFileError as e:
                import ipdb
                ipdb.sset_trace()
                raise InconsistentVectorFileError(
                    "In line: %s" % line_number
                )
    return dict(vectors)


def to_pickle(path, vector_length=300):
    word_vectors = parse_file(path, vector_length)
    pickle_dump(word_vectors, config.word_vector_path)


def from_pickle():
    return pickle_load(config.word_vector_path)
