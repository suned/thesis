import numpy
from keras.preprocessing import sequence

from . import tokenization
from . import config


class BadTokenizationError(Exception):
    pass


def format_arguments(arguments):
    if arguments is None:
        return ""
    return "(" + arguments[0] + "," + arguments[1] + ")"


def trim(vector):
    excess = (len(vector) - config.max_len)
    if excess % 2 == 0:
        left = right = excess // 2
    else:
        left = (excess // 2) + 1
        right = (excess // 2)
    return vector[left:len(vector) - right]


def pad(vector):
    missing = (config.max_len - len(vector))
    if missing % 2 == 0:
        left = right = missing // 2
    else:
        left = (missing // 2) + 1
        right = missing // 2
    return numpy.pad(vector, pad_width=(left, right), mode="constant")


class GroundTruth:
    def __init__(
            self,
            sentence_id,
            sentence,
            e1_offset,
            e2_offset,
            relation,
            arguments):
        self.sentence_id = sentence_id
        self.sentence = tokenization.tokenize(sentence)
        self.e1 = self._offset_to_index(e1_offset)
        self.e2 = self._offset_to_index(e2_offset)
        self.relation = (
            relation
            + format_arguments(arguments)
        )

    def _offset_to_index(self, e1_offset):
        start_char, end_char = e1_offset
        start_index, end_index = self._find_token(start_char, end_char)
        if start_index is None or end_index is None:
            raise BadTokenizationError()
        return start_index, end_index

    def _find_token(self, start_char, end_char):
        start_index = end_index = None
        for i, token in enumerate(self.sentence):
            if token.idx == start_char:
                start_index = i
            if (token.idx + len(token.text) == end_char or
                                token.idx + len(
                                token.text) - 1 == end_char):
                end_index = i
                break
        return start_index, end_index

    def _ids_to_index(self, arguments):
        if arguments is None:
            return None
        first, _ = arguments
        if first == "e1":
            return self.e1, self.e2
        else:
            return self.e2, self.e1

    def feature_vector(self):
        vector = numpy.array([token.rank if token.has_vector
                              else tokenization.max_rank + 1
                              for token in self.sentence])
        if len(vector) < config.max_len:
            return pad(vector)
        return trim(vector)

    def position_vectors(self):
        e1_position_vector = []
        e2_position_vector = []
        for i in range(len(self.sentence)):
            e1_position_vector.append(abs(i - self.e1[0]))
            e2_position_vector.append(abs(i - self.e2[0]))
        e1_position_vector = numpy.array(e1_position_vector)
        e2_position_vector = numpy.array(e2_position_vector)
        if len(self.sentence) < config.max_len:
            return pad(e1_position_vector), pad(e2_position_vector)
        else:
            return trim(e1_position_vector), trim(e2_position_vector)

    def __str__(self):
        e1_start, e1_end = self.e1
        e2_start, e2_end = self.e2
        e1 = self.sentence[e1_start:e1_end + 1]
        e2 = self.sentence[e2_start:e2_end + 1]
        e1 = "<e1>" + str(e1) + "</e1> "
        e2 = "<e2>" + str(e2) + "</e2>"
        before_e1 = self.sentence[:e1_start].string
        between_es = self.sentence[e1_end + 1:e2_start].string
        after_e2 = self.sentence[e2_end + 1:]
        if not after_e2[0].is_punct:
            e2 += " "
        sentence = before_e1 + e1 + between_es + e2 + after_e2.string
        return self.relation + " : " + sentence

    def __repr__(self):
        return self.__str__()
