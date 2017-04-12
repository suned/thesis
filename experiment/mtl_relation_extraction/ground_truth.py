import numpy

from .io import arguments
from . import tokenization


class BadTokenizationError(Exception):
    pass


def format_arguments(relation_args):
    if relation_args is None:
        return ""
    return "(" + relation_args[0] + "," + relation_args[1] + ")"


def at_beginning(left):
    return left == 0


def beyond_edge(right, vector):
    return right > len(vector)


def trim(vector, e1):
    left, right = find_edges(e1, vector)
    return vector[left:right]


def find_edges(e1, vector):
    left = e1[0]
    right = left + arguments.max_len
    while (beyond_edge(right, vector)
           and not at_beginning(left)):
        left -= 1
        right -= 1
    return left, right


def pad(vector):
    missing = (arguments.max_len - len(vector))
    if missing % 2 == 0:
        left = right = missing // 2
    else:
        left = (missing // 2) + 1
        right = missing // 2
    return numpy.pad(vector, pad_width=(left, right), mode="constant")


def trim_position(position_vector, e1):
    left, right = find_edges(e1, position_vector)
    shifted = numpy.roll(position_vector, -left)
    return shifted[:arguments.max_len] + arguments.max_len


def entity_distance(i, entity):
    if entity[0] <= i < entity[1]:
        return 0
    if i < entity[0]:
        return i - entity[0]
    if i >= entity[1]:
        return i - entity[1] + 1


class GroundTruth:
    def __init__(
            self,
            sentence_id,
            sentence,
            e1_offset,
            e2_offset,
            relation,
            relation_args):
        self.sentence_id = sentence_id
        self.sentence = tokenization.tokenize(sentence)
        self.e1 = self._offset_to_index(e1_offset)
        self.e2 = self._offset_to_index(e2_offset)
        self.relation = (
            relation
            + format_arguments(relation_args)
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
        return start_index, end_index + 1

    def _ids_to_index(self, relation_args):
        if relation_args is None:
            return None
        first, _ = relation_args
        if first == "e1":
            return self.e1, self.e2
        else:
            return self.e2, self.e1

    def feature_vector(self):
        vector = numpy.array([token.rank if token.has_vector
                              else tokenization.max_rank + 1
                              for token in self.sentence])
        if len(vector) < arguments.max_len:
            return pad(vector)
        return trim(vector, self.first_entity())

    def first_entity(self):
        if self.e1[0] < self.e2[0]:
            return self.e1
        else:
            return self.e2

    def last_entity(self):
        if self.e2[1] > self.e1[1]:
            return self.e2
        else:
            return self.e1

    def position_vectors(self):
        e1_position_vector = []
        e2_position_vector = []
        for i in range(len(self.sentence)):
            e1_position_vector.append(entity_distance(i, self.e1))
            e2_position_vector.append(entity_distance(i, self.e2))
        e1_position_vector = numpy.array(e1_position_vector)
        e2_position_vector = numpy.array(e2_position_vector)
        if len(self.sentence) < arguments.max_len:
            return pad(e1_position_vector), pad(e2_position_vector)
        else:
            return (
                trim_position(e1_position_vector, self.first_entity()),
                trim_position(e2_position_vector, self.first_entity())
            )

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
