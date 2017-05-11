import numpy

from . import nlp


class BadTokenizationError(Exception):
    pass


def format_arguments(relation_args):
    if relation_args is None:
        return ""
    return "(" + relation_args[0] + "," + relation_args[1] + ")"


def entity_distance(i, entity):
    if entity[0] <= i < entity[1]:
        return 1
    if i < entity[0]:
        return i - entity[0]
    if i >= entity[1]:
        return i - entity[1] + 2


class Relation:
    def __init__(
            self,
            sentence_id,
            sentence,
            e1_offset,
            e2_offset,
            relation,
            relation_args):
        self.sentence_id = sentence_id
        self.sentence = nlp.tokenize(sentence)
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
        return start_index, end_index + 1

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

    def _ids_to_index(self, relation_args):
        if relation_args is None:
            return None
        first, _ = relation_args
        if first == "e1":
            return self.e1, self.e2
        else:
            return self.e2, self.e1

    def feature_vector(self):
        return numpy.array([nlp.vocabulary[str(token)]
                            for token in self.sentence])

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
        return e1_position_vector, e2_position_vector

    def at_entity(self, i):
        return (
            entity_distance(i, self.e1) == 0
            or entity_distance(i, self.e2) == 0
        )


class Sequence:
    def __init__(self, sentence, tags):
        if type(sentence) == str:
            self.sentence = nlp.tokenize(sentence)
        else:
            self.sentence = sentence
        if len(self.sentence) != len(tags):
            raise BadTokenizationError()
        self.tags = tags

    def get_words(self):
        if type(self.sentence) == list:
            return self.sentence
        return [token.string for token in self.sentence]

    def feature_vector(self):
        return numpy.array([nlp.vocabulary[str(token)]
                            for token in self.sentence])
