import spacy
from . import tokenization


class BadTokenizationError(Exception):
    pass


class BadOffsetError(Exception):
    pass


class GroundTruth:
    def __init__(
            self,
            sentence_id,
            sentence,
            e1_offset,
            e2_offset,
            relation,
            direction):
        self.sentence_id = sentence_id
        self.sentence = tokenization.tokenize(sentence)
        self.e1_index = self._offset_to_index(e1_offset)
        self.e2_index = self._offset_to_index(e2_offset)
        self.relation = relation
        self.direction = self._ids_to_index(direction)

    def _offset_to_index(self, e1_offset):
        start, end = e1_offset
        for i, token in enumerate(self.sentence):
            if token.idx == start:
                return i
        return BadOffsetError()

    def _ids_to_index(self, direction):
        if direction is None:
            return None
        if len(direction) > 2:
            import ipdb
            ipdb.sset_trace()
        first, _ = direction
        if first == "e1":
            return self.e1_index, self.e2_index
        else:
            return self.e2_index, self.e1_index
