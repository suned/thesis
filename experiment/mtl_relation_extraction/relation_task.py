import numpy
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from ..io import arguments
from .task import Task, beyond_edge, \
    at_beginning, make_input
from . import nlp


class RelationTask(Task):

    def longest_sentence(self):
        return max(len(relation.sentence)
                   for relation in self.relations)

    def get_vocabulary(self):
        return set([token.string for relation in self.relations
                for token in relation.sentence])

    def __init__(self, is_target, name):
        super().__init__(name, is_target)
        self.relations = None
        self.labels = None
        self.input_length = None

    def load(self):
        raise NotImplementedError("Must be implemented in sub-class.")

    def init_encoder(self):
        self.encoder = LabelEncoder()
        self.encoder.fit(self.labels)
        self.num_classes = len(self.encoder.classes_)

    def pad(self, vector):
        missing = (self.input_length - len(vector))
        if missing % 2 == 0:
            left = right = missing // 2
        else:
            left = (missing // 2) + 1
            right = missing // 2
        return numpy.pad(
            vector,
            pad_width=(left, right),
            mode="constant",
            constant_values=nlp.pad_rank
        )

    def find_edges(self, e1, vector):
        left = e1[0]
        right = left + self.input_length
        while (beyond_edge(right, vector)
               and not at_beginning(left)):
            left -= 1
            right -= 1
        return left, right

    def trim(self, vector, e1):
        left, right = self.find_edges(e1, vector)
        return vector[left:right]

    def get_features(self, relations):
        vectors = []
        for relation in relations:
            vector = relation.feature_vector()
            if numpy.any(vector > nlp.vocabulary.length):
                import ipdb
                ipdb.sset_trace()
            if len(vector) <= self.input_length:
                vector = self.pad(vector)
            else:
                vector = self.trim(vector, relation.first_entity())
            vectors.append(vector)
        return numpy.array(vectors)

    def get_positions(self, relations):
        position1 = []
        position2 = []
        for relation in relations:
            (position1_vector,
             position2_vector) = relation.position_vectors()
            if len(position1_vector) <= self.input_length:
                position1_vector = self.pad(
                    position1_vector
                )
                position2_vector = self.pad(
                    position2_vector
                )
            else:
                position1_vector = self.trim(
                    position1_vector,
                    relation.first_entity()
                )
                position2_vector = self.trim(
                    position2_vector,
                    relation.first_entity()
                )
            position1.append(position1_vector)
            position2.append(position2_vector)
        return (numpy.array(position1) + nlp.longest_sentence,
                numpy.array(position2) + nlp.longest_sentence)

    def format_set(self, labels, relations):
        features = self.get_features(relations)
        position1, position2 = self.get_positions(relations)
        input_data = make_input(
            features,
            position1,
            position2,
        )
        labels = {
            self.output_name: self.encode(labels)
        }
        return input_data, labels

    def training_set(self):
        return self.format_set(
            self.labels,
            self.relations
        )

    def get_batch(self, size=arguments.batch_size):
        n = len(self.relations)

        batch_indices = numpy.random.randint(
            0,
            high=n,
            size=size
        )
        batch_relations = self.relations[batch_indices]
        batch_labels = self.labels[batch_indices]
        return self.format_set(batch_labels, batch_relations)

    def encode(self, labels):
        integer_labels = self.encoder.transform(labels)
        return to_categorical(integer_labels, self.num_classes)

    def decode(self, labels):
        integer_labels = labels.argmax(axis=1)
        return self.encoder.inverse_transform(integer_labels)

    def get_output(self):
        output_layer = Dense(
            self.num_classes,
            activation="softmax",
            name=self.output_name
        )
        return output_layer

    def compile_model(self):
        raise NotImplementedError
