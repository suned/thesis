import numpy
from keras.engine import Layer
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn import model_selection, preprocessing

from mtl_relation_extraction import config, log
from mtl_relation_extraction.preprocessing import longest_sentence


def split(relations, test_ratio):
    iterator = model_selection.ShuffleSplit(
            n_splits=1,
            random_state=config.random_state,
            test_size=test_ratio
    )
    train_indices, test_indices = next(
        iterator.split(relations)
    )
    return relations[train_indices], relations[test_indices]


def get_features(max_len, relations):
    return numpy.array(
        [train_relation.feature_vector(max_len)
         for train_relation in relations]
    )


def get_labels(relations):
    return numpy.array(
        [train_relation.relation for train_relation in relations]
    )


def get_positions(max_len, train_relations):
    position1 = []
    position2 = []
    for train_relation in train_relations:
        (position1_vector,
         position2_vector) = train_relation.position_vectors(max_len)
        position1.append(position1_vector)
        position2.append(position2_vector)
    return numpy.array(position1), numpy.array(position2)


def make_input(features,
               position1_vectors,
               position2_vectors):
    return {
        "word_input": features,
        "position1_input": position1_vectors,
        "position2_input": position2_vectors
    }


class Task:
    def __init__(self):
        self.is_target = False
        self.name = None
        self.train_relations = None
        self.train_features = None
        self.train_position1_vectors = None
        self.train_position2_vectors = None
        self.train_labels = None
        self.encoder = None
        self.num_classes = None
        self.num_positions = None

    def __repr__(self):
        return self.name

    def load(self):
        raise NotImplementedError("Must be implemented in sub-class.")

    def init_encoder(self):
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(self.train_labels)
        self.num_classes = len(self.encoder.classes_)

    def get_batch(self):
        n = len(self.train_features)
        batch_indices = numpy.random.randint(
            0,
            high=n,
            size=config.batch_size
        )
        batch_features = self.train_features[batch_indices]
        batch_position1 = self.train_position1_vectors[batch_indices]
        batch_position2 = self.train_position2_vectors[batch_indices]
        batch_labels = self.train_labels[batch_indices]
        return (
            make_input(
                batch_features,
                batch_position1,
                batch_position2
            ),
            {
                self.name + "_output": self.encode(batch_labels)
            }
        )

    def encode(self, labels):
        integer_labels = self.encoder.transform(labels)
        return to_categorical(integer_labels, self.num_classes)

    def decode(self, labels):
        integer_labels = labels.argmax(axis=1)
        return self.encoder.inverse_transform(integer_labels)

    def get_output(self, shared_layers: Layer):
        log.info("Adding %s output", self.name)
        output_layer = Dense(
            self.num_classes,
            activation="softmax",
            name=self.name + "_output"
        )(shared_layers)
        return output_layer

    def init_num_positions(self):
        self.num_positions = longest_sentence(self.train_relations)