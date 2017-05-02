import numpy
from keras.engine import Layer
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn import model_selection, preprocessing

from .. import config
from ..io import arguments
from . import log


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


def get_features(relations):
    return numpy.array(
        [train_relation.feature_vector()
         for train_relation in relations]
    )


def get_labels(relations):
    return numpy.array(
        [train_relation.relation for train_relation in relations]
    )


def get_entity_markers(relations):
    return numpy.array(
        [relation.entity_markers() for relation in relations]
    )


def get_positions(train_relations):
    position1 = []
    position2 = []
    for train_relation in train_relations:
        (position1_vector,
         position2_vector) = train_relation.position_vectors()
        position1.append(position1_vector)
        position2.append(position2_vector)
    return numpy.array(position1), numpy.array(position2)


def make_input(features,
               position1_vectors,
               position2_vectors,
               entity_markers):
    return {
        "word_input": features,
        "position1_input": position1_vectors,
        "position2_input": position2_vectors
    } if not arguments.entity_markers else {
        "word_input": features,
        "entity_marker_input": entity_markers.reshape(
            entity_markers.shape + (1,)
        )
    }


class Task:
    def __init__(self):
        self.is_target = False
        self.name = None
        self.train_relations = None
        self.early_stopping_relations = None
        self.train_labels = None
        self.early_stopping_labels = None
        self.encoder = None
        self.num_classes = None

    def __repr__(self):
        return self.name

    def load(self):
        raise NotImplementedError("Must be implemented in sub-class.")

    def init_encoder(self):
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(self.train_labels)
        self.num_classes = len(self.encoder.classes_)

    def format_set(self, labels, relations):
        features = get_features(relations)
        position1, position2 = get_positions(relations)
        entity_markers = get_entity_markers(relations)
        inputs = make_input(
            features,
            position1,
            position2,
            entity_markers
        )
        output = {
            self.name + "_output": self.encode(labels)
        }
        return inputs, output

    def training_set(self):
        return self.format_set(
            self.train_labels,
            self.train_relations
        )

    def early_stopping_set(self):
        if self.early_stopping_relations is None:
            raise ValueError(
                "Early stopping relations not initialised in task %s" %
                self.name

            )
        return self.format_set(
            self.early_stopping_labels,
            self.early_stopping_relations
        )

    def get_batch(self):
        n = len(self.train_relations)

        batch_indices = numpy.random.randint(
            0,
            high=n,
            size=arguments.batch_size
        )
        batch_relations = self.train_relations[batch_indices]
        batch_labels = self.train_labels[batch_indices]
        return self.format_set(batch_labels, batch_relations)

    def encode(self, labels):
        integer_labels = self.encoder.transform(labels)
        return to_categorical(integer_labels, self.num_classes)

    def decode(self, labels):
        integer_labels = labels.argmax(axis=1)
        return self.encoder.inverse_transform(integer_labels)

    def get_output(self, shared_layers: Layer):
        log.info("Adding %s output", self.name)
        for i in range(1, arguments.task_layer_depth + 1):
            shared_layers = Dense(
                arguments.hidden_layer_dimension,
                activation="relu",
                name=self.name + "_hidden_" + str(i)
            )(shared_layers)
        output_layer = Dense(
            self.num_classes,
            activation="softmax",
            name=self.name + "_output"
        )(shared_layers)
        return output_layer
