from keras.models import Model
from keras.layers import Layer, Dense
import os
from sklearn import (model_selection, preprocessing, metrics)
import numpy
from typing import List

from . import config
from . import log
from .io import arguments, semeval_parser
from .preprocessing import max_sentence_length, longest_sentence
from .pipeline import Pipeline


def split(features):
    iterator = model_selection.ShuffleSplit(
            n_splits=1,
            random_state=config.random_state,
            test_size=.2
    )
    return next(
        iterator.split(features)
    )


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
        self.name = None
        self.train_relations = None
        self.train_features = None
        self.train_position1_vectors = None
        self.train_position2_vectors = None
        self.train_labels = None
        self.encoder = None
        self.num_classes = None
        self.pipeline = None
        self.num_positions = None

    def load(self):
        raise NotImplementedError("Must be implemented in sub-class.")

    def compile_model(self, shared_layers: Model, inputs: List[Layer]):
        raise NotImplementedError("Must be implement in sub-class.")

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
                self.name + "_output": batch_labels
            }
        )


class TargetTask(Task):
    def __init__(self):
        super().__init__()
        self.name = "SemEval"
        self.model = None

        self.test_relations = None
        self.validation_relations = None

        self.test_features = None
        self.validation_features = None

        self.test_position1_vectors = None
        self.validation_position1_vectors = None

        self.test_position2_vectors = None
        self.validation_position2_vectors = None

        self.test_labels = None
        self.validation_labels = None

        self.max_length = None

    def load(self):
        log.info("Loading %s task", self.name)
        self.load_training_set()
        self.load_test_set()
        self.init_encoder()

    def validation_set(self):
        validation_input = make_input(
            self.validation_features,
            self.validation_position1_vectors,
            self.validation_position2_vectors
        )
        validation_labels = {
            self.name + "_output": self.validation_labels
        }
        return validation_input, validation_labels

    def compile_model(self, shared_layers: Layer, inputs: List[Layer]):
        log.info("Adding %s output", self.name)
        output_layer = Dense(
            self.num_classes,
            activation="softmax",
            name=self.name + "_output"
        )(shared_layers)
        self.model = Model(
            inputs=inputs,
            outputs=output_layer
        )
        self.model.compile(
            optimizer=config.optimizer,
            loss="categorical_crossentropy"
        )
        self.pipeline = Pipeline(self.encoder, self.model)

    def validation_f1(self):
        validation_input = make_input(
            self.validation_features,
            self.validation_position1_vectors,
            self.validation_position2_vectors
        )
        predicted_labels = self.pipeline.predict(validation_input)
        return metrics.f1_score(
            predicted_labels,
            self.validation_labels,
            average="micro"
        )

    def training_f1(self):
        training_input = make_input(
            self.train_features,
            self.train_position1_vectors,
            self.train_position2_vectors
        )
        predicted_labels = self.pipeline.predict(training_input)
        return metrics.f1_score(
            predicted_labels,
            self.train_labels,
            average="micro"
        )

    def validation_loss(self):
        validation_input = make_input(
            self.validation_features,
            self.validation_position1_vectors,
            self.validation_position2_vectors
        )
        validation_labels = {
            self.name + "_output": self.validation_labels
        }
        return self.pipeline.evaluate(
            validation_input,
            validation_labels,
            verbose=0
        )

    def training_loss(self):
        training_input = make_input(
            self.train_features,
            self.train_position1_vectors,
            self.train_position2_vectors
        )
        training_labels = {
            self.name + "_output": self.train_labels
        }
        return self.pipeline.evaluate(
            training_input,
            training_labels,
            verbose=0
        )

    def load_training_set(self):
        train_path = os.path.join(
            arguments.data_path,
            config.semeval_train_path
        )
        train_relations = semeval_parser.read_file(train_path)

        self.max_length = max_sentence_length(train_relations)
        self.num_positions = longest_sentence(train_relations)

        train_features = get_features(self.max_length, train_relations)
        position1_vectors, position2_vectors = get_positions(
            self.max_length,
            train_relations
        )
        train_labels = get_labels(train_relations)
        train_indices, validation_indices = split(train_features)
        self.train_relations = train_relations[train_indices]
        self.validation_relations = train_relations[validation_indices]
        self.train_features = train_features[train_indices]
        self.validation_features = train_features[validation_indices]
        self.train_position1_vectors = position1_vectors[
            train_indices
        ]
        self.validation_position1_vectors = position1_vectors[
            validation_indices
        ]
        self.train_position2_vectors = position2_vectors[
            train_indices
        ]
        self.validation_position2_vectors = position2_vectors[
            validation_indices
        ]
        self.train_labels = train_labels[train_indices]
        self.validation_labels = train_labels[validation_indices]

    def load_test_set(self):
        test_path = os.path.join(
            arguments.data_path,
            config.semeval_test_path
        )
        self.test_relations = semeval_parser.read_file(test_path)
        self.test_features = get_features(
            self.max_length,
            self.test_relations
        )
        self.test_labels = get_labels(self.test_relations)
        (self.test_position1_vectors,
         self.test_position2_vectors) = get_positions(
            self.max_length,
            self.test_relations
        )

target_task = TargetTask()
auxiliary_tasks = []


def load_tasks():
    target_task.load()
    for auxiliary_task in auxiliary_tasks:
        auxiliary_task.load()
