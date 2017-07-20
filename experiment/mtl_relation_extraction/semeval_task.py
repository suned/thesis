import os

import numpy

from ..io import arguments
from sklearn import metrics

from .. import config
from ..io import semeval_parser
from . import preprocessing
from .task import (
    get_labels,
    split
)
from .cnn import CNN


class SemEvalTask(CNN):

    def reduce_train_data(self, fraction):
        n = len(self.train_relations)
        reduced_n = int(n * fraction) if fraction != 0. else 0
        indices = numpy.random.randint(
            0,
            high=n,
            size=reduced_n
        )
        self.train_relations = self.train_relations[indices]
        self.train_labels = self.train_labels[indices]

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

    def __init__(self):
        super().__init__(
            name="SemEval",
            is_target=True
        )
        self.train_relations = None
        self.validation_relations = None
        self.early_stopping_relations = None

        self.early_stopping_labels = None
        self.validation_labels = None
        self.train_labels = None

    def split(self, train_indices, test_indices):
        self.validation_relations = self.relations[test_indices]
        self.validation_labels = self.labels[test_indices]

        train_relations = self.relations[train_indices]
        (train_relations,
         early_stopping_relations) = split(
            train_relations,
            test_ratio=arguments.early_stopping_ratio
        )
        self.train_relations = train_relations
        self.train_labels = get_labels(train_relations)

        self.early_stopping_relations = early_stopping_relations
        self.early_stopping_labels = get_labels(
            early_stopping_relations
        )

    def load(self):
        train_path = os.path.join(
            arguments.data_path,
            config.semeval_train_path
        )
        train_relations = semeval_parser.read_file(train_path)
        test_path = os.path.join(
            arguments.data_path,
            config.semeval_test_path
        )
        test_relations = semeval_parser.read_file(test_path)
        self.relations = numpy.append(
            train_relations,
            test_relations
        )
        self.input_length = preprocessing.max_distance(
            self.relations
        )
        self.labels = get_labels(self.relations)
        self.init_encoder()

    def validation_set(self):
        return self.format_set(
            self.validation_labels,
            self.validation_relations
        )

    def validation_metrics(self):
        validation_input, validation_labels = self.validation_set()
        one_hot_prediction = self.model.predict(validation_input)
        label_prediction = self.decode(one_hot_prediction)
        validation_metrics = {
            "f1": metrics.f1_score(
                self.validation_labels,
                label_prediction,
                average="macro"
            ),
            "precision": metrics.precision_score(
                self.validation_labels,
                label_prediction,
                average="macro"
            ),
            "recall": metrics.recall_score(
                self.validation_labels,
                label_prediction,
                average="macro"
            )
        }
        return validation_metrics

    def validation_report(self):
        validation_input, _ = self.validation_set()
        one_hot_y = self.model.predict(validation_input)
        pred_y = self.decode(one_hot_y)
        report = metrics.classification_report(
            self.validation_labels,
            pred_y
        )
        return report
