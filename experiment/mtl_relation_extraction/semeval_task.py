import os

import numpy

from ..io import arguments
from sklearn import metrics

from .. import config
from ..io import semeval_parser
from . import log
from . import preprocessing
from .task import (
    get_labels,
    split
)
from .cnn import CNN
from .relation_task import get_vocabulary


class SemEvalTask(CNN):
    def get_validation_vocabulary(self):
        early_stopping_vocab = get_vocabulary(
            self.early_stopping_relations
        )
        validation_vocab = get_vocabulary(self.validation_relations)
        test_vocab = get_vocabulary(self.test_relations)
        return early_stopping_vocab.union(
            validation_vocab
        ).union(
            test_vocab
        )

    def __init__(self):
        super().__init__(
            name="SemEval",
            is_target=True
        )

        self.test_relations = None
        self.validation_relations = None

        self.test_labels = None
        self.validation_labels = None

    def load(self):
        self.load_training_set()
        self.load_test_set()
        self.init_encoder()

    def validation_set(self):
        return self.format_set(
            self.validation_labels,
            self.validation_relations
        )

    def load_training_set(self):
        train_path = os.path.join(
            arguments.data_path,
            config.semeval_train_path
        )
        relations = semeval_parser.read_file(train_path)

        train_relations, validation_relations = split(
            relations,
            arguments.validation_ratio
        )
        train_relations, early_stopping_relations = split(
            train_relations,
            arguments.early_stopping_ratio
        )

        self.train_relations = train_relations
        self.input_length = preprocessing.max_distance(train_relations)
        self.validation_relations = validation_relations
        self.early_stopping_relations = early_stopping_relations

        self.train_labels = get_labels(train_relations)
        self.validation_labels = get_labels(validation_relations)
        self.early_stopping_labels = get_labels(
            early_stopping_relations
        )

    def load_test_set(self):
        test_path = os.path.join(
            arguments.data_path,
            config.semeval_test_path
        )
        self.test_relations = semeval_parser.read_file(test_path)
        self.test_labels = get_labels(self.test_relations)

    def validation_f1(self):
        validation_input, validation_labels = self.validation_set()
        one_hot_prediction = self.model.predict(validation_input)
        label_prediction = self.decode(one_hot_prediction)
        return metrics.f1_score(
            self.validation_labels,
            label_prediction,
            average="macro"
        )

    def validation_report(self):
        validation_input, _ = self.validation_set()
        one_hot_y = self.model.predict(validation_input)
        pred_y = self.decode(one_hot_y)
        report = metrics.classification_report(
            self.validation_labels,
            pred_y
        )
        return report

    def test_report(self):
        log.warning("Computing test set metrics")
        test_input, _ = self.test_set()
        one_hot_y = self.model.predict(test_input)
        pred_y = self.decode(one_hot_y)
        report = metrics.classification_report(
            self.test_labels,
            pred_y
        )
        return report

    def test_set(self):
        return self.format_set(
            self.test_labels,
            self.test_relations
        )
