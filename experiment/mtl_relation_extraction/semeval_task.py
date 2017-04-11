import os

from sklearn import metrics

from . import config
from .io import arguments
from mtl_relation_extraction import log
from mtl_relation_extraction.io import arguments, semeval_parser
from mtl_relation_extraction.preprocessing import max_sentence_length, \
    longest_sentence
from mtl_relation_extraction.task import Task, make_input, get_features, \
    get_positions, get_labels, split


class SemEvalTask(Task):
    def __init__(self):
        super().__init__()
        self.name = "SemEval"
        self.is_target = True
        self.model = None

        self.test_relations = None
        self.validation_relations = None
        self.early_stopping_relations = None

        self.test_features = None
        self.validation_features = None
        self.early_stopping_features = None

        self.test_position1_vectors = None
        self.validation_position1_vectors = None
        self.early_stopping_position1_vectors = None

        self.test_position2_vectors = None
        self.validation_position2_vectors = None
        self.early_stopping_position2_vectors = None

        self.test_labels = None
        self.validation_labels = None
        self.early_stopping_labels = None

        self.max_len = None

    def load(self):
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
            self.name + "_output": self.encode(self.validation_labels)
        }
        return validation_input, validation_labels

    def early_stopping_set(self):
        early_stopping_input = make_input(
            self.early_stopping_features,
            self.early_stopping_position1_vectors,
            self.early_stopping_position2_vectors
        )
        early_stopping_labels = {
            self.name + "_output": self.encode(
                self.early_stopping_labels
            )
        }
        return early_stopping_input, early_stopping_labels

    def load_training_set(self):
        train_path = os.path.join(
            arguments.data_path,
            config.semeval_train_path
        )
        relations = semeval_parser.read_file(train_path)

        self.num_positions = longest_sentence(relations)

        train_relations, validation_relations = split(
            relations,
            arguments.validation_ratio
        )
        train_relations, early_stopping_relations = split(
            train_relations,
            arguments.early_stopping_ratio
        )

        if arguments.dynamic_max_len:
            arguments.max_len = max_sentence_length(train_relations)
            log.info(
                "Dynamically updated max sentence length to %i",
                arguments.max_len
            )

        self.train_relations = train_relations
        self.validation_relations = validation_relations
        self.early_stopping_relations = early_stopping_relations

        self.train_features = get_features(
            train_relations
        )
        self.validation_features = get_features(
            validation_relations
        )
        self.early_stopping_features = get_features(
            early_stopping_relations
        )

        (self.train_position1_vectors,
         self.train_position2_vectors) = get_positions(
            train_relations
        )
        (self.validation_position1_vectors,
         self.validation_position2_vectors) = get_positions(
            validation_relations
        )
        (self.early_stopping_position1_vectors,
         self.early_stopping_position2_vectors) = get_positions(
            early_stopping_relations
        )

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
        self.test_features = get_features(
            self.test_relations
        )
        self.test_labels = get_labels(self.test_relations)
        (self.test_position1_vectors,
         self.test_position2_vectors) = get_positions(
            self.test_relations
        )

    def validation_f1(self):
        validation_input, validation_labels = self.validation_set()
        one_hot_prediction = self.model.predict(validation_input)
        label_prediction = self.decode(one_hot_prediction)
        return metrics.f1_score(
            self.validation_labels,
            label_prediction,
            average="macro"
        )
