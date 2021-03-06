import argparse
import logging
import os
import sys

from .. import config

log_level = None
save = None
auxiliary_tasks = None
data_path = None
validation_ratio = None
early_stopping_ratio = None
batch_size = None
dropout = None
hidden_layer_dimension = None
patience = None
epochs = None
filters = None
position_embedding_dimension = None
n_grams = None
share_filters = None
word_embedding_dimension = None
k_folds = None
iterations = None
window_size = None
learning_surface = None
multi_channel = None


_log_levels = [
    logging.getLevelName(level)
    for level in [
        logging.DEBUG,
        logging.WARN,
        logging.INFO,
        logging.ERROR,
        logging.FATAL,
    ]
]


class LevelAction(argparse.Action):
    def __call__(self, parser, namespace, level, option_string=None):
        level = getattr(logging, level)
        setattr(namespace, self.dest, level)


_parser = argparse.ArgumentParser(
    description="""Deep Multi-Task Learning for Relation Extraction.
    Run an experiment on relation extraction data."""
)

_parser.add_argument(
    "--log-level",
    type=str,
    help="Logging level.",
    default=logging.INFO,
    choices=_log_levels,
    action=LevelAction
)
_parser.add_argument(
    "--data-path",
    type=str,
    default=os.environ["MTL_DATA"],
    help="""Path to relation data folder. Expected to contain files and
    folders semeval/train.txt, semeval/test/txt, ace/<ace files>
    """
)
_parser.add_argument(
    "--auxiliary-tasks",
    help="List of auxiliary tasks to use or none",
    nargs="*",
    choices=[
        "ACE",
        "KBP37",
        "Conll2000POS",
        "none",
        "Conll2000Chunk",
        "GMB-NER"],
    default=["ACE"]
)
_parser.add_argument(
    "--validation-ratio",
    help="Validation/train set ratio",
    type=float,
    default=.2
)
_parser.add_argument(
    "--early-stopping-ratio",
    help="""Early stopping/train ratio. 
    Split is made after validation/train split""",
    type=float,
    default=.1
)
_parser.add_argument(
    "--word-embedding-dimension",
    help="dimension of the shared word embedding",
    type=int,
    default=300
)
_parser.add_argument(
    "--batch-size",
    help="SGD batch size",
    type=int,
    default=64
)
_parser.add_argument(
    "--dropout",
    help="Add dropout layer before final layer",
    default=False,
    action="store_true"
)
_parser.add_argument(
    "--epochs",
    help="Max number of epochs",
    type=int,
    default=sys.maxsize
)
_parser.add_argument(
    "--patience",
    help="Max number of epochs without improvement "
         "on early stopping set",
    type=int,
    default=200
)
_parser.add_argument(
    "--filters",
    help="number of filters in n-gram convolutions. (CNN only)",
    type=int,
    default=150
)
_parser.add_argument(
    "--n-grams",
    help="width of convolutions",
    type=int,
    nargs="*",
    default=[1, 2, 3, 4, 5]
)
_parser.add_argument(
    "--position-embedding-dimension",
    help="dimension of position embedding",
    type=int,
    default=50
)
_parser.add_argument(
    "save",
    help="name of experiment results to save",
    type=str
)
_parser.add_argument(
    "--hidden-layer-dimension",
    help="dimensionality of dense hidden layers",
    type=int,
    default=500
)
_parser.add_argument(
    "--share-filters",
    help="share convolution filters between sentence models. "
         "Only compatible with --auxiliary-tasks=ACE",
    action="store_true"
)
_parser.add_argument(
    "--k-folds",
    help="number of folds in cross validation",
    type=int,
    default=5
)
_parser.add_argument(
    "--multi-channel",
    help="use a multi-channel strategy for weight sharing",
    action="store_true"
)
_parser.add_argument(
    "--iterations",
    help="number of iterations to perform cross-validation",
    type=int,
    default=20
)
_parser.add_argument(
    "--window-size",
    help="window size for sequence CNN models",
    type=int,
    default=7
)
_parser.add_argument(
    "--learning-surface",
    help="save learning surface data",
    action="store_true"
)


class ExperimentExistsError(Exception):
    pass


def experiment_exists():
    return False


class IncompatibleModelError(Exception):
    pass


def parse():
    arguments = sys.modules[__name__]
    _args = _parser.parse_args()
    for argument, value in vars(_args).items():
        if getattr(arguments, argument) is None:
            setattr(arguments, argument, value)
    if experiment_exists():
        raise ExperimentExistsError
    if share_filters and auxiliary_tasks != ["ACE"]:
        raise IncompatibleModelError
