import argparse
import logging
import os
import sys


_args = None
log_level = None
auxiliary_tasks = None
data_path = None
max_len = None
dynamic_max_len = None
freeze_embeddings = None
validation_ratio = None
early_stopping_ratio = None
batch_size = None
dropout = None
shared_layer_depth = None
patience = None
epochs = None
filters = None
position_embedding_dimension = None
n_grams = None

_log_levels = [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL"
]

_parser = argparse.ArgumentParser(
    description="""Deep Multi-Task Learning for Relation Extraction.
    Run an experiment on relation extraction data."""
)

_parser.add_argument(
    "--loglevel",
    type=str,
    help="Logging level.",
    default="DEBUG",
    choices=_log_levels
)
_parser.add_argument(
    "--data_path",
    type=str,
    default=os.environ["MTL_DATA"],
    help="""Path to relation data folder. Expected to contain files and
    folders semeval/train.txt, semeval/test/txt, ace/<ace files>
    """
)
_parser.add_argument(
    "--aux-tasks",
    help="List of auxliary tasks to use or none",
    nargs="*",
    choices=["ACE", "none"],
    default=["ACE"]
)
_parser.add_argument(
    "--max-len",
    help="Cut sentences with a length that exceeds max-len. (CNN only)",
    type=int,
    default=15
)
_parser.add_argument(
    "--dynamic-max-len",
    help="Compute max-len based on longest distance between entities.",
    action="store_true",
)
_parser.add_argument(
    "--freeze-embeddings",
    help="Do not backpropagate into word embeddings",
    action="store_true"
)
_parser.add_argument(
    "--validation_ratio",
    help="Validation/train set ration",
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
    "--batch-size",
    help="SGD batch size",
    type=float,
    default=64
)
_parser.add_argument(
    "--dropout",
    help="Add dropout layer before final layer",
    default=False,
    action="store_true"
)
_parser.add_argument(
    "--shared-layer-depth",
    help="Depth of shared dense layers",
    type=int,
    default=0
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
    default=20
)
_parser.add_argument(
    "--filters",
    help="number of filters in n-gram convolutions",
    type=int,
    default=200
)
_parser.add_argument(
    "--n-grams",
    help="width of convolutions",
    type=int,
    nargs="*",
    default=[2, 3, 4, 5]
)
_parser.add_argument(
    "--position-embedding-dim",
    help="dimension of position embedding",
    type=int,
    default=50
)


def parse():
    global _args
    global log_level
    global auxiliary_tasks
    global data_path
    global max_len
    global dynamic_max_len
    global freeze_embeddings
    global validation_ratio
    global early_stopping_ratio
    global batch_size
    global dropout
    global shared_layer_depth
    global patience
    global epochs
    global filters
    global position_embedding_dimension
    global n_grams
    _args = _parser.parse_args()

    log_level = getattr(logging, _args.loglevel)
    auxiliary_tasks = _args.aux_tasks
    data_path = _args.data_path
    max_len = _args.max_len
    dynamic_max_len = _args.dynamic_max_len
    freeze_embeddings = _args.freeze_embeddings
    validation_ratio = _args.validation_ratio
    early_stopping_ratio = _args.early_stopping_ratio
    batch_size = _args.batch_size
    dropout = _args.dropout
    shared_layer_depth = _args.shared_layer_depth
    patience = _args.patience
    epochs = _args.epochs
    filters = _args.filters
    position_embedding_dimension = _args.position_embedding_dim
    n_grams = _args.n_grams
