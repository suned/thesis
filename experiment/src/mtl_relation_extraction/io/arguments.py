import argparse
import logging
import os

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
    help="Path to relation data."
)

_args = _parser.parse_args()

log_level = getattr(logging, _args.loglevel)
data_path = _args.data_path
