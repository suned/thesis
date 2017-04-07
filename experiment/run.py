import os

from mtl_relation_extraction import model, config
from mtl_relation_extraction.io import semeval_parser, arguments

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":
    model.train()

