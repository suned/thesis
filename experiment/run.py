import os
os.environ["KERAS_BACKEND"] = "theano"

from mtl_relation_extraction import model


if __name__ == "__main__":
    model.train()

