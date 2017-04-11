import os
os.environ["KERAS_BACKEND"] = "theano"

from mtl_relation_extraction.io import arguments


if __name__ == "__main__":
    arguments.parse()
    from mtl_relation_extraction import model
    from mtl_relation_extraction import report
    model.train()
    report.save()
