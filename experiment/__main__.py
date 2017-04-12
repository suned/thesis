import os

from .io import arguments

os.environ["KERAS_BACKEND"] = "theano"

if __name__ == "__main__":
    try:
        arguments.parse()
        from .mtl_relation_extraction import model
        from .mtl_relation_extraction import report
        model.train()
        if arguments.save is not None:
            report.save()
    except arguments.ExperimentExistsError:
        from .mtl_relation_extraction import log
        log.error("Experiment already exists!")
