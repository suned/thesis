import os
from .io import arguments

os.environ["KERAS_BACKEND"] = "theano"


def train(architecture):
    from .mtl_relation_extraction import fit
    from .mtl_relation_extraction.tasks import load_tasks
    load_tasks()
    architecture.compile_models()
    if arguments.fit_sequential:
        fit.sequential()
    else:
        fit.interleaved()


if __name__ == "__main__":
    try:
        arguments.parse()
        from .mtl_relation_extraction import cnn, rnn
        from .mtl_relation_extraction import report
        if arguments.model == "rnn":
            train(rnn)
        else:
            train(cnn)
        if arguments.save is not None:
            report.save()
    except arguments.ExperimentExistsError:
        from .mtl_relation_extraction import log
        log.error("Experiment already exists!")
