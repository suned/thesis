import os
from .io import arguments

os.environ["KERAS_BACKEND"] = "theano"


def run():
    try:
        arguments.parse()
        from .mtl_relation_extraction import tasks, nlp
        nlp.load_glove_vectors()
        tasks.load_tasks()
        from .mtl_relation_extraction import (
            models,
            report,
            embeddings
        )
        embeddings.make_shared_embeddings()
        models.compile()
        models.train()
        if arguments.save is not None:
            report.save()
    except arguments.ExperimentExistsError:
        from .mtl_relation_extraction import log
        log.error("Experiment already exists!")


if __name__ == "__main__":
    run()
