import os

os.environ["KERAS_BACKEND"] = "theano"

from .io import arguments
import gc


def run():
    try:
        arguments.parse()
        from .mtl_relation_extraction import tasks, nlp
        nlp.load_glove_vectors()
        tasks.load_tasks()
        from .mtl_relation_extraction import (
            models,
            report,
            embeddings,
            convolutions,
            log
        )
        embeddings.make_shared_embeddings()
        convolutions.make_shared_convolutions()
        nlp.glove_vectors = None
        log.info("Garbage collecting")
        gc.collect()
        models.compile()
        models.train()
    except arguments.ExperimentExistsError:
        from .mtl_relation_extraction import log
        log.error("Experiment already exists!")


if __name__ == "__main__":
    run()
