from . import (
    fit
)
from . import log
from .tasks import experiment_tasks
from ..io import arguments


def compile():
    for task in experiment_tasks:
        log.info("Compiling model for task: %s", task.name)
        task.compile_model()


def train():
    if arguments.fit_sequential:
        fit.sequential()
    else:
        fit.interleaved()

