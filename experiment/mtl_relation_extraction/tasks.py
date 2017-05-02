from ..io import arguments
from .ace_task import ACETask
from .semeval_task import SemEvalTask
from .kbp37_task import KBP37Task
from . import log
from . import nlp

target_task = SemEvalTask()
auxiliary_tasks = [
    ACETask(),
    KBP37Task()
]
experiment_tasks = []
all_tasks = auxiliary_tasks + [target_task]


def load_tasks():
    global experiment_tasks
    log.info("Loading %s task", target_task.name)
    target_task.load()
    experiment_tasks.append(target_task)
    for auxiliary_task in auxiliary_tasks:
        if auxiliary_task.name in arguments.auxiliary_tasks:
            log.info("Loading %s task", auxiliary_task.name)
            auxiliary_task.load()
            experiment_tasks.append(auxiliary_task)
    nlp.add_vocabularies(experiment_tasks)
