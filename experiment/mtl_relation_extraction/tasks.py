from ..io import arguments
from .ace_task import ACETask
from .semeval_task import SemEvalTask
from . import log

target_task = SemEvalTask()
auxiliary_tasks = [
    ACETask()
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
