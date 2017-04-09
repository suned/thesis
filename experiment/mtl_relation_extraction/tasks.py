from mtl_relation_extraction.ace_task import ACETask
from mtl_relation_extraction.semeval_task import SemEvalTask
from . import config
from . import log

target_task = SemEvalTask()
auxiliary_tasks = [
    ACETask()
]

all_tasks = auxiliary_tasks + [target_task]


def load_tasks():
    log.info("Loading %s task", target_task.name)
    target_task.load()
    for auxiliary_task in auxiliary_tasks:
        log.info("Loading %s task", auxiliary_task.name)
        auxiliary_task.load()


def get_batch():
    input_batch = {}
    output_batch = {}
    for task in all_tasks:
        task_input, task_output = task.get_batch()
        input_batch.update(task_input)
        output_batch.update(task_output)
    return input_batch, output_batch
