from ..io import arguments
from .ace_task import ACE
from .semeval_task import SemEvalTask
from .kbp37_task import KBP37Task
from .conll2000_pos_task import Conll2000PosTask
from . import log
from . import nlp

target_task = SemEvalTask()
longest_sentence = None
auxiliary_tasks = [
    ACE(),
    KBP37Task(),
    Conll2000PosTask()
]
experiment_tasks = []
all_tasks = auxiliary_tasks + [target_task]


def load_tasks():
    global experiment_tasks
    global longest_sentence
    log.info("Loading %s task", target_task.name)
    target_task.load()
    experiment_tasks.append(target_task)
    for auxiliary_task in auxiliary_tasks:
        if auxiliary_task.name in arguments.auxiliary_tasks:
            log.info("Loading %s task", auxiliary_task.name)
            auxiliary_task.load()
            experiment_tasks.append(auxiliary_task)
    longest_sentence = max(task.longest_sentence()
                           for task in experiment_tasks)
    nlp.add_vocabularies(experiment_tasks)
