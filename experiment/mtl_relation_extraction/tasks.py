from . import log
from . import nlp
from .ace_task import ACE
from .conll2000_pos_task import Conll2000PosTask
from .kbp37_task import KBP37Task
from .semeval_task import SemEvalTask
from .conll2000_chunk_task import Conll2000ChunkTask
from .gmb_ner_task import GMBNERTask
from ..io import arguments

target_task = SemEvalTask()
auxiliary_tasks = [
    ACE(),
    KBP37Task(),
    Conll2000PosTask(),
    Conll2000ChunkTask(),
    GMBNERTask()
]
experiment_tasks = []
all_tasks = auxiliary_tasks + [target_task]


def load_tasks():
    global experiment_tasks
    log.info("Loading %s task", target_task.name)
    target_task.load()
    experiment_tasks.append(target_task)
    for auxiliary_task in auxiliary_tasks:
        log.info("Loading %s task", auxiliary_task.name)
        auxiliary_task.load()
        if auxiliary_task.name in arguments.auxiliary_tasks:
            experiment_tasks.append(auxiliary_task)
    nlp.longest_sentence = max(task.longest_sentence()
                               for task in experiment_tasks)
    nlp.add_vocabularies(all_tasks)
