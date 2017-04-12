import os
import sys
from datetime import datetime

from sklearn.metrics import classification_report
from io import StringIO

from .. import config
from ..io import arguments
from . import log
from .tasks import target_task

report_string = """# {}
## Time\t: {}
### Auxiliary Tasks
{}
### Hyper-Parameters
{}
### SemEval Model Summary
```
{}
```
### Validation Set Report
```
{}
```
"""

test_report = """### Test Report Report
```
{}
```
"""

hyperparam_string = """
| Parameter              | Value |
|-----------------------:|-------|
| max-len                | {:d}  |
| trainable embedding    | {}    |
| batch size             | {}    |
| patience               | {}    |
| dropout                | {}    |
| filters                | {}    |
| n_grams                | {}    |
| position embedding dim | {}    |
"""


def save():
    log.info("Saving result")
    model = target_task.model
    summary = get_summary(model)
    aux_tasks = [task for task in arguments.auxiliary_tasks
                 if task != "none"]
    tasks = "\t".join(aux_tasks)
    true_y = target_task.validation_labels
    report = target_task.validation_report()
    date = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    headline = arguments.save
    hyper_params = hyperparam_string.format(
        arguments.max_len,
        not arguments.freeze_embeddings,
        arguments.batch_size,
        arguments.patience,
        arguments.dropout,
        arguments.filters,
        arguments.n_grams,
        arguments.position_embedding_dimension
    )
    if len(aux_tasks) > 0:
        headline += " + " + " + ".join(aux_tasks)
    output = report_string.format(
        headline,
        date,
        tasks,
        hyper_params,
        summary,
        report
    )
    if arguments.test_set:
        report = target_task.test_report()
        output += test_report.format(report)

    root = os.path.join(config.out_path, arguments.save)
    os.mkdir(root)
    report_path = os.path.join(
        root,
        "report.md"
    )
    with open(report_path, "w") as report_file:
        report_file.write(output)


def get_summary(model):
    stdout = sys.stdout
    sys.stdout = StringIO()
    model.summary()
    summary = sys.stdout.getvalue()
    sys.stdout = stdout
    return summary
