import sys
from io import StringIO
from datetime import datetime
from .io import arguments
from . import config
from sklearn.metrics import classification_report
import os

from .tasks import target_task
from . import log


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
### Report
```
{}
```
"""

hyperparam_string = """
| Parameter           | Value |
|---------------------|-------|
| max-len             | {:d}  |
| trainable embedding | {}    |
| batch size          | {}    |
| patience            | {}    |
"""



def save():
    log.info("Saving result")
    model = target_task.model
    summary = get_summary(model)
    aux_tasks = [task for task in arguments.auxiliary_tasks
                 if task != "none"]
    tasks = "\t".join(aux_tasks)
    true_y = target_task.validation_labels
    validation_input, _ = target_task.validation_set()
    one_hot_y = model.predict(validation_input)
    pred_y = target_task.decode(one_hot_y)
    report = classification_report(true_y, pred_y)
    date = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    headline = "SemEval"
    hyper_params = hyperparam_string.format(
        arguments.max_len,
        not arguments.freeze_embeddings,
        arguments.batch_size,
        arguments.patience

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
    report_path = os.path.join(
        config.out_path,
        date.replace(" ", "_") + ".md"
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