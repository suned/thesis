import os
import sys
from datetime import datetime
import pandas

from io import StringIO

from .. import config
from ..io import arguments
from . import log, fit
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
### Validation Metrics
{}
"""

metrics_string = """
| Metric    | Mean      | Std       |
|----------:|:----------|:----------|
| Macro-F1  | {0:<0.4f} | {1:<0.4f} |
| Precision | {2:<0.4f} | {3:<0.4f} |
| Recall    | {4:<0.4f} | {5:<0.4f} |
"""

hyperparam_string = """
| Parameter              | Value |
|-----------------------:|:------|
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
    metrics = pandas.DataFrame(fit.metrics)
    mean = metrics.mean()
    std = metrics.std()
    metrics_out = metrics_string.format(
        mean["f1"],
        std["f1"],
        mean["precision"],
        std["precision"],
        mean["recall"],
        std["recall"]
    )
    date = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    headline = arguments.save
    hyper_params = hyperparam_string.format(
        arguments.batch_size,
        arguments.patience,
        arguments.dropout,
        arguments.filters,
        arguments.n_grams,
        arguments.position_embedding_dimension
    )
    output = report_string.format(
        headline,
        date,
        tasks,
        hyper_params,
        summary,
        metrics_out
    )

    root = os.path.join(config.out_path, arguments.save)
    os.mkdir(root)
    report_path = os.path.join(
        root,
        "report.md"
    )
    with open(report_path, "w") as report_file:
        report_file.write(output)
    metrics_path = os.path.join(root, "metrics.csv")
    with open(metrics_path, "a") as metrics_file:
        metrics.to_csv(metrics_file, index=False)


def get_summary(model):
    stdout = sys.stdout
    sys.stdout = StringIO()
    model.summary()
    summary = sys.stdout.getvalue()
    sys.stdout = stdout
    return summary
