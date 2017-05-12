from .. import config
from ..io import arguments
from . import log
from .tasks import target_task, experiment_tasks


log_header = """{0:<10} {1:<20} {2:<15} {3:<15}""".format(
    "Epoch",
    "Task",
    "Training Loss",
    "Early Stopping Loss"
)
log_header += "\n" + "=" * len(log_header)
log_line = """{0:<10d} {1:<20} {2:<15.4f} {3:<1.4f} {4}"""


def interleaved():
    best_validation_loss = float("inf")
    best_weights = None
    log.info(
        "Training interleaved with %i training samples, "
        "%i early stopping samples",
        len(target_task.train_relations),
        len(target_task.early_stopping_relations),
    )
    epochs_without_improvement = 0
    (early_stopping_input,
     early_stopping_labels) = target_task.early_stopping_set()
    task_count = len(experiment_tasks)
    log.info(log_header)
    for epoch in range(1, arguments.epochs + 1):
        task = experiment_tasks[epoch % task_count]
        batch_input, batch_labels = task.get_batch()
        epoch_stats = task.model.fit(
            batch_input,
            batch_labels,
            epochs=1,
            verbose=config.keras_verbosity
        )
        training_loss = epoch_stats.history["loss"][0]
        validation_loss = target_task.model.evaluate(
            early_stopping_input,
            early_stopping_labels,
            verbose=config.keras_verbosity
        )
        if validation_loss < best_validation_loss:
            optimum = "*"
            best_validation_loss = validation_loss
            best_weights = target_task.model.get_weights()
            epochs_without_improvement = 0
        else:
            optimum = ""
            epochs_without_improvement += 1
        log.info(
            log_line.format(
                epoch,
                task.name,
                training_loss,
                validation_loss,
                optimum
            )
        )
        if task.is_target and training_loss < .01:
            log.info("Training loss maximised. Stopping")
            break
        if epochs_without_improvement > arguments.patience:
            log.info("Patience exceeded. Stopping")
            break
    log.info(
        "Finished training with best loss: %f",
        best_validation_loss
    )
    target_task.model.set_weights(best_weights)
    log.info("Validation F1: %f", target_task.validation_f1())


def sequential():
    auxiliary_tasks = [task for task in experiment_tasks
                       if not task.is_target]
    log.info("Training sequentially")
    for task in auxiliary_tasks:
        fit_early_stopping(task)
    fit_early_stopping(target_task)
    log.info("Validation F1: %f", target_task.validation_f1())


def fit_early_stopping(task):
    best_early_stopping_loss = float("inf")
    best_weights = None
    log.info("Training on task: %s", task.name)
    log.info(log_header)
    early_stopping_set = task.early_stopping_set()
    epochs_without_improvement = 0
    for epoch in range(1, arguments.epochs + 1):
        training_input, training_labels = task.get_batch()
        epoch_stats = task.model.fit(
            training_input,
            training_labels,
            epochs=1,
            verbose=config.keras_verbosity,
            validation_data=early_stopping_set,
        )
        training_loss = epoch_stats.history["loss"][0]
        early_stopping_loss = epoch_stats.history["val_loss"][0]
        if early_stopping_loss < best_early_stopping_loss:
            optimum = "*"
            best_early_stopping_loss = early_stopping_loss
            best_weights = task.model.get_weights()
            epochs_without_improvement = 0
        else:
            optimum = ""
            epochs_without_improvement += 1
        log.info(
            log_line.format(
                epoch,
                task.name,
                training_loss,
                early_stopping_loss,
                optimum
            )
        )
        if training_loss < .01:
            log.info("Training F1 maximised. Stopping")
            break
        if epochs_without_improvement > arguments.patience:
            log.info("Patience exceeded. Stopping")
            break
    log.info(
        "Finished training with best loss: %f",
        best_early_stopping_loss
    )
    task.model.set_weights(best_weights)

