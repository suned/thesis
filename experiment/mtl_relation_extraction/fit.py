from .. import config
from ..io import arguments
from . import log
from .tasks import target_task, experiment_tasks
from sklearn.model_selection import KFold


log_header = """{0:<10} {1:<20} {2:<15} {3:<15}""".format(
    "Epoch",
    "Task",
    "Training Loss",
    "Early Stopping Loss"
)
log_header += "\n" + "=" * len(log_header)
log_line = """{0:<10d} {1:<20} {2:<15.4f} {3:<1.4f} {4}"""

metrics = {
    "precision": [],
    "recall": [],
    "f1": []
}


def init_weights():
    log.info("Initialising weights")
    for task in experiment_tasks:
        task.init_weights()


def interleaved():
    for iteration in range(1, arguments.iterations + 1):
        log.info(
            "Starting iteration %i of %i",
            iteration,
            arguments.iterations
        )
        iterator = KFold(n_splits=arguments.k_folds)
        k = 1
        for train_indices, test_indices in iterator.split(
                target_task.relations
        ):
            log.info("Starting fold %i of %i", k, arguments.k_folds)
            target_task.split(train_indices, test_indices)
            early_stopping()
            k += 1
            init_weights()


def append(validation_metrics):
    for metric, value in validation_metrics.items():
        metrics[metric].append(value)


def early_stopping():
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
    validation_metrics = target_task.validation_metrics()
    append(validation_metrics)
    log.info("Validation F1: %f", validation_metrics["f1"])
