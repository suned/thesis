import gc

import math
import random

from .. import config
from ..io import arguments
from . import log, embeddings
from .tasks import target_task, experiment_tasks
from sklearn.model_selection import KFold
from keras import backend
import os
import pandas
from pygpu.gpuarray import GpuArrayException

log_header = """{0:<10} {1:<20} {2:<15} {3:<15}""".format(
    "Epoch",
    "Task",
    "Training Loss",
    "Early Stopping Loss"
)
log_header += "\n" + "=" * len(log_header)
log_line = """{0:<10d} {1:<20} {2:<15.4f} {3:<1.4f} {4}"""


def init_metrics():
    return {
        "precision": [],
        "recall": [],
        "f1": []
    }


metrics = init_metrics()


def init_weights():
    log.info("Initialising weights")
    embeddings.make_shared_embeddings()
    for task in experiment_tasks:
        task.init_weights()


def save_metrics():
    if arguments.save:
        root = os.path.join(config.out_path, arguments.save)
        os.makedirs(root, exist_ok=True)
        metrics_path = os.path.join(root, "metrics.csv")
        metrics_frame = pandas.DataFrame(metrics)
        if os.path.exists(metrics_path):
            metrics_frame.to_csv(
                metrics_path,
                index=False,
                mode="a",
                header=False
            )
        else:
            metrics_frame.to_csv(
                metrics_path,
                index=False
            )


def interleaved():
    global metrics
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
            try:
                early_stopping()
                k += 1
                init_weights()
                save_metrics()
                metrics = init_metrics()
            except RuntimeError as e:
                log.error(str(e))
            except GpuArrayException as e:
                log.error(str(e))


def append(validation_metrics):
    for metric, value in validation_metrics.items():
        metrics[metric].append(value)


def early_stopping():
    best_early_stopping_loss = float("inf")
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
    log.info(log_header)
    epoch = 1
    while epoch <= arguments.epochs:
        task = random.choice(experiment_tasks)
        batch_input, batch_labels = task.get_batch()
        epoch_stats = task.model.fit(
            batch_input,
            batch_labels,
            epochs=1,
            verbose=config.keras_verbosity
        )
        training_loss = epoch_stats.history["loss"][0]
        early_stopping_loss = target_task.model.evaluate(
            early_stopping_input,
            early_stopping_labels,
            verbose=config.keras_verbosity
        )
        if math.isnan(training_loss) or math.isnan(early_stopping_loss):
            log.error("Illegal loss detected, restarting fold")
            init_weights()
            epoch = 1
            QQ = 0
            best_early_stopping_loss = float("inf")
            log.info(log_header)
            continue
        if early_stopping_loss < best_early_stopping_loss:
            optimum = "*"
            best_early_stopping_loss = early_stopping_loss
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
                early_stopping_loss,
                optimum
            )
        )
        if task.is_target and training_loss < .001:
            log.info("Training loss maximised. Stopping")
            break
        if epochs_without_improvement > arguments.patience:
            log.info("Patience exceeded. Stopping")
            break
        epoch += 1
    log.info(
        "Finished training with best loss: %f",
        best_early_stopping_loss
    )
    target_task.model.set_weights(best_weights)
    validation_metrics = target_task.validation_metrics()
    append(validation_metrics)
    log.info("Validation F1: %f", validation_metrics["f1"])
