import gc

import math
import random

import numpy
import itertools

from .. import config
from ..io import arguments
from . import log, embeddings, convolutions, inputs
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
        "f1": [],
        "targetFraction": [],
        "auxFraction": []
    }


metrics = init_metrics()


def init_weights():
    log.info("Initialising weights")
    embeddings.make_shared_embeddings()
    convolutions.make_shared_convolutions()
    for task in experiment_tasks:
        task.init_weights()


def save_metrics():
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


def load_fractions():
    root = os.path.join(config.out_path, arguments.save)
    metrics_path = os.path.join(root, "metrics.csv")
    if os.path.exists(root):
        metrics_frame = pandas.read_csv(metrics_path)
        metrics_frame.targetFraction = metrics_frame.targetFraction.astype(str)
        metrics_frame.auxFraction = metrics_frame.auxFraction.astype(str)
        fraction_counts = (metrics_frame
                           .groupby(["targetFraction", "auxFraction"])
                           .count()
                           )
        missing_fractions = []
        for target_fraction in config.fractions:
            for auxiliary_fraction in config.fractions:
                was_started = (str(target_fraction),
                               str(auxiliary_fraction)) in fraction_counts.index
                if was_started and fraction_counts.ix[
                    str(target_fraction),
                    str(auxiliary_fraction)
                ]['f1'] < arguments.k_folds * arguments.iterations:
                    missing_fractions.append((target_fraction, auxiliary_fraction))
                elif not was_started:
                    missing_fractions.append((target_fraction, auxiliary_fraction))
        return missing_fractions
    else:
        fractions = list(itertools.product(config.fractions, config.fractions))
        return fractions


def find_start_iteration():
    root = os.path.join(config.out_path, arguments.save)
    metrics_path = os.path.join(root, "metrics.csv")
    if os.path.exists(metrics_path):
        metrics_frame = pandas.read_csv(metrics_path)
        max_aux_fraction = metrics_frame.auxFraction.max()
        max_target_fraction = metrics_frame[
            metrics_frame.auxFraction == max_aux_fraction
            ].targetFraction.max()
        folds = len(metrics_frame[
                        (metrics_frame.targetFraction == max_target_fraction) &
                        (metrics_frame.auxFraction == max_aux_fraction)
                        ])
        start_iteration = (folds % arguments.k_folds) - 1
        return start_iteration
    else:
        return 1


def interleaved():
    global metrics
    if arguments.learning_surface:
        fractions = load_fractions()
    else:
        fractions = [(1.0, 1.0)]
    start_iteration = 1
    for target_fraction, auxiliary_fraction in fractions:
        log.info("Starting auxiliary fraction %f", auxiliary_fraction)
        log.info("Starting target fraction %f", target_fraction)
        reduce_aux_data(auxiliary_fraction)
        for iteration in range(start_iteration,
                               arguments.iterations + 1):
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
                log.info(
                    "Starting fold %i of %i",
                    k,
                    arguments.k_folds
                )
                target_task.split(train_indices, test_indices)
                target_task.reduce_train_data(target_fraction)
                early_stopping(
                    target_fraction,
                    auxiliary_fraction
                )
                k += 1
                init_weights()
                save_metrics()
                metrics = init_metrics()
        start_iteration = 1
log.info("Done!!")


def reduce_aux_data(auxiliary_fraction):
    for task in experiment_tasks:
        if not task.is_target:
            task.reduce_train_data(auxiliary_fraction)


def append(validation_metrics):
    for metric, value in validation_metrics.items():
        metrics[metric].append(value)


def is_empty(batch_input):
    rows, columns = batch_input[inputs.word_input].shape
    return rows == 0


def early_stopping(target_fraction, auxiliary_fraction):
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
    import ipdb
    ipdb.sset_trace()
    while epoch <= arguments.epochs:
        task = random.choice(experiment_tasks)
        batch_input, batch_labels = task.get_batch()
        epoch_stats = task.model.fit(
            batch_input,
            batch_labels,
            epochs=1,
            batch_size=arguments.batch_size,
            verbose=config.keras_verbosity
        )
        training_loss = (epoch_stats.history["loss"][0]
                         if "loss" in epoch_stats.history
                         else float("inf"))
        early_stopping_loss = target_task.model.evaluate(
            early_stopping_input,
            early_stopping_labels,
            verbose=config.keras_verbosity
        )
        if math.isnan(training_loss) or math.isnan(early_stopping_loss):
            log.error("Illegal loss detected, skipping fold")
            import ipdb
            ipdb.sset_trace()
            return
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
    validation_metrics["targetFraction"] = target_fraction
    validation_metrics["auxFraction"] = auxiliary_fraction
    append(validation_metrics)
    log.info("Validation F1: %f", validation_metrics["f1"])
