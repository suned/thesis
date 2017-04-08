from keras import optimizers
import tensorflow as tf
import sys

semeval_train_path = "semeval/train.txt"
semeval_test_path = "semeval/test.txt"
max_len_buffer = 5
max_len = 15
dynamic_max_len = True
batch_size = 200
random_state = 1
keras_verbosity = 0
epochs = sys.maxsize
patience = 20
optimizer = optimizers.TFOptimizer(
    tf.train.AdamOptimizer()
)
validation_ratio = .2
