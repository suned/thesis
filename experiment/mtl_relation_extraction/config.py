from keras import optimizers
import tensorflow as tf
import sys

semeval_train_path = "semeval/train.txt"
semeval_test_path = "semeval/test.txt"
max_len_buffer = 5
batch_size = 100
random_state = 1
keras_verbosity = 2
epochs = sys.maxsize
optimizer = optimizers.TFOptimizer(
            tf.train.AdamOptimizer()
)
