import sys
import numpy

semeval_train_path = "semeval/train.txt"
semeval_test_path = "semeval/test.txt"
ace_path = "ace"
max_len_buffer = 5
max_len = 15
dynamic_max_len = True
batch_size = 32
random_state = 1
keras_verbosity = 0
epochs = sys.maxsize
patience = 20
optimizer = "adam"
validation_ratio = .2
early_stopping_ratio = .2
train_word_embeddings = True

numpy.random.seed(random_state)
