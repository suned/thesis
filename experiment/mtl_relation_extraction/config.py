import numpy

semeval_train_path = "semeval/train.txt"
semeval_test_path = "semeval/test.txt"
out_path = "results"
ace_path = "ace"
max_len_buffer = 1
random_state = 1
keras_verbosity = 0
optimizer = "adam"

numpy.random.seed(random_state)
