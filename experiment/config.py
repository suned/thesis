from keras import optimizers

semeval_train_path = "semeval/train.txt"
semeval_test_path = "semeval/test.txt"
spacy_vectors = "GloVe/spacy.vectors.glove.840B.300d.bin"
out_path = "results"
ace_path = "ace"
max_len_buffer = 1
keras_verbosity = 0
optimizer = optimizers.Adam()
word_vector_path = "GloVe/glove.840B.300d.pkl"
kbp37_train = "kbp37/train.clean.txt"
kbp37_test = "kbp37/test.txt"
kbp37_dev = "kbp37/dev.txt"
gmb_root = "gmb-2.2.0/data"
fractions = [.0, .2, .4, .6, .8, 1.0]
