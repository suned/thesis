from keras import layers

from ..io import arguments


def make_convolution_layers(prefix=""):
    word_convolution_layers = []
    for n_gram in arguments.n_grams:
        word_convolution_layer = layers.Conv1D(
            kernel_size=n_gram,
            filters=arguments.filters,
            activation="relu",
            name=prefix + "word_convolution_" + str(n_gram) + "_gram"
        )
        word_convolution_layers.append(word_convolution_layer)
    return word_convolution_layers

shared_convolutions = make_convolution_layers(
    prefix="shared_"
)


def make_shared_convolutions():
    global shared_convolutions
    shared_convolutions = make_convolution_layers(
        prefix="shared_"
    )
