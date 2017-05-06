from keras import layers

from ..io import arguments


def make_convolution_layers(prefix=""):
    convolution_layers = []
    for n_gram in arguments.n_grams:
        convolution_layer = layers.Conv1D(
            kernel_size=n_gram,
            filters=arguments.filters,
            activation="relu",
            name=prefix + "convolution_" + str(n_gram) + "_gram"
        )
        convolution_layers.append(convolution_layer)
    return convolution_layers

shared_convolutions = make_convolution_layers(prefix="shared_")
