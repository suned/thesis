from keras import layers

from ..io import arguments


def make_convolution_layers(prefix=""):
    word_convolution_layers = []
    position_convolution_layers = []
    for n_gram in arguments.n_grams:
        word_convolution_layer = layers.Conv1D(
            kernel_size=n_gram,
            filters=arguments.filters,
            activation="relu",
            name=prefix + "word_convolution_" + str(n_gram) + "_gram"
        )
        word_convolution_layers.append(word_convolution_layer)
        position_convolution_layer = layers.Conv1D(
            kernel_size=n_gram,
            filters=arguments.filters,
            activation="relu",
            name=prefix +
                 "position_convolution_" +
                 str(n_gram) +
                 "_gram"
        )
        position_convolution_layers.append(position_convolution_layer)
    return word_convolution_layers, position_convolution_layers

(shared_word_convolutions,
 shared_position_convolutions) = make_convolution_layers(
    prefix="shared_"
)
