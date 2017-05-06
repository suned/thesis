from keras import layers


word_input = "word_input"
position1_input = "position1_input"
position2_input = "position2_input"


def make_word_input(input_length):
    return layers.Input(
        (input_length,),
        dtype="int32",
        name=word_input
    )


def make_position_inputs(input_length):
    position1_input_layer = layers.Input(
        (input_length,),
        dtype="int32",
        name=position1_input
    )
    position2_input_layer = layers.Input(
        (input_length,),
        dtype="int32",
        name=position2_input
    )
    return position1_input_layer, position2_input_layer
