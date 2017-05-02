from keras import layers

from ..io import arguments


def make_inputs():
    word_input = layers.Input(
        (arguments.max_len,),
        dtype="int32",
        name="word_input"
    )
    position1_input = layers.Input(
        (arguments.max_len,),
        dtype="int32",
        name="position1_input"
    )
    position2_input = layers.Input(
        (arguments.max_len,),
        dtype="int32",
        name="position2_input"
    )
    entity_marker_input = layers.Input(
        (arguments.max_len, 1),
        name="entity_marker_input"
    )
    return (
        word_input,
        position1_input,
        position2_input,
        entity_marker_input
    )