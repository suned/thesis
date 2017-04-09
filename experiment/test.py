import os
os.environ["KERAS_BACKEND"] = "theano"

import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split

x1, y1 = datasets.make_blobs(centers=2, n_samples=10000)
x2, y2 = datasets.make_blobs(centers=2, n_samples=10000)
x1_train, x1_val, y1_train, y1_val = train_test_split(
    x1,
    y1,
    test_size=.2
)
x1_input = keras.layers.Input((2,), name="x1_input")
x2_input = keras.layers.Input((2,), name="x2_input")
input_layer = keras.layers.concatenate([x1_input, x2_input])
hidden = keras.layers.Dense(
    1000,
    name="shared_hidden",
    activation="relu"
)(input_layer)
x1_output = keras.layers.Dense(
    1,
    name="x1_output",
    activation="sigmoid"
)(hidden)
x2_output = keras.layers.Dense(
    1,
    name="x2_output",
    activation="sigmoid"
)(hidden)

model = keras.models.Model(
    inputs=[x1_input, x2_input],
    outputs=[x1_output, x2_output]
)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)
print(model.summary())
print(model.summary())
model.fit(
    {
        "x1_input": x1,
        "x2_input": x2
    },
    {
        "x1_output": y1,
        "x2_output": y2
    },
    validation_split=.2,
    epochs=10
)
