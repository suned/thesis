from keras import utils

from . import config


class Pipeline:
    def __init__(self, encoder, model):
        self.encoder = encoder
        self.model = model

    def batch_fit(self,
                  batch_input,
                  train_labels,
                  validation_data=None,
                  *args,
                  **kwargs):
        if validation_data is not None:
            validation_input, validation_labels = validation_data
            validation_labels = self.encode(validation_labels)
            validation_data = (validation_input, validation_labels)
        one_hot_labels = self.encode(train_labels)
        return self.model.fit(
            batch_input,
            one_hot_labels,
            verbose=config.keras_verbosity,
            validation_data=validation_data,
            *args,
            **kwargs
        )

    def predict(self, test_input):
        y_one_hot = self.model.predict(
            test_input,
            verbose=config.keras_verbosity
        )
        y_indices = y_one_hot.argmax(axis=1)
        return self.encoder.inverse_transform(y_indices)

    def evaluate(self, test_input, labels, *args, **kwargs):
        one_hot_labels = self.encode(labels)
        return self.model.evaluate(
            test_input,
            one_hot_labels,
            verbose=config.keras_verbosity
            *args,
            **kwargs
        )

    def encode(self, train_labels):
        encoded_labels = {}
        for output, labels in train_labels.items():
            integer_labels = self.encoder.transform(labels)
            one_hot_labels = utils.to_categorical(integer_labels)
            encoded_labels[output] = one_hot_labels
        return encoded_labels

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
