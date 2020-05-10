import tensorflow as tf


class ModelFactory:
    def getActorModel(self, inputSize, outputSize):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(inputSize,)),
                tf.keras.layers.Dense(120, activation="relu"),
                tf.keras.layers.Dense(250, activation="relu"),
                tf.keras.layers.Dense(outputSize, activation="softmax"),
            ]
        )

        return model

    def getCriticModel(self, inputSize):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(inputSize,)),
                tf.keras.layers.Dense(120, activation="relu"),
                tf.keras.layers.Dense(250, activation="relu"),
                tf.keras.layers.Dense(1, activation="tanh"),
            ]
        )

        return model
