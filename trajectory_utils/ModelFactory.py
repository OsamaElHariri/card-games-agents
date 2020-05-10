import tensorflow as tf


class ModelFactory:
    def getActorModel(self, input_size, output_size):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_size,)),
                tf.keras.Dense(120, activation="relu"),
                tf.keras.Dense(250, activation="relu"),
                tf.keras.Dense(output_size, activation="softmax"),
            ]
        )

        return model

    def getCriticModel(self, input_size):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_size,)),
                tf.keras.Dense(120, activation="relu"),
                tf.keras.Dense(250, activation="relu"),
                tf.keras.Dense(1, activation="tanh"),
            ]
        )

        return model
