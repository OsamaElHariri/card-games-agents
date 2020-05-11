import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Dense(120, activation="relu"),
        tf.keras.layers.Dense(250, activation="relu"),
        tf.keras.layers.Dense(1, activation="tanh"),
    ]
)

dataSet = [
    [1, -1],
    [1, 1],
    [-1, -1],
    [-1, 1]
]

dataOutput = [
    1.0,
    -1.0,
    1.0,
    -1.0
]


print(model(tf.reshape(dataSet, [4, 2])))
optimizer = tf.keras.optimizers.Adam(lr=0.01)


def step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(tf.reshape(inputs, [4, 2]))
        targets = tf.reshape(targets, [4, 1])
        loss = tf.math.pow(predictions - targets, 2)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


for i in range(100):
    step(dataSet, dataOutput)


print(model(tf.reshape(dataSet, [4, 2])))
