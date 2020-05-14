import tensorflow as tf
import random


class ModelTrainer:
    def __init__(self, actorModel, criticModel, experienceBag):
        self.actorModel = actorModel
        self.criticModel = criticModel
        self.experienceBag = experienceBag
        self.criticLearningRate = 0.01
        self.actorLearningRate = 0.01
        self.gamma = 0.99

    def trainActorCritic(self):
        allSteps = self.experienceBag.getAllSteps()
        allStates = self.arrayToTFMatrix(
            self.experienceBag.getStates(allSteps))
        currentActionProbabilities = self.actorModel(allStates)
        currentStateValues = self.criticModel(allStates)

        for i in range(len(currentActionProbabilities)):
            allSteps[i].previousActionProbabilities = currentActionProbabilities[i]
            allSteps[i].previousStateValues = currentStateValues[i]

        # minibatch = self.getMinibatch(allSteps, 2)
        minibatch = self.getMinibatch(allSteps, len(allSteps))
        self.trainOnMinibatch(minibatch)

    def getMinibatch(self, allSteps, size):
        # random.shuffle(allSteps)
        return allSteps[0:size]

    def trainOnMinibatch(self, minibatch):
        self.trainActor(minibatch)
        self.trainCritic(minibatch)

    # Train the actor using L_CLIP that is defined in the PPO paper
    def trainActor(self, minibatch):
        optimizer = tf.keras.optimizers.Adam(lr=self.actorLearningRate)

        states = self.arrayToTFMatrix(self.experienceBag.getStates(minibatch))
        previousActionProbabilities = self.arrayToTFMatrix(
            self.experienceBag.getPreviousActionProbabilities(minibatch))
        advantages = self.arrayToTFMatrix(
            self.experienceBag.getAdvantages(minibatch, self.criticModel, self.gamma))

        with tf.GradientTape() as tape:
            predictions = self.actorModel(states)
            ratio = tf.math.exp(tf.math.log(predictions) -
                                tf.math.log(previousActionProbabilities))
            advantageRatio = tf.math.multiply(tf.cast(ratio, tf.float64), tf.cast(advantages, tf.float64))

            ratioClipped = tf.keras.backend.clip(ratio, 0.8, 1.2)
            clippedAdvantageRatio = tf.math.multiply(tf.cast(ratioClipped, tf.float64), tf.cast(advantages, tf.float64))

            gain = tf.math.minimum(advantageRatio, clippedAdvantageRatio)
            # loss = tf.math.multiply(gain, -1)
            loss = tf.math.multiply(gain, 1)

        grads = tape.gradient(loss, self.actorModel.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, self.actorModel.trainable_variables))

    # Train the critic using the squared error
    def trainCritic(self, minibatch):
        optimizer = tf.keras.optimizers.Adam(lr=self.criticLearningRate)

        states = self.arrayToTFMatrix(self.experienceBag.getStates(minibatch))
        stateValues = self.arrayToTFMatrix(
            self.experienceBag.getStateValues(minibatch, self.criticModel, self.gamma))

        with tf.GradientTape() as tape:
            predictions = self.criticModel(states)
            loss = tf.keras.losses.mse(tf.cast(stateValues, tf.float64), tf.cast(predictions, tf.float64))
        grads = tape.gradient(loss, self.criticModel.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, self.criticModel.trainable_variables))

    def arrayToTFMatrix(self, array):
        return tf.reshape(array, [len(array), len(array[0])])
