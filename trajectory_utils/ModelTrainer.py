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

    def trainActorCritic(self, minibatchSize, minibatchTrainCount):
        allSteps = self.experienceBag.getAllSteps()
        allStates = self.arrayToTFMatrix(
            self.experienceBag.getStates(allSteps))
        currentActionProbabilities = self.actorModel(allStates)
        currentStateValues = self.criticModel(allStates)

        for i in range(len(currentActionProbabilities)):
            allSteps[i].previousActionProbabilities = currentActionProbabilities[i]
            allSteps[i].previousStateValues = currentStateValues[i]

        for i in range(minibatchTrainCount):
            minibatch = self.getMinibatch(allSteps, minibatchSize)
            self.trainOnMinibatch(minibatch)

    def getMinibatch(self, allSteps, size):
        random.shuffle(allSteps)
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

            ratio = tf.cast(ratio, tf.float64)
            advantages = tf.cast(advantages, tf.float64)

            advantageRatio = ratio * advantages

            ratioClipped = tf.keras.backend.clip(ratio, 0.8, 1.2)
            ratioClipped = tf.cast(ratioClipped, tf.float64)

            clippedAdvantageRatio = ratioClipped * advantages

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

            predictions = tf.cast(predictions, tf.float64)
            stateValues = tf.cast(stateValues, tf.float64)

            loss = tf.keras.losses.mse(stateValues, predictions)

        grads = tape.gradient(loss, self.criticModel.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, self.criticModel.trainable_variables))

    def arrayToTFMatrix(self, array):
        return tf.reshape(array, [len(array), len(array[0])])
