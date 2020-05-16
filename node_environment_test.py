from environments.TestNodesEnvironment import TestNodesEnvironment
from trajectory_utils.ModelFactory import ModelFactory
from trajectory_utils.ExperienceBag import ExperienceBag
from trajectory_utils.ModelTrainer import ModelTrainer
import numpy as np
import tensorflow as tf
import gym


cartPoleEnv = gym.make("FrozenLake-v0", is_slippery=False)
actionSpaceSize = cartPoleEnv.action_space.n
stateSpaceSize = cartPoleEnv.observation_space.n

print(stateSpaceSize, actionSpaceSize)

numberOfStates = 5


def numberToOneHot(index, size=numberOfStates):
    array = [0] * size
    array[index] = 1
    return array


env = TestNodesEnvironment(numberOfStates)

actorModel = ModelFactory().getActorModel(stateSpaceSize, actionSpaceSize)
criticModel = ModelFactory().getCriticModel(stateSpaceSize)

# actorModel = ModelFactory().getActorModel(numberOfStates, 2)
# criticModel = ModelFactory().getCriticModel(numberOfStates)


def printAllStateValues():
    values = []
    for state in range(numberOfStates):
        oneHotState = numberToOneHot(state)
        value = criticModel(tf.reshape([oneHotState], [1, numberOfStates]))[
            0].numpy()[0]
        values.append(value)
    print(values)


def printAllActionProbabilities():
    values = []
    for state in range(numberOfStates):
        oneHotState = numberToOneHot(state)
        value = actorModel(tf.reshape([oneHotState], [1, numberOfStates]))[
            0].numpy()
        values.append(value)
    print(values)

# bag = ExperienceBag()

# bag.addEpisode([
#     ([0, 0, 1, 0, 0], [0, 1], 0, [0, 0, 0, 1, 0], False),
#     ([0, 0, 0, 1, 0], [0, 1], 1, [0, 0, 0, 0, 1], True)
# ])
# bag.addEpisode([
#     ([0, 0, 1, 0, 0], [1, 0], 0, [0, 1, 0, 0, 0], False),
#     ([0, 1, 0, 0, 0], [1, 0], -1, [1, 0, 0, 0, 0], True)
# ])


# trainer = ModelTrainer(actorModel, criticModel, bag)

# printAllStateValues()
# for i in range(100):
#     trainer.trainActorCritic()
# printAllStateValues()

# exit()

while True:
    experienceBag = ExperienceBag()
    for runIndex in range(10):
        done = False
        state = cartPoleEnv.reset()
        episode = []
        reward = 0
        steps = []
        while not done or len(episode) > 200:
            oneHotState = numberToOneHot(state, stateSpaceSize)

            prediction = actorModel(tf.reshape([oneHotState], [1, stateSpaceSize]))[
                0].numpy()
            action = np.random.choice(len(prediction), p=prediction)
            oneHotAction = numberToOneHot(action, len(prediction))

            nextState, reward, done, _ = cartPoleEnv.step(action)
            oneHotNextState = numberToOneHot(nextState, stateSpaceSize)
            episode.append((oneHotState, oneHotAction,
                            reward, oneHotNextState, done))
            steps.append("{}, {}".format(state, np.argmax(prediction)))
            if done:
                steps.append(reward)
            state = nextState
        experienceBag.addEpisode(episode)
        print(steps)
        print("------------------------------------")
    print("Ep")
    trainer = ModelTrainer(actorModel, criticModel, experienceBag)
    trainer.trainActorCritic(100, 16)
