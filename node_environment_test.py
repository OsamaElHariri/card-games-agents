from environments.TestNodesEnvironment import TestNodesEnvironment
from trajectory_utils.ModelFactory import ModelFactory
from trajectory_utils.ExperienceBag import ExperienceBag
from trajectory_utils.ModelTrainer import ModelTrainer
import numpy as np
import tensorflow as tf

numberOfStates = 5


def numberToOneHot(index, size=numberOfStates):
    array = [0] * size
    array[index] = 1
    return array


env = TestNodesEnvironment(numberOfStates)

actorModel = ModelFactory().getActorModel(numberOfStates, 2)
criticModel = ModelFactory().getCriticModel(numberOfStates)


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

for runIndex in range(100):
    done = False
    state = env.reset()
    experienceBag = ExperienceBag()
    episode = []
    states = [state]
    while not done:
        oneHotState = numberToOneHot(state)

        prediction = actorModel(tf.reshape([oneHotState], [1, numberOfStates]))[
            0].numpy()
        action = np.random.choice(len(prediction), p=prediction)
        oneHotAction = numberToOneHot(action, len(prediction))
        print("ACTION = {}".format(action))

        nextState, reward, done = env.step(action)
        states.append(nextState)
        if done:
            states.append("REWARD = {}".format(reward))
        oneHotNextState = numberToOneHot(nextState)
        # print(oneHotState, oneHotAction, reward, oneHotNextState, done)
        episode.append((oneHotState, oneHotAction, reward, oneHotNextState, done))
        # print("state {}, reward {}, done {}".format(nextState, reward, done))
        state = nextState
    printAllStateValues()
    printAllActionProbabilities()
    # print(states)
    experienceBag.addEpisode(episode)
    trainer = ModelTrainer(actorModel, criticModel, experienceBag)
    # print("Training")
    trainer.trainActorCritic(2, 6)
    # print("Trained")
