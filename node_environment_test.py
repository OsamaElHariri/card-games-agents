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
    for state in range(stateSpaceSize):
        oneHotState = numberToOneHot(state, stateSpaceSize)
        value = criticModel(tf.reshape([oneHotState], [1, stateSpaceSize]))[
            0].numpy()[0]
        values.append(value)
    print(values)


def printAllStateValuesGrid():
    values = []
    for state in range(stateSpaceSize):
        oneHotState = numberToOneHot(state, stateSpaceSize)
        # print(tf.reshape([oneHotState], [1, stateSpaceSize]))
        value = criticModel(tf.reshape([oneHotState], [1, stateSpaceSize]))[
            0].numpy()[0]
        values.append(value)
        # print(value)
    for i in range(4):
        print(values[i * 4: (i + 1) * 4])
    # print(values)


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

wins = 0
count = 0
criticBag = ExperienceBag()
while True:
    experienceBag = ExperienceBag()
    for runIndex in range(10):
        count += 1
        done = False
        state = cartPoleEnv.reset()
        episode = []
        reward = 0
        steps = []
        states = []
        while not done and len(episode) < 20:
            oneHotState = numberToOneHot(state, stateSpaceSize)

            prediction = actorModel(tf.reshape([oneHotState], [1, stateSpaceSize]))[
                0].numpy()
            action = np.random.choice(len(prediction), p=prediction)
            # action = np.random.choice(
            #     len(prediction), p=[0.25, 0.25, 0.25, 0.25])
            # if state == 0:
            #     action = 2
            # elif state == 1:
            #     action = 2
            # elif state == 2:
            #     action = 1
            # elif state == 6:
            #     action = 1
            # elif state == 10:
            #     action = 1
            # elif state == 14:
            #     action = 2
            oneHotAction = numberToOneHot(action, len(prediction))

            nextState, reward, done, _ = cartPoleEnv.step(action)
            oneHotNextState = numberToOneHot(nextState, stateSpaceSize)

            # if state == 0 and nextState == 1:
            #     reward = 0.1
            # elif state == 1 and nextState == 2:
            #     reward = 0.1
            # elif state == 2 and nextState == 6:
            #     reward = 0.1
            # elif state == 6 and nextState == 10:
            #     reward = 0.1
            # elif state == 10 and nextState == 14:
            #     reward = 0.1
            # elif state == 14 and nextState == 15:
            #     reward = 1

            reward -= 0.01
            # if state == nextState:
            #     reward = -1

            # if state in states:
            #     reward -= 0.01
            # else:
            #     states.append(state)
            if nextState == 15:
                wins += 1

            if done:
                steps.append(reward)

            episode.append((oneHotState, oneHotAction,
                            reward, oneHotNextState, done))
            steps.append("{}, {}, {}".format(
                state, np.argmax(prediction), reward))
            state = nextState
        experienceBag.addEpisode(episode)
        criticBag.addEpisode(episode)
        print(steps)
        print("------------------------------------")
    # print("Ep")
    printAllStateValuesGrid()
    print("Ep {}, wins {}".format(count, wins))
    if len(criticBag.experiences) > 100:
        allFirstSteps = criticBag.experiences[0].getAllSteps()
        if allFirstSteps[len(allFirstSteps) - 1].reward > 2:
            first = criticBag.experiences.pop(0)
            criticBag.experiences.append(first)
        criticBag.experiences = criticBag.experiences[1:]
    trainer = ModelTrainer(actorModel, criticModel, experienceBag, criticBag)
    trainer.trainActorCritic(10, 16)
