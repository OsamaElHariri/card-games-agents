import tensorflow as tf


class Step:
    def __init__(self, state, action, reward, nextState, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.done = done

        self.nextStep = None
        self.previousStep = None

    # Get all the steps, including this one, as an array
    def getAllSteps(self):
        steps = [self]
        if self.nextStep != None:
            steps += self.nextStep.getAllSteps()
        return steps

    # Get the TD advantage for this step A = R_t+1 + GAMMA * V(S_t+1) - V(S)
    def getAdvantageValue(self, lookAheadCount, gamma, stateValuePredictor):
        return self.getStateValue(
            lookAheadCount, gamma, stateValuePredictor
        ) - stateValuePredictor(self.arrayToTFMatrix([self.state]))

    # recursively get the state value V(S_t) using TD Lambda, where lambda is the variable called lookAheadCount
    # the state value is the reward if this is the terminal state (the state the episode ends on)
    # otherwise, the V(S_t) = R_t+1 + GAMMA * V(S_t+1)
    def getStateValue(self, lookAheadCount, gamma, stateValuePredictor):
        if self.done or self.nextStep == None:
            return self.reward
        elif lookAheadCount > 0:
            return self.reward + gamma * self.nextStep.getStateValue(
                lookAheadCount - 1, gamma, stateValuePredictor
            )
        else:
            return stateValuePredictor(self.arrayToTFMatrix([self.state]))
    
    def arrayToTFMatrix(self, array):
        return tf.reshape(array, [len(array), len(array[0])])
