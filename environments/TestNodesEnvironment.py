# This class represents a simple environment where there are 2 actions
# and numberOfNodes nodes. Conceptually, the nodes are placed next to each other horizontally,
# the agent gets a positive reward when it reaches the rightmost node, and a negative reward
# when it reaches the leftmost node.
# The agent starts at the center node


class TestNodesEnvironment:
    def __init__(self, numberOfNodes):
        self.numberOfNodes = numberOfNodes
        self.state = self.numberOfNodes // 2

    def reset(self):
        self.state = self.numberOfNodes // 2
        return self.state

    def step(self, action):
        self.state += 1 if action > 0 else -1
        nextState = self.state

        reward = 0
        done = False
        if self.state == self.numberOfNodes - 1:
            reward = 1
            done = True
        if self.state == 0:
            reward = -1
            done = True

        return nextState, reward, done
