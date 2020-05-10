import cards.CardUtils as CardUtils
from players.Player import Player
import numpy as np


class PredictorPlayer(Player):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    async def selectCard(self, state, env):
        predictions = self.predictor.predict([state])
        print("predictions")
        action = np.random.choice(len(predictions), p=predictions)

        print(action)
        print(predictions)
        for card in env.getValidCards():
            suite, rank = CardUtils.indexToSuiteAndRank(card)
            print("{} - {} of {}".format(card, rank, suite))
        return int(input())
