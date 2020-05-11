import cards.CardUtils as CardUtils
from players.Player import Player
import numpy as np


class PredictorPlayer(Player):
    def __init__(self, actionPredictor, stateValuePredictor):
        super().__init__()
        self.actionPredictor = actionPredictor
        self.stateValuePredictor = stateValuePredictor

    async def selectCard(self, state, env):
        validCards = env.getValidCards()
        predictions = self.actionPredictor.predict([state])[0]
        predictions = self.removeInvalidActions(predictions, validCards)
        action = np.random.choice(len(predictions), p=predictions)
        return action

    # set the probability of selecting an invalid action to 0
    # and normailze the probabilty array so that its sum equals 1
    def removeInvalidActions(self, predictions, validCards):
        for actionIndex in range(len(predictions)):
            if not actionIndex in validCards:
                predictions[actionIndex] = 0
        total = np.sum(predictions)
        for predictionIndex in range(len(predictions)):
            predictions[predictionIndex] = predictions[predictionIndex] / total

        return predictions
