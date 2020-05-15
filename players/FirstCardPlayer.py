from players.Player import Player
from cards.Suite import Suite
import cards.CardUtils as CardUtils


class FirstCardPlayer(Player):
    def __init__(self):
        super().__init__()

    def selectCard(self, state, env):
        validCards = sorted(env.getValidCards(), key=self.heartsLastSort)
        return validCards[len(validCards) - 1]

    def heartsLastSort(self, card):
        suite, _ = CardUtils.indexToSuiteAndRank(card)
        return card + 1000 if suite == Suite.HEARTS else card
