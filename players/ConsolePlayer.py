import cards.CardUtils as CardUtils
from players.Player import Player


class ConsolePlayer(Player):
    def __init__(self):
        super().__init__()

    def selectCard(self, state, env):
        print("~~~~~~~~~~~~~~~ Your Turn! ~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Valid cards:")
        for card in env.getValidCards():
            suite, rank = CardUtils.indexToSuiteAndRank(card)
            print("{} - {} of {}".format(card, rank, suite))
        return int(input())
