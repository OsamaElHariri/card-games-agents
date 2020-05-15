import random
from players.Player import Player
import cards.CardUtils as CardUtil


class CardGameEnvironment:
    def __init__(self, players, cardsForEachPlayer):
        self.roundCount = -1
        self.players = players
        self.playerCount = len(players)
        self.cardsForEachPlayer = cardsForEachPlayer
        self.reset()
        self.playerInControl = self.getPlayerInControl()
        self.currentPlayer = self.playerInControl
        self.done = False
        self.playerGenerators = [None] * self.playerCount

    def getPlayerInControl(self):
        return 0

    def playRound(self):
        self.roundCount += 1
        self.printPlayerCards()
        for i in range(self.playerCount):
            playerIndex = (i + self.playerInControl) % self.playerCount
            self.currentPlayer = playerIndex
            player = self.players[playerIndex]

            card = None
            if self.roundCount == 0:
                gen = player.takeAction(self.getState(playerIndex), self)
                self.playerGenerators[playerIndex] = gen
                card = gen.send(None)
            else:
                card = self.playerGenerators[playerIndex].send(
                    (self.getState(playerIndex), 0, False, self)
                )

            if not self.validCardPlay(card):
                print("Player {} played an invalid card {}".format(playerIndex, card))
                raise ValueError
            suite, rank = CardUtil.indexToSuiteAndRank(card)
            print("Player {} played card {} of {}".format(playerIndex, rank, suite))
            player.play(card)
        self.onRoundEnd()

    def validCardPlay(self, card):
        pass

    def onRoundEnd(self):
        for player in self.players:
            player.clearCardsOnTable()

    def reset(self):
        self.cardsOnTable = []
        allCards = [i for i in range(52)]
        random.shuffle(allCards)
        for i in range(self.playerCount):
            cards = allCards[
                i * self.cardsForEachPlayer : (i + 1) * self.cardsForEachPlayer
            ]
            self.players[i].setCards(cards)

    def getPlayerExperiences(self):
        experiences = []
        for i in range(self.playerCount):
            player = self.players[i]
            experiences.append(player.steps)
        return experiences

    # The state for a player at index playerIndex is represented as
    # array of length 4 that represent the starting player
    # four arrays of length 52 that represent the cards already played by each player
    # three arrays of length 52 that represent the cards currently on the ground for the other players
    # array of length 52 that represents the cards currently in the players hand
    #
    # the arrays are ordered starting from playerIndex, then playerIndex + 1, playerIndex + 2...
    def getState(self, playerIndex):
        state = self.getPlayerStartState(playerIndex)
        state += self.getAlreadyPlayedState(playerIndex)
        state += self.getCardsOnTableState(playerIndex)
        state += self.getPlayerHandState(playerIndex)
        return state

    def getPlayerStartState(self, playerIndex):
        state = [0] * self.playerCount
        state[
            (self.playerInControl + self.playerCount - playerIndex) % self.playerCount
        ] = 1
        return state

    def getAlreadyPlayedState(self, playerIndex):
        state = []
        for i in range(self.playerCount):
            cards = self.players[i % self.playerCount].cardsPlayed
            state += self.cardNumbersToOneHot(cards)
        return state

    def getCardsOnTableState(self, playerIndex):
        state = []
        for i in range(self.playerCount - 1):
            cards = self.players[(i + 1) % self.playerCount].cardsOnTable
            state += self.cardNumbersToOneHot(cards)
        return state

    def getPlayerHandState(self, playerIndex):
        return self.cardNumbersToOneHot(self.players[playerIndex].cards)

    def cardNumbersToOneHot(self, cards):
        allCards = [0] * 52
        for card in cards:
            allCards[card] = 1
        return allCards

    def printPlayerCards(self):
        print("Round {}".format(self.roundCount))
        for player in self.players:
            print(player.cards)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
