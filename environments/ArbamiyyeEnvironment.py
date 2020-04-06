from environments.CardGameEnvironment import CardGameEnvironment
from environments.RoundSummary import RoundSummary
from cards.Suite import Suite
import cards.CardUtils as CardUtils


class ArbamiyyeEnvironment(CardGameEnvironment):
    def __init__(self, players, cardsForEachPlayer):
        super().__init__(players, cardsForEachPlayer)
        self.topSuite = Suite.HEARTS
        self.playerTricks = [[]] * self.playerCount
        self.roundSummaries = []

    async def playGame(self):
        print("Playing Arbamiyye")
        for _ in range(13):
            await self.playRound()

        winningPlayer, highestTrickCount = self.getGameWinner()
        print("Game Ended. Player {} won with {} tricks".format(
            winningPlayer, highestTrickCount))
        self.done = True

    def getValidCards(self):
        cards = []
        for card in self.players[self.currentPlayer].cards:
            if self.validCardPlay(card):
                cards.append(card)
        return cards

    def validCardPlay(self, card):
        if self.currentPlayer == self.playerInControl:
            return True

        firstCardPlayed = self.players[self.playerInControl].cardsOnTable[0]
        suite, _ = CardUtils.indexToSuiteAndRank(card)
        roundSuite, _ = CardUtils.indexToSuiteAndRank(firstCardPlayed)
        isHoldingSuite = self.playerIsHoldingSuite(
            self.currentPlayer, roundSuite)

        if isHoldingSuite and suite != roundSuite:
            return False
        else:
            return True

    def playerIsHoldingSuite(self, playerIndex, searchSuite):
        for card in self.players[playerIndex].cards:
            suite, _ = CardUtils.indexToSuiteAndRank(card)
            if suite == searchSuite:
                return True
        return False

    def onRoundEnd(self):
        cardsPlayed = []
        for player in self.players:
            cardsPlayed.append(player.cardsOnTable[0])

        roundWinner = self.getStrongestCardIndex(cardsPlayed)
        print("Round Winner = {}".format(roundWinner))
        super().onRoundEnd()

        summary = RoundSummary(
            self, cardsPlayed, roundWinner, self.roundCount == 12)
        self.roundSummaries.append(summary)

        for i in range(self.playerCount):
            self.playerTricks[roundWinner].append(
                cardsPlayed if i == roundWinner else [])

        self.playerInControl = roundWinner

    def getGameWinner(self):
        highestTrickCount = -1
        winningPlayer = -1
        for i in range(self.playerCount):
            trickCount = self.getTricksWon(i)
            if trickCount > highestTrickCount:
                winningPlayer = i
                highestTrickCount = trickCount
        return winningPlayer, highestTrickCount

    def getTricksWon(self, playerIndex):
        trickCount = 0
        for summary in self.roundSummaries:
            if summary.roundWinner == playerIndex:
                trickCount += 1
        return trickCount

    def getStrongestCardIndex(self, cards):
        firstCardPlayed = cards[self.playerInControl]
        roundSuite, _ = CardUtils.indexToSuiteAndRank(firstCardPlayed)

        bestCardIndex = self.playerInControl
        bestCard = firstCardPlayed

        for i in range(self.playerCount - 1):
            playerIndex = (i + self.playerInControl + 1) % self.playerCount
            betterCard = self.getBetterCard(
                roundSuite, bestCard, cards[playerIndex])

            if betterCard != bestCard:
                bestCardIndex = playerIndex

            bestCard = betterCard

        return bestCardIndex

    def getBetterCard(self, roundSuite, card1, card2):
        card1Suite, card1Rank = CardUtils.indexToSuiteAndRank(card1)
        card2Suite, card2Rank = CardUtils.indexToSuiteAndRank(card2)

        if card1Suite == card2Suite:
            return card1 if card1Rank.value > card2Rank.value else card2
        else:
            if card1Suite == self.topSuite:
                return card1
            elif card2Suite == self.topSuite:
                return card2
            elif card1Suite == roundSuite:
                return card1
            elif card2Suite == roundSuite:
                return card2
            else:
                return card1 if card1Rank.value > card2Rank.value else card2
