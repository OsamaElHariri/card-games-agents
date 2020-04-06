class RoundSummary:
    def __init__(self, env, cardsPlayed, roundWinner, gameEnded):
        self.env = env
        self.roundCount = env.roundCount
        self.startingPlayer = env.playerInControl
        self.cardsPlayed = cardsPlayed
        self.roundWinner = roundWinner
        self.gameEnded = gameEnded

        self.playersCardsOnTable = [[]] * 4
        self.playersCardsPlayed = [[]] * 4
        for player in env.players:
            self.playersCardsOnTable = player.cardsOnTable
            self.playersCardsPlayed = player.cardsPlayed

    def getStateAtPlayerTurn(self, playerIndex):

        for i in range(self.env.playerCount):
            playerIndex = (i + self.env.playerInControl) % self.env.playerCount
