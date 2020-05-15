import cards.CardUtils as CardUtils


class Player:
    def __init__(self):
        self.cards = []
        self.cardsOnTable = []
        self.cardsPlayed = []
        self.steps = []

    def setCards(self, cards):
        self.cards = cards
        self.cardsOnTable = []
        self.cardsPlayed = []

    def pickCard(self, playerIndex, environment):
        if len(self.cards) == 0:
            print(
                "Player {} attempt to pick a card from an empty hand".format(
                    playerIndex
                )
            )
            raise ValueError
        return self.cards[0]

    def play(self, card):
        if card in self.cards:
            self.cardsOnTable.append(card)
            self.cards.remove(card)
        else:
            print("Tried playing {}, but this card is not in hand".format(card))
            raise ValueError

    def clearCardsOnTable(self):
        self.cardsPlayed += self.cardsOnTable
        self.cardsOnTable = []

    def takeAction(self, state, env):
        done = False
        while not done:
            action = self.selectCard(state, env)
            nextState, reward, done, env = yield action
            self.steps.append((state, action, reward, nextState, done))
            state = nextState

        yield None

    def selectCard(self, state, env):
        cardsInHand = state[-52:]
        cardsInHand = self.oneHotToCardNumbers(cardsInHand)
        return cardsInHand[0]

    def oneHotToCardNumbers(self, allCards):
        cards = []
        for i in range(len(allCards)):
            if allCards[i] == 1:
                cards.append(i)
        return cards
