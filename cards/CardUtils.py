from cards.Rank import Rank
from cards.Suite import Suite


def indexToSuiteAndRank(index):
    suiteIndex = index // len(Rank)
    rankIndex = index % len(Rank)
    return Suite(suiteIndex), Rank(rankIndex)


def suiteRankToIndex(suite, rank):
    suiteIndex = suite.value
    rankIndex = rank.value
    return suiteIndex * len(Rank) + rankIndex
