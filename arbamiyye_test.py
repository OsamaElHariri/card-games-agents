
from environments.ArbamiyyeEnvironment import ArbamiyyeEnvironment
from players.ConsolePlayer import ConsolePlayer
from players.FirstCardPlayer import FirstCardPlayer
import cards.CardUtils as CardUtils
import asyncio

players = [FirstCardPlayer(), FirstCardPlayer(),
           FirstCardPlayer(), ConsolePlayer()]
env = ArbamiyyeEnvironment(players, 13)
asyncio.run(env.playGame())
