
from environments.ArbamiyyeEnvironment import ArbamiyyeEnvironment
from players.ConsolePlayer import ConsolePlayer
from players.FirstCardPlayer import FirstCardPlayer
from players.PredictorPlayer import PredictorPlayer
from trajectory_utils.ModelFactory import ModelFactory
import cards.CardUtils as CardUtils
import asyncio


model = ModelFactory().getActorModel(4 + 52 * 4 + 52 * 3 + 52, 52)
players = [PredictorPlayer(model), FirstCardPlayer(),
           FirstCardPlayer(), FirstCardPlayer()]
# players = [ConsolePlayer(), FirstCardPlayer(),
#            FirstCardPlayer(), FirstCardPlayer()]
env = ArbamiyyeEnvironment(players, 13)
asyncio.run(env.playGame())
