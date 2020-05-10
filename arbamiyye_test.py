
from environments.ArbamiyyeEnvironment import ArbamiyyeEnvironment
from players.ConsolePlayer import ConsolePlayer
from players.FirstCardPlayer import FirstCardPlayer
from players.PredictorPlayer import PredictorPlayer
from trajectory_utils.ModelFactory import ModelFactory
from trajectory_utils.ExperienceBag import ExperienceBag
import cards.CardUtils as CardUtils
import asyncio


model = ModelFactory().getActorModel(4 + 52 * 4 + 52 * 3 + 52, 52)
players = [PredictorPlayer(model), PredictorPlayer(model),
           PredictorPlayer(model), PredictorPlayer(model)]
# players = [ConsolePlayer(), FirstCardPlayer(),
#            FirstCardPlayer(), FirstCardPlayer()]
env = ArbamiyyeEnvironment(players, 13)
asyncio.run(env.playGame())

experiences = ExperienceBag()
for player in players:
    experiences.addEpisode(player.steps)

step = experiences.experiences[0]

while step != None:
    print("Action {} reward {}".format(step.action, step.reward))
    step = step.nextStep
