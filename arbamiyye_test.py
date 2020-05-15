
from environments.ArbamiyyeEnvironment import ArbamiyyeEnvironment
from players.ConsolePlayer import ConsolePlayer
from players.FirstCardPlayer import FirstCardPlayer
from players.PredictorPlayer import PredictorPlayer
from trajectory_utils.ModelFactory import ModelFactory
from trajectory_utils.ExperienceBag import ExperienceBag
from trajectory_utils.ModelTrainer import ModelTrainer
import cards.CardUtils as CardUtils


model = ModelFactory().getActorModel(4 + 52 * 4 + 52 * 3 + 52, 52)
stateModel = ModelFactory().getCriticModel(4 + 52 * 4 + 52 * 3 + 52)
players = [PredictorPlayer(model, stateModel), PredictorPlayer(model, stateModel),
           PredictorPlayer(model, stateModel), PredictorPlayer(model, stateModel)]
# players = [ConsolePlayer(), FirstCardPlayer(),
#            FirstCardPlayer(), FirstCardPlayer()]
env = ArbamiyyeEnvironment(players, 13)
env.playGame()

experiences = ExperienceBag()
for player in players:
    experiences.addEpisode(player.steps)

step = experiences.experiences[0]

while step != None:
    print("Action {} reward {}".format(step.action, step.reward))
    step = step.nextStep

trainer = ModelTrainer(model, stateModel, experiences)

trainer.trainActorCritic(32, 16)
