
from environments.ArbamiyyeEnvironment import ArbamiyyeEnvironment
from players.ConsolePlayer import ConsolePlayer
from players.FirstCardPlayer import FirstCardPlayer
from players.PredictorPlayer import PredictorPlayer
from trajectory_utils.ModelFactory import ModelFactory
from trajectory_utils.ExperienceBag import ExperienceBag
from trajectory_utils.ModelTrainer import ModelTrainer
from trajectory_utils.ExperienceCollector import ExperienceCollector
import cards.CardUtils as CardUtils


model = ModelFactory().getActorModel(4 + 52 * 4 + 52 * 3 + 52, 52)
stateModel = ModelFactory().getCriticModel(4 + 52 * 4 + 52 * 3 + 52)

while True:
    collector = ExperienceCollector(model, stateModel)

    experiences = collector.collect(10)
    trainer = ModelTrainer(model, stateModel, experiences)
    trainer.trainActorCritic(40, 16)
