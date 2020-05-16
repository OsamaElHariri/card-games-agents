from environments.ArbamiyyeEnvironment import ArbamiyyeEnvironment
from players.PredictorPlayer import PredictorPlayer
from players.FirstCardPlayer import FirstCardPlayer
from trajectory_utils.ExperienceBag import ExperienceBag


class ExperienceCollector:

    def __init__(self, actorModel, criticModel):
        self.actorModel = actorModel
        self.criticModel = criticModel
        self.won = 0
        self.tricks = []

    def collect(self, numberOfGames):
        experienceBag = ExperienceBag()
        for _ in range(numberOfGames):
            experienceBag.addEpisode(self.collectOne())
        if len(self.tricks) <= 0:
            print("Player 0 won ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {}".format(
                self.won))
        else:
            print("Player 0 won ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {}, avg {}".format(
                self.won, sum(self.tricks) / len(self.tricks)))
        return experienceBag

    def collectOne(self):
        players = [PredictorPlayer(self.actorModel, self.criticModel), FirstCardPlayer(),
                   FirstCardPlayer(), FirstCardPlayer()]
        env = ArbamiyyeEnvironment(players, 13)
        player, trickCount = env.playGame()
        if player == 0:
            self.won += 1
            self.tricks.append(trickCount)
        return players[0].steps
