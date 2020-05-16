from trajectory_utils.Step import Step


class ExperienceBag:
    def __init__(self):
        self.experiences = []

    # episode is a tuple array of the form (state, action, reward, nextState, done)
    def addEpisode(self, episode):
        step = None
        firstStep = None
        for experienceStep in episode:
            state, action, reward, nextState, done = experienceStep
            nextStep = Step(state, action, reward, nextState, done)
            if step != None:
                step.nextStep = nextStep
                nextStep.previousStep = step
            else:
                firstStep = nextStep
            step = nextStep
        self.experiences.append(firstStep)

    def stepCount(self):
        return len(self.getAllSteps())

    def getAllSteps(self):
        allSteps = []
        for experience in self.experiences:
            allSteps += experience.getAllSteps()
        return allSteps

    def getStates(self, steps):
        return list(map(lambda x: x.state, steps))

    def getPreviousActionProbabilities(self, steps):
        return list(map(lambda x: x.previousActionProbabilities, steps))

    def getAdvantages(self, steps, stateValuePredictor, gamma):
        return list(map(lambda x: [x.getAdvantageValue(1, gamma, stateValuePredictor)], steps))

    def getStateValues(self, steps, stateValuePredictor, gamma):
        return list(map(lambda x: [x.getStateValue(1, gamma, stateValuePredictor)], steps))
