from trajectory_utils.Step import Step


class ExperienceBag:
    def __init__(self, stateValuePredictor, actionValuePredictor):
        self.experiences = []
        self.stateValuePredictor = stateValuePredictor
        self.actionValuePredictor = actionValuePredictor

    # episode is a tuple array of the form (state, action, reward, nextState, done)
    def addEpisode(self, episode):
        step = None
        firstStep = None
        for step in episode:
            nextStep = Step(step)
            if step != None:
                step.nextStep = nextStep
                nextStep.previousStep = step
            else:
                firstStep = nextStep
            step = nextStep
        self.experiences.append(firstStep)
