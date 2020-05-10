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
