from pacai.agents.capture.capture import CaptureAgent
import random

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.dummyAgent',
        second = 'pacai.student.myTeam.dummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        dummyAgent(firstIndex),
        dummyAgent(secondIndex)
    ]

class dummyAgent(CaptureAgent):

    def __init__(self, index, timeForComputing = 0.1, **kwargs):
        super().__init__(index, timeForComputing, **kwargs)

    def chooseAction(self, gameState):
        """
        Randomly pick an action.
        """

        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)
