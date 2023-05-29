from pacai.util import reflection
# from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions


def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.OffensiveAgent',
        second = 'pacai.agents.capture.defense.DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = reflection.qualifiedImport(first)
    secondAgent = reflection.qualifiedImport(second)

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex)
    ]

class OffensiveAgent(ReflexCaptureAgent):
    """
    CREDIT: Part of this code comes from:
        pacai.agents.capture.offense.OffensiveReflexiveAgent
        pacai.agents.capture.defense.DefensiveRefelexiveAgent
    The OffensiveAgent inherits from the given ReflexiveCaptureAgent and utilizes additional
    features for a more effective OffensiveAgent.

    DESCRIPTION: We used the following features for our implementation of getFeatures:
        1. successorScore - The score of the successor, so we can see given an action whether or
            not our score decrease/increases and this makes the OffensiveAgent more inclined to
            take food.
        2. remainingFood - The amount of food left in the game.
        3. distanceToFood - Minimum distance to a food.
        4. distanceToCapsules - Minimum distance to a capsule.
        5. stop - The stop feature puts a negative weight to Directions.STOP action.
        6. reverse - Negative weight to moving back to previous position.
        7. normalValue - Value obtained from enemy defenders that are not scared, weighted
            positively if they are farther away.
        8. scaredValue - Value obtained from enemy defenders that are scared, weighted
            positively if they are closer.
        9. invaderValue - Minimum distance to enemy invaders, weighted less than defense
            but still here in case our offense agent is close enough to their attacker
        10. numInvaders - Included so pacman will attack enemy invader if next to it.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['remainingFood'] = len(foodList)
            features['distanceToFood'] = minDistance

        # Discourage stop action
        if (action == Directions.STOP):
            features['stop'] = 1

        # Discourage moving backwards
        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        # Calculate the distance to the closest capsule
        capsuleList = self.getCapsules(gameState)
        minCapsule = -100000
        if capsuleList:
            minCapsule = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])

        # Set feature value for closest capsule if capsule exists in game
        if minCapsule != -100000:
            if minCapsule == 0:
                features['distanceToCapsules'] = -100
            else:
                features['distanceToCapsules'] = minCapsule

        # Initialize variables for computing defenders and invaders feature
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        normalVal = 0
        scaredVal = 0
        scared = []
        normal = []

        if defenders:
            # Append to normal and scared the normal or scared defending agents
            for i in defenders:
                if i.getScaredTimer() != 0:
                    scared.append(i)
                else:
                    normal.append(i)

        # Calculate feature for normal (not scared) enemy defending agents
        if normal:
            distsNormal = [self.getMazeDistance(myPos, a.getPosition()) for a in normal]
            normalVal = min(distsNormal)
            features['normalValue'] = normalVal
            if normalVal <= 1:
                features['normalValue'] = -100000

        # Calculate minimum distance to scared enemy defending agents
        if scared:
            distsScared = [self.getMazeDistance(myPos, a.getPosition()) for a in scared]
            scaredVal = min(distsScared)

            # Update scaredValue if fits requirements
            if scaredVal == 0:
                features['scaredValue'] = 5

        # Calculate the invaderValue
        if invaders:
            distsInvaders = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderValue'] = min(distsInvaders)

        # Include numInvaders so it would kill enemy pacman when they're invading
        features['numInvaders'] = len(invaders)

        return features

    def getWeights(self, gameState, action):
        # To silence warnings
        gameState = gameState
        action = action

        # return weights
        return {
            'successorScore': 100,
            'invaderValue': -50,
            'numInvaders': -1000,
            'distanceToFood': -1,
            'remainingFood': -1,
            'distanceToCapsules': -2,
            'stop': -100,
            'reverse': -2,
            'normalValue': 2,
            'scaredValue': 2
        }
