from pacai.util import reflection

# from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="pacai.student.myTeam.OffensiveAgent",
    second="pacai.student.myTeam.DefensiveAgent",
):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = reflection.qualifiedImport(first)
    secondAgent = reflection.qualifiedImport(second)

    return [firstAgent(firstIndex), secondAgent(secondIndex)]


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
        features["successorScore"] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features["remainingFood"] = len(foodList)
            features["distanceToFood"] = minDistance
            features["densityFood"] = self.calculateDensityValue(successor, False)
        # Discourage stop action
        if action == Directions.STOP:
            features["stop"] = 1

        # Discourage moving backwards
        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if action == rev:
            features["reverse"] = 1

        # Calculate the distance to the closest capsule
        capsuleList = self.getCapsules(gameState)
        minCapsule = -100000
        if capsuleList:
            minCapsule = min(
                [self.getMazeDistance(myPos, capsule) for capsule in capsuleList]
            )

        # Set feature value for closest capsule if capsule exists in game
        if minCapsule != -100000:
            if minCapsule == 0:
                features["distanceToCapsules"] = -100
            else:
                features["distanceToCapsules"] = minCapsule

        # Initialize variables for computing defenders and invaders feature
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [
            a for a in enemies if not a.isPacman() and a.getPosition() is not None
        ]
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
            features["normalValue"] = normalVal
            if normalVal <= 1:
                features["normalValue"] = -100000

        # Calculate minimum distance to scared enemy defending agents
        if scared:
            distsScared = [self.getMazeDistance(myPos, a.getPosition()) for a in scared]
            scaredVal = min(distsScared)

            # Update scaredValue if fits requirements
            if scaredVal == 0:
                features["scaredValue"] = 1

        # Calculate the invaderValue
        if invaders:
            distsInvaders = [
                self.getMazeDistance(myPos, a.getPosition()) for a in invaders
            ]
            features["invaderValue"] = min(distsInvaders)

        # Include numInvaders so it would kill enemy pacman when they're invading
        features["numInvaders"] = len(invaders)

        return features

    def getDensityDict(self, gameState, d=False):
        # Initialize variables to get the density
        foodGrid = self.getFood(gameState)
        defendFood = self.getFoodYouAreDefending(gameState)
        foodList = self.getFood(gameState).asList()
        densities = {}
        bounds = None

        # Check whether on defense or not
        if d:
            bounds = (foodGrid.getWidth(), foodGrid.getHeight())
        else:
            bounds = (defendFood.getWidth(), defendFood.getHeight())

        # Iterate through each food in the foodList
        for x1, y1 in foodList:
            # Set boundaries
            minX = max(1, x1 - 3)
            maxX = min(bounds[0], x1 + 3)
            minY = max(1, y1 - 3)
            maxY = min(bounds[1], y1 + 3)

            # Iterate through the x and y range and see if food exists
            foodCount = 0
            for x2 in range(minX, maxX):
                for y2 in range(minY, maxY):
                    if x2 < bounds[0] and x2 > 0 and y2 < bounds[1] and y2 > 0:
                        if d and defendFood[x2][y2]:
                            foodCount += 1
                        elif not d and foodGrid[x2][y2]:
                            foodCount += 1

            # Add to densities dictionary
            densities[(x1, y1)] = foodCount

        # Returns a dictionary with (key, value) as (position, foodFound)
        # near the key's position.
        return densities

    def calculateDensityValue(self, gameState, d=False):
        # Get densities dictionary, position, and initialize returned value
        densities = self.getDensityDict(gameState, d)
        densityCalculation = 0
        myPos = gameState.getAgentState(self.index).getPosition()

        # Iterate through densitity keys and multiply by distance from agent
        for key in densities.keys():
            densityCalculation += self.getMazeDistance(myPos, key) * densities[key]
        densityCalculation = 1.0 / densityCalculation

        # Return accumulated variable
        return densityCalculation

    def getWeights(self, gameState, action):
        # To silence warnings
        gameState = gameState
        action = action

        # return weights
        return {
            "successorScore": 200,
            "invaderValue": -50,
            "numInvaders": -1000,
            "distanceToFood": -10,
            "remainingFood": -1,
            "distanceToCapsules": -2,
            "stop": -100,
            "reverse": -20,
            "normalValue": 2,
            "scaredValue": 100000,
            "densityFood": 300,
        }


class DefensiveAgent(ReflexCaptureAgent):
    """
    CREDIT: Part of this code comes from:
        pacai.agents.capture.offense.OffensiveReflexiveAgent
        pacai.agents.capture.defense.DefensiveRefelexiveAgent
    The DefensiveAgent inherits from the given ReflexiveCaptureAgent and utilizes additional
    features for a more effective DefensiveAgent.

    DESCRIPTION: We used the following features for our implementation of getFeatures:
        1. numInvaders - Checking whether there are invaders or not
        2. onDefense - Check whether pacman is on defense or not on defense
        3. invaderDistance - Check distance to invaders, getting closer yields a higher value
        4. stop - Discourage stopping
        5. reverse - Discourage reversing, i.e. moving back in the same direction
        6. edge - The edges that the agent should stick near, this is a value somewhere towards
        the middle of the board
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)

        # Initialize walls
        walls = gameState.getWalls().asList()

        # Find x coordinate edge
        x = max([w[0] for w in walls]) / 2
        if not self.red:
            x += 2

        # Create y coordinate edge
        topY = max([w[1] for w in walls])
        upperEdge = topY * 2 / 3
        lowerEdge = topY / 3

        self.edges = [(x, upperEdge), (x, lowerEdge)]

        # False when travelling to bottom, True when travelling top
        self.edge = False

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features["onDefense"] = 1
        if myState.isPacman():
            features["onDefense"] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features["numInvaders"] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features["invaderDistance"] = min(dists)
        else:
            if self.edge and myPos == self.edges[0]:
                self.edge = False
                features["edge"] = self.getMazeDistance(myPos, self.edges[0])
            elif not self.edge and myPos == self.edges[1]:
                self.edge = True
                features["edge"] = self.getMazeDistance(myPos, self.edges[1])

        if action == Directions.STOP:
            features["stop"] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if action == rev:
            features["reverse"] = 1

        # return features
        return features

    def getWeights(self, gameState, action):
        # To silence warnings
        gameState = gameState
        action = action

        # return weights
        return {
            "numInvaders": -1000,
            "onDefense": 100,
            "invaderDistance": -200,
            "stop": -100,
            "reverse": -10,
            "edge": -10,
        }
