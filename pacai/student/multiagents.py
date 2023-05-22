import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        eval = 0

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        oldFoodList = currentGameState.getFood().asList()
        newFoodList = successorGameState.getFood().asList()

        newGhostStates = successorGameState.getGhostStates()

        # Find the distance to the closest ghost
        for state in newGhostStates:
            closest = float('inf')
            dist = distance.manhattan(state.getPosition(), newPosition)
            # If the state is losing, return a very low score
            if dist <= 1 and state.getScaredTimer() <= 1:
                return -999999
            closest = min(closest, dist)

        # Factor in the distance to the closest ghost
        if closest != 0 and closest != float('inf'):
            eval -= (1 / closest)

        # If the state is winning, return an very high score
        if len(newFoodList) == 0 or len(newFoodList) < len(oldFoodList):
            return 999999

        # Factor in the distance to the closest food
        eval += (1 / min(distance.manhattan(newPosition, food) for food in newFoodList))

        return eval

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        return self.minimax(gameState)

    def minimax(self, gameState):
        """
        Uses minimax algorithm to return the best possible action for the first agent
        within a given tree depth
        """
        return self.maxValue(gameState, self.getTreeDepth())

    def maxValue(self, gameState, treeDepth):
        """
        Finds the maximum value of the current state's possible actions
        """
        v = float('-inf')

        if self.terminalNode(gameState, treeDepth):
            return self.getEvaluationFunction()(gameState)

        bestAction = None
        for action in gameState.getLegalActions(0):
            newV = self.minValue(gameState.generateSuccessor(0, action), treeDepth)
            if newV > v:
                v = newV
                bestAction = action

        return bestAction if treeDepth == self.getTreeDepth() else v

    def minValue(self, gameState, treeDepth, agentIndex = 1):
        """
        Returns the minimum tree values for each min agent
        """

        v = float('inf')

        if self.terminalNode(gameState, treeDepth):
            return self.getEvaluationFunction()(gameState)

        # If current agent is the last, switch to max agent
        if agentIndex == gameState.getNumAgents() - 1:
            return self.maxValue(gameState, treeDepth - 1)
        for action in gameState.getLegalActions(agentIndex):
            v = min(v, self.minValue(gameState.generateSuccessor(agentIndex, action),
                                     treeDepth, agentIndex + 1))
        return v

    def terminalNode(self, gameState, treeDepth):
        """
        Checks whether a given state is a terminal node
        """
        return gameState.isWin() or gameState.isLose() or treeDepth == 0

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        return self.alphaBeta(gameState)

    def alphaBeta(self, gameState):
        """
        Uses minimax algorithm with alpha-beta pruning to return the
        best possible action for the first agent within a given tree depth
        """
        treeDepth = self.getTreeDepth()

        if self.terminalNode(gameState, treeDepth):
            return self.getEvaluationFunction()(gameState)

        # Initializes v, alpha, and beta
        v = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        bestAction = None
        for action in gameState.getLegalActions(0):
            newV = self.minValue(gameState.generateSuccessor(0, action),
                                 treeDepth, alpha, beta)
            if newV > v:
                v = newV
                bestAction = action
            alpha = max(alpha, v)
        return bestAction

    def maxValue(self, gameState, treeDepth, alpha, beta):
        v = float('-inf')

        if self.terminalNode(gameState, treeDepth):
            return self.getEvaluationFunction()(gameState)

        for action in gameState.getLegalActions(0):
            v = max(v, self.minValue(gameState.generateSuccessor(0, action),
                                     treeDepth, alpha, beta))
            # If v is greater or equal to our beta, we can prune the rest of the subtree
            if v >= beta:
                return v
            # Update alpha
            alpha = max(alpha, v)
        return v

    def minValue(self, gameState, treeDepth, alpha, beta, agentIndex = 1):
        v = float('inf')

        if self.terminalNode(gameState, treeDepth):
            return self.getEvaluationFunction()(gameState)

        if agentIndex == gameState.getNumAgents() - 1:
            return self.maxValue(gameState, treeDepth - 1, alpha, beta)
        for action in gameState.getLegalActions(agentIndex):
            v = min(v, self.minValue(gameState.generateSuccessor(agentIndex, action),
                                     treeDepth, alpha, beta, agentIndex + 1))
            # If v is less than or equal to our alpha, we can prune the rest of the subtree
            if v <= alpha:
                return v
            # Update beta
            beta = min(beta, v)
        return v

    def terminalNode(self, gameState, treeDepth):
        return gameState.isWin() or gameState.isLose() or treeDepth == 0

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        return self.value(gameState, self.getTreeDepth())

    def value(self, gameState, treeDepth, agentIndex=0):
        """
        Returns the value of the best action given an agent's possible actions
        """
        if self.terminalNode(gameState, treeDepth):
            return self.getEvaluationFunction()(gameState)
        if agentIndex == 0:
            action, val = self.maxValue(gameState, treeDepth, agentIndex)
            return action if treeDepth == self.getTreeDepth() else val
        else:
            return self.expValue(gameState, treeDepth, agentIndex)

    def maxValue(self, gameState, treeDepth, agentIndex):
        v = float('-inf')
        bestAction = None
        # Decrements the tree depth if the current agent is the last agent
        if agentIndex == gameState.getNumAgents() - 1:
            treeDepth -= 1
        for action in gameState.getLegalActions(agentIndex):
            newV = self.value(gameState.generateSuccessor(agentIndex, action),
                              treeDepth, (agentIndex + 1) % gameState.getNumAgents())
            if newV > v:
                v = newV
                bestAction = action
        return bestAction, v

    def expValue(self, gameState, treeDepth, agentIndex):
        v = 0
        # Decrements the tree depth if the current agent is the last agent
        if agentIndex == gameState.getNumAgents() - 1:
            treeDepth -= 1
        for action in gameState.getLegalActions(agentIndex):
            v += self.value(gameState.generateSuccessor(agentIndex, action),
                            treeDepth, (agentIndex + 1) % gameState.getNumAgents())
        return v / len(gameState.getLegalActions(agentIndex))

    def terminalNode(self, gameState, treeDepth):
        return gameState.isWin() or gameState.isLose() or treeDepth == 0

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: This evaluation function considers the distances to food, ghosts and corners,
    and applies different weights to prioritize Pacman's actions.
    """

    # Useful information you can extract.
    foodList = currentGameState.getFood().asList()
    currentPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # Weights to tweak the behavior of Pacman
    foodWeight = 1.0
    ghostWeight = 10.0
    capsuleWeight = 1.5
    scaredWeight = 1.0

    eval = currentGameState.getScore()

    # Factor in the distance to the ghosts and their scared times
    for state in ghostStates:
        dist = distance.manhattan(state.getPosition(), currentPosition)
        scaredTime = state.getScaredTimer()
        if dist > 2:
            if scaredTime > dist:
                eval += ghostWeight * (1 / dist)
            else:
                eval -= ghostWeight * (1 / dist)
        else:
            return -999999

    # Factor in the ghost with the smallest scared time
    eval += scaredWeight * min(ghost.getScaredTimer() for ghost in ghostStates)

    # Factor in the distance to the closest fooda
    if len(foodList) > 0:
        minFoodDist = min(distance.manhattan(currentPosition, food) for food in foodList)
        eval += foodWeight * (1 / minFoodDist)
    else:
        return 999999

        # Factor in the distance to the capsules
    if len(capsules) > 0:
        minCapsuleDist = min(distance.manhattan(currentPosition, capsule) for capsule in capsules)
        eval += capsuleWeight * (1 / minCapsuleDist)

    return eval

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
