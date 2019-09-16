import abc
import copy

from pacai.core import game

INITIAL_HASH_VALUE = 17
HASH_MULTIPLIER = 37

class AbstractGameState(abc.ABC):
    """
    A GameState specifies the full game state, including the food, capsules,
    agent configurations, and score changes.

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.

    Only use the accessor methods to get data about the game state.
    """

    def __init__(self, layout):
        self._lastAgentMoved = None
        self._gameover = False
        self._win = False

        self._layout = layout

        # For food and capsules, we will only copy on write (if we eat one of them).
        # This avoid additional copies on successors that don't eat.

        self._foodCopied = False
        self._food = layout.food.copy()
        self._lastFoodEaten = None

        self._capsulesCopied = False
        self._capsules = layout.capsules.copy()
        self._lastCapsuleEaten = None

        self._agentStates = []
        for (isPacman, position) in layout.agentPositions:
            self._agentStates.append(game.AgentState(game.Configuration(position, game.Directions.STOP), isPacman))

        self._score = 0

    @abc.abstractmethod
    def generateSuccessor(self, agentIndex, action):
        """
        Returns the successor state after the specified agent takes the action.
        Treat the returned state as a SHALLOW copy that has been modified.
        """

        pass

    @abc.abstractmethod
    def getLegalActions(self, agentIndex = 0):
        """
        Gets the legal actions for the agent specified.
        """

        pass

    def addScore(self, score):
        self._score += score

    def eatCapsule(self, x, y):
        """
        Mark the capsule at the given location as eaten.
        """

        if (not self.hasCapsule(x, y)):
            return

        if (not self._capsulesCopied):
            self._capsules = self._capsules.copy()
            self._capsulesCopied = True

        self._capsules.remove((x, y))
        self._lastCapsuleEaten = (x, y)

    def eatFood(self, x, y):
        """
        Mark the food at the given location as eaten.
        """

        if (not self.hasFood(x, y)):
            return

        if (not self._foodCopied):
            self._food = self._food.copy()
            self._foodCopied = True

        self._food[x][y] = False
        self._lastFoodEaten = (x, y)

    def endGame(self, win):
        self._gameover = True
        self._win = win

    def getAgentPosition(self, index):
        """
        Returns a location tuple of the agent with the given index.
        It is possible for this method to return None if the agent's position is unknown
        (like if it just died and is respawning).
        """

        position = self._agentStates[index].getPosition()
        if (position is None):
            return None

        # Ensure positions are ints.
        return tuple(int(pos) for pos in position)

    def getAgentState(self, index):
        return self._agentStates[index]

    def getAgentStates(self):
        return self._agentStates

    def getCapsules(self):
        """
        Returns a list of positions (x, y) of the remaining capsules.
        """

        return self._capsules

    def getFood(self):
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation.
        So to check if there is food at (x, y), just do something like: food[x][y].

        Callers should favor hasFood() over this, since this will make a copy of the grid.
        """

        return self._food.copy()

    def getInitialAgentPosition(self, agentIndex):
        return self._layout.agentPositions[agentIndex][1]

    def getInitialLayout(self):
        """
        Get the initial layout this state starte with.
        User's should typically call one of the more detailed methods directly,
        e.g. getWalls().
        """

        return self._layout

    def getLastAgentMoved(self):
        return self._lastAgentMoved

    def getLastCapsuleEaten(self):
        return self._lastCapsuleEaten

    def getLastFoodEaten(self):
        return self._lastFoodEaten

    def getNumAgents(self):
        return len(self._agentStates)

    def getNumCapsules(self):
        """
        Get the amount of capsules left on the board.
        """

        return len(self._capsules)

    def getNumFood(self):
        """
        Get the amount of food left on the board.
        """

        return self._food.count()

    def getScore(self):
        return self._score

    def getWalls(self):
        """
        Returns a Grid of boolean wall indicator variables.

        Grids can be accessed via list notation.
        So to check if there is a wall at (x, y), just do something like: walls[x][y].

        The caller should not try to modify the walls.
        """

        return self._layout.walls

    def hasCapsule(self, x, y):
        """
        Returns true if the location (x, y) has a capsule.
        """

        return (x, y) in self._capsules

    def hasFood(self, x, y):
        """
        Returns true if the location (x, y) has food.
        """

        return self._food[x][y]

    def hasWall(self, x, y):
        """
        Returns true if (x, y) has a wall, false otherwise.
        """

        return self._layout.walls[x][y]

    def isLose(self):
        return self.isOver() and not self._win

    def isOver(self):
        return self._gameover

    def isWin(self):
        return self.isOver() and self._win

    def _initSuccessor(self):
        """
        Get a state that will eventually serve as a successor.
        Initialize the successor to look like this state.
        """

        # Start with a shallow copy.
        successor = copy.copy(self)

        # Leave food and capsules as a shallow copy, but mark them to be copied on write.
        successor._foodCopied = False
        successor._capsuleCopied = False

        # Agent states need to be deep copied.
        successor._agentStates = [agentState.copy() for agentState in self._agentStates]

        return successor

    def __eq__(self, other):
        if (other is None):
            return False

        # Reference equality check.
        if (self is other):
            return True

        if (type(self) != type(other)):
            return False

        # Note that not all fields are being used because we are checking if two states are equal,
        # not is they got to this confiruation in the same way.

        # Check simple fields first.
        if (self._score != other._score
                or self._gameover != other._gameover
                or self._win != other._win):
            return False

        # Now check the complex fields in increasing order of complexity.
        return (self._capsules == other._capsules
                and self._food == other._food
                and self._agentStates == other._agentStates
                and self._layout == other._layout)

    def __hash__(self):
        hashCode = INITIAL_HASH_VALUE

        hashCode = hashCode * HASH_MULTIPLIER + self._score
        hashCode = hashCode * HASH_MULTIPLIER + hash(self._gameover)
        hashCode = hashCode * HASH_MULTIPLIER + hash(self._win)

        hashCode = hashCode * HASH_MULTIPLIER + hash(self._capsules)
        hashCode = hashCode * HASH_MULTIPLIER + hash(self._food)
        hashCode = hashCode * HASH_MULTIPLIER + hash(self._agentStates)
        hashCode = hashCode * HASH_MULTIPLIER + hash(self._layout)

        return int(hashCode)