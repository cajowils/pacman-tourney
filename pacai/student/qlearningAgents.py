from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util.probability import flipCoin
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.qValues = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        return self.qValues.get((state, action), 0.0)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """

        return self.getValueActionPair(state)[0]

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """

        return self.getValueActionPair(state)[1]

    def getValueActionPair(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return (0.0, None)

        bestAction = (-float('inf'), None)

        for action in actions:
            val = self.getQValue(state, action)
            if val > bestAction[0]:
                bestAction = (val, action)

        return bestAction

    def getAction(self, state):
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
        if flipCoin(self.getEpsilon()):
            return random.choice(actions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):

        alpha = self.getAlpha()
        discount = self.getDiscountRate()

        # Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (reward + discount * max_a' Q(s', a'))
        self.qValues[(state, action)] = (1 - alpha) * self.getQValue(state, action) + \
            alpha * (reward + discount * self.getValue(nextState))


class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION:
        Uses an approximate q-function to learn the q-values of the game.
        This is acheived by using a feature extractor to extract features from the game state,
        and then learning the weights of the features to calculate the q-value of the state.
        The q-value is calculated by taking the dot product of the weights and the features.
        The weights are updated using the q-learning update rule.
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)
        # You might want to initialize weights here.
        self.weights = {}

    def getWeights(self, feature):
        return self.weights.get(feature, 0.0)

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(self, state, action)

        # Q(s, a) = w * featureVector
        return sum([features[feature] * self.getWeights(feature) for feature in features])

    # used pseudocode from professor on Piazza
    def update(self, state, action, nextState, reward):
        alpha = self.getAlpha()
        discount = self.getDiscountRate()
        features = self.featExtractor.getFeatures(self, state, action)

        newQVal = reward + discount * self.getValue(nextState)
        correction = alpha * (newQVal - self.getQValue(state, action))

        for feature, value in features.items():
            self.weights[feature] = self.getWeights(feature) + correction * value

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            pass
