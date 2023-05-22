from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Initialize each state's value to be 0.
        for state in mdp.getStates():
            self.values[state] = 0.0

        # Perform value iteration.
        for _ in range(iters):
            temp = {}
            for state in mdp.getStates():
                actions = mdp.getPossibleActions(state)
                if len(actions) > 0:
                    temp[state] = max([self.getQValue(state, action) for action in actions])
            self.values = temp

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

    def getQValue(self, state, action):
        """
        Returns the q-value of the state action pair.
        """

        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        return sum([prob * (self.mdp.getReward(state, action, nextState)
            + self.discountRate * self.getValue(nextState)) for nextState, prob in transitions])

    def getPolicy(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        actions = self.mdp.getPossibleActions(state)
        if not len(actions):
            return None
        return max(actions, key = lambda action: self.getQValue(state, action))
