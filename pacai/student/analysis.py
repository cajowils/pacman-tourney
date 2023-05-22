"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    Drop the noise to 0 to make the agent more confident about walking in between the cliffs.
    """

    answerDiscount = 0.9
    answerNoise = 0.0

    return answerDiscount, answerNoise

def question3a():
    """
    By making the living reward negative, the agent will try to get to a
    terminal state as fast as possible. -2.0 is the lowest value that
    will make the agent go to the closest pit while also risking the cliff.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -2.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    We can lower the discount rate to make the agent care less about the future,
    encouraging it to take the shorter path. We need to keep the living reward negative
    to make it go to a terminal state quicker.
    """

    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = -1.5

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Similar to 3a, we need a negative living reward to make the agent risk the cliff.
    However, we need it to go the the further pit this time, so we need a higher living reward.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    We can make the living reward less negative to encourage the agent to
    take the longer path to the further terminal state.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    By making the living reward arbitrarily positive, the agent will try to stay
    alive as long as possible, avoiding terminal states such as the cliff and rewards.
    """

    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = 100

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    50 iterations is not enough for the agent to learn the optimal policy
    for any epsilon-learning rate combination.
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
