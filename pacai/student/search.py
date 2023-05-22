from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

def genericSearch(problem, dataStructure, heuristic = None):
    """
    A generic search function that takes a problem and a data structure
    and returns a list of actions to reach the goal.
    """

    # If the data structure is a priority queue, we need to push a tuple of (state, cost).
    if isinstance(dataStructure, PriorityQueue):
        dataStructure.push((problem.startingState(), 0), 0)
    else:
        dataStructure.push((problem.startingState(), 0))

    path = {}
    visited = set()

    while not dataStructure.isEmpty():
        state, curr_cost = dataStructure.pop()
        # Once the goal is reached, we can reconstruct the path.
        if problem.isGoal(state):
            actions = []
            # Reconstruct the path by following the parent pointers.
            while state != problem.startingState():
                state, action = path[state]
                actions.append(action)
            # Return the reversed list of actions.
            return actions[::-1]

        # If the state has not been visited, add it to the visited set and expand it.
        if state not in visited:
            visited.add(state)
            # Add the successor states to the data structure.
            for successor in problem.successorStates(state):
                cost = successor[2] + curr_cost
                # Add the state and backwards cost to the data structure.
                if isinstance(dataStructure, PriorityQueue):
                    # If a heuristic is provided (from A*), add it to the cost.
                    dataStructure.push((successor[0], cost),
                        cost + (heuristic(successor[0], problem)) if heuristic else 0)
                else:
                    dataStructure.push((successor[0], cost))
                # Add the parent pointer to the path.
                if successor[0] not in path:
                    path[successor[0]] = (state, successor[1])

    # If the data structure is empty, there is no solution.
    return []

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].
    """

    return genericSearch(problem, Stack())

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    return genericSearch(problem, Queue())

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    return genericSearch(problem, PriorityQueue())

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    return genericSearch(problem, PriorityQueue(), heuristic)
