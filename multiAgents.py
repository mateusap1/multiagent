# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent, AgentState, Grid
from pacman import GameState

from typing import Tuple, List


def ray_distance(
    grid: List[List[bool]],
    grid_width: int,
    grid_height: int,
    player_pos: Tuple[int, int],
    max_breadth: int,
) -> int:
    for cur_breadth in range(max_breadth + 1):
        if ray_trigger(grid, grid_width, grid_height, player_pos, cur_breadth):
            return cur_breadth

    return None


def ray_trigger(
    grid: List[List[bool]],
    grid_width: int,
    grid_height: int,
    player_pos: Tuple[int, int],
    breadth: int,
) -> bool:
    player_pos_x, player_pox_y = player_pos

    if breadth == 0:
        # print("Yuhai!")
        # print("grid", grid)
        # print("postions", player_pos_x, player_pox_y, grid[player_pos_x][player_pox_y])
        # print("Yuhai!")
        return grid[player_pos_x][player_pox_y]

    initial_pos_x_start = max(player_pos[0] - breadth, 0)
    initial_pos_x_end = min(player_pos[0] + breadth, grid_width - 1)

    initial_pos_y_start = max(player_pos[1] - breadth, 0)
    initial_pos_y_end = min(player_pos[1] + breadth, grid_height - 1)

    # First let's check the edges

    if grid[initial_pos_x_start][initial_pos_y_start]:
        return True

    if grid[initial_pos_x_start][initial_pos_y_end]:
        return True

    if grid[initial_pos_x_end][initial_pos_y_start]:
        return True

    if grid[initial_pos_x_end][initial_pos_y_end]:
        return True

    # Now the rest

    # UP
    for x in range(initial_pos_x_start + 1, initial_pos_x_end):
        if grid[x][initial_pos_y_start]:
            return True

    # DOWN
    for x in range(initial_pos_x_start + 1, initial_pos_x_end):
        if grid[x][initial_pos_y_end]:
            return True

    # LEFT
    for y in range(initial_pos_y_start + 1, initial_pos_y_end):
        if grid[initial_pos_x_start][y]:
            return True

    # RIGHT
    for y in range(initial_pos_y_start + 1, initial_pos_y_end):
        if grid[initial_pos_x_end][y]:
            return True

    return False


def normalize(value: int, max_value: int) -> float:
    return value / max_value


def normalize_opposite(value: int, max_value: int) -> float:
    return normalize(max_value - value, max_value)


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        playerPos: Tuple[int, int] = successorGameState.getPacmanPosition()
        playerPosX, playerPosY = int(playerPos[0]), int(playerPos[1])

        foodGrid: Grid = successorGameState.getFood()

        ghostStates: List["AgentState"] = successorGameState.getGhostStates()
        ghostPositions: List[Tuple[int, int]] = [
            ghostState.getPosition() for ghostState in ghostStates
        ]

        minEnemyDistance: float = max(foodGrid.width, foodGrid.height)
        for ghostPosX, ghostPosY in ghostPositions:
            enemyDistance = (playerPosX - ghostPosX) ** 2 + (
                playerPosY - ghostPosY
            ) ** 2

            if enemyDistance < minEnemyDistance:
                minEnemyDistance = enemyDistance

        minEnemyDistanceNormalized = normalize(
            minEnemyDistance, max(foodGrid.width, foodGrid.height)
        )

        ateFood = (currentGameState.getNumFood() - successorGameState.getNumFood()) == 1
        distanceToFood = 0
        if not ateFood:
            distanceToFood = ray_distance(
                foodGrid, foodGrid.width, foodGrid.height, playerPos, 5
            )
        print("absolute food distance", distanceToFood)
        distanceToFoodNormalized = normalize_opposite(
            distanceToFood, 5
        ) if distanceToFood is not None else 0

        print("food distance", distanceToFoodNormalized)

        return minEnemyDistanceNormalized + 5 * distanceToFoodNormalized


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
