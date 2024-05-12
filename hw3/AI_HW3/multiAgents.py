from util import manhattanDistance, Queue
from game import Directions
import random, util
from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        """
        In the recursion of minimax. 
        When it's pacman turn, we choose the max value. 
        When it's ghosts turn, we choose the min value. 
        When it's depth = 0 or win or Lose, we evaluate the state value
        Then, we choose the action with biggest value among all of the returns
        """
        legalActions = getNonStopActions(gameState, 0)
        values = [self.minimax(gameState.getNextState(0, action), self.depth, 1) for action in legalActions]
        maxi = max(values)
        max_indexs = [i for i in range(len(legalActions)) if (values[i] == maxi)]

        choose = random.choice(max_indexs)
        return legalActions[max_indexs[0]]
        # End your code (Part 1)

    def minimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # Pacman's turn (maximizing player)
        if agentIndex == 0:
            return max(self.minimax(state.getNextState(agentIndex, action), depth, 1) for action in
                       getNonStopActions(state, agentIndex))
        # Ghosts' turn (minimizing players)
        else:
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                nextAgent = 0  # Reset to Pacman's turn
            if nextAgent == 0:
                depth -= 1  # Reduce depth when it's Pacman's turn again
            return min(self.minimax(state.getNextState(agentIndex, action), depth, nextAgent) for action in
                       getNonStopActions(state, agentIndex))


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        """
        The recursion of alpha_beta_pruning is like minimax function.
        However, we prune the possible outcomes to speed up the recursion.
        When we find a max value in finding max turn, 
            we set it to be a lower bound (alpha) of recursion. 
            That is, if we have a value which definitely lower than the lower bound (beta < alpha) 
            the outcome will not be our choose, so we break the recursion (pruing)
        Similarly, we set upper bound (beta) in finding min turn
        Finally, we choose the action with biggest value among all of the returns
        """
        legalActions = getNonStopActions(gameState, 0)
        values = []
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in legalActions:
            value = self.alpha_beta_pruning(gameState.getNextState(0, action), self.depth, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
            values.append(value)

        maxi = max(values)
        max_indexs = [i for i in range(len(legalActions)) if (values[i] == maxi)]
        choose = random.choice(max_indexs)
        return legalActions[max_indexs[0]]
        # End your code (Part 2)

    def alpha_beta_pruning(self, state, depth, agentIndex, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # Pacman's turn (maximizing player)
        if agentIndex == 0:
            v = float('-inf')
            for action in getNonStopActions(state, agentIndex):
                v = max(v, self.alpha_beta_pruning(state.getNextState(agentIndex, action), depth, 1, alpha, beta))
                alpha = max(alpha, v)
                if beta < alpha:
                    break
            return v

        # Ghosts' turn (minimizing players)
        else:
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                nextAgent = 0  # Reset to Pacman's turn
            if nextAgent == 0:
                depth -= 1     # Reduce depth when it's Pacman's turn again

            v = float('inf')
            for action in getNonStopActions(state, agentIndex):
                v = min(v, self.alpha_beta_pruning(state.getNextState(agentIndex, action), depth, nextAgent, alpha, beta))
                beta = min(v, beta)
                if beta < alpha:
                    break
            return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        """
        In the recursion of minimax. 
        When it's pacman turn, we choose the max value. 
        When it's ghosts turn, we take weighted sum of values.
        ( In this case, we take uniform of all values ) 
        When it's depth = 0 or win or Lose, we evaluate the state value
        Then, we choose the action with biggest value among all of the returns
        """
        legalActions = getNonStopActions(gameState, 0)
        bestAction = None
        bestValue = float('-inf')
        values = []

        for action in legalActions:
            v = self.expectimax(gameState.getNextState(0, action), self.depth, 1)
            if v > bestValue:
                bestAction = action
                bestValue = v
            values.append(v)

        maxi = max(values)
        max_indexs = [i for i in range(len(legalActions)) if (values[i] == maxi)]
        choose = random.choice(max_indexs)
        return legalActions[max_indexs[0]]
        # End your code (Part 3)

    def expectimax(self, state, depth, agentIndex):

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agentIndex == 0:  # Pacman's turn (maximizing player)
            return max(self.expectimax(state.getNextState(agentIndex, action), depth, 1) for action in
                       getNonStopActions(state, agentIndex))

        else:  # Ghosts' turn (chance node)
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                nextAgent = 0  # Reset to Pacman's turn
            if nextAgent == 0:
                depth -= 1     # Reduce depth when it's Pacman's turn again

            legalActions = getNonStopActions(state, agentIndex)
            numActions = len(legalActions)
            return sum(self.expectimax(state.getNextState(agentIndex, action), depth, nextAgent) for action in
                       legalActions) / numActions


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    """
    We evaluate state based on NonScared ghosts, Scared ghost, Food, and Capsule
    First, if lose or almost loss ( minNonScared <= 1)
        we return -300000 or -200000
    Second, if we could eat the Scared Ghost (minScaredTime > minScared)
        evaluation += 150000 * (1 / minScared)
    Third, we encourage pacman to eat capsules,
        evaluatoin += 200 * (1 / nearCapsule) + 10
    Fourth, we encourage pacman to eat foods,
        evaluation += 10 * (1 / nearFood) + 5
    Besides, we use bfs to find the correct distance to object,
        and we initialize evaluation = 2 * score to let it sensitive to the change of score 
    """
    dxs = [1, 0, -1, 0]
    dys = [0, 1, 0, -1]

    def findEnd(start, end):
        q = Queue()
        q.push(start)
        dist = {}
        dist[start] = 0
        while not q.isEmpty():
            xy = q.pop()
            if xy == end:
                return dist[xy]
            for dx, dy in zip(dxs, dys):
                if not currentGameState.hasWall(xy[0] + dx, xy[1] + dy) and (xy[0] + dx, xy[1] + dy) not in dist:
                    q.push((xy[0] + dx, xy[1] + dy))
                    dist[(xy[0] + dx, xy[1] + dy)] = dist[xy] + 1
        return None

    def findFood(start):
        q = Queue()
        q.push(start)
        dist = {}
        dist[start] = 0
        while not q.isEmpty():
            xy = q.pop()
            if currentGameState.hasFood(xy[0], xy[1]):
                return dist[xy]
            for dx, dy in zip(dxs, dys):
                if not currentGameState.hasWall(xy[0] + dx, xy[1] + dy) and (xy[0] + dx, xy[1] + dy) not in dist:
                    q.push((xy[0] + dx, xy[1] + dy))
                    dist[(xy[0] + dx, xy[1] + dy)] = dist[xy] + 1
        return None

    # score, pos
    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    # if currentGameState.isLose():
    #     return -10000

    # food
    foods = currentGameState.getFood().asList()
    nearFood = findFood(pos)

    # capsule
    capsule = [findEnd(pos, capsule_pos) for capsule_pos in currentGameState.getCapsules()
               if findEnd(pos, capsule_pos) is not None]
    nearCapsule = min(capsule) if len(capsule) > 0 else 9999

    # ghosts
    ghosts = currentGameState.getGhostStates()
    minScared = 9999
    minNonScared = 9999
    minScaredTime = 9999
    for ghost in ghosts:
        dist = findEnd(pos, ghost.getPosition())
        if ghost.scaredTimer > 0 and dist is not None:
            minScared = min(minScared, dist)
            minScaredTime = ghost.scaredTimer
        elif ghost.scaredTimer == 0 and dist is not None:
            minNonScared = min(minNonScared, dist)

    # main evaluation
    evaluation = 2 * score
    if currentGameState.isLose():
        return -300000
    if minNonScared <= 1:
        return -200000
    if minScaredTime > minScared:
        evaluation += 150000 * (1 / minScared)

    if len(capsule) > 0:
        evaluation += 200 * (1 / nearCapsule) + 10

    if nearFood is not None:
        evaluation += 10 * (1 / nearFood) + 5

    # print(f"near food = {nearFood}, evaluation = {evaluation}")
    return evaluation


def getNonStopActions(state, agentIndex):
    actions = state.getLegalActions(agentIndex)
    # if Directions.STOP in actions:
    #     actions.remove(Directions.STOP)
    return actions


# Abbreviation
better = betterEvaluationFunction
