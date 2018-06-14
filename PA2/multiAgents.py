# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
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
                some Directions.X for some X in the set {North, South, West, East, Stop}
                """
                # Collect legal moves and successor states
                legalMoves = gameState.getLegalActions()

                # Choose one of the best actions
                scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
                bestScore = max(scores)
                bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
                chosenIndex = random.choice(bestIndices) # Pick randomly among the best

                "Add more of your code here if you want to"

                return legalMoves[chosenIndex]

        def evaluationFunction(self, currentGameState, action):
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
                newPos = successorGameState.getPacmanPosition()
                newFood = successorGameState.getFood()
                newGhostStates = successorGameState.getGhostStates()
                newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

                ManhattanDistance = []

                PacmanPositionOnMaze = list(successorGameState.getPacmanPosition())

                if action == 'Stop':
                    return -float("inf")

                for StateOfGhosts in newGhostStates:

                    #Grab the ghost's position
                    GhostPosition = ghostState.getPosition() 

                    #Grab the timer that determines if the ghost is white (vulnerable)
                    GhostTimer = ghostState.scaredTimer

                    #Is the ghost in Pacman's position and is the ghost white?
                    if(GhostPosition == tuple(PacmanPositionOnMaze) and GhostTimer == 0):
                                return -float("inf")

                #List of Food Pellets
                PelletList = currentGameState.getFood().asList()

                for Pellets in PelletList:
                    ManhattanDistance.append(CalculateManhattanDistance(Pellets[0], PacmanPositionOnMaze[0], Pellets[1], PacmanPositionOnMaze[1]))
                return max(ManhattanDistance)


#Takes an x1, x2, y1, y2.
def CalculateManhattanDistance(PelletePositionX, PacmanPositionX, PelletePositionY, PacmanPositionY):
        X = -1*abs(PelletePositionX - PacmanPositionX)
        Y = -1*abs(PelletePositionY - PacmanPositionY)
        return X,Y


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
            Your minimax agent (question 2)
        """

        def getAction(self, gameState):
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
                """
                Results = self.MinimaxSearch(gameState, 1, 0)
                return Results

        #Minmax search algorithm
        #Based on wikipedia pseudo code: 
        #https://en.wikipedia.org/wiki/Minimax#Minimax_algorithm_with_alternate_moves
        #I am keeping track of index depth as well.
        def MinimaxSearch(self, gameState, Depth, Index):

            if(Depth > self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)

            #Possible legal moves based on index
            PossibleMoves = []
            for Action in gameState.getLegalActions(Index):
                if Action != 'Stop':
                    PossibleMoves.append(Action)

            NumberOfAgents = gameState.getNumAgents()

            if(Index + 1 >= NumberOfAgents):
                Results = [self.MinimaxSearch( gameState.generateSuccessor(Index, Action), Depth+1, 0) for Action in PossibleMoves]
            else:
                Results = [self.MinimaxSearch( gameState.generateSuccessor(Index, Action), Depth, Index + 1) for Action in PossibleMoves]

            #If it is our first move (i.e. depth 1, Index 0)
            if Index == 0 and Depth == 1:

                #Get the best move based on results for MinimaxSearch
                BestMove = max(Results)
                
                #Get the best possible indexes
                BestIndexes = [Index for Index in range(len(Results)) if Results[Index] == BestMove]
                #return the soonest one
                return PossibleMoves[BestIndexes[0]]
            
            if(Index != 0):
                #If the index is NOT 0, return the minimum of the Results
                #For the minimizer
                BestMove = min(Results) 
            else:
                #If the index IS 0, return the maximum of Results
                #For the maximizer
                BestMove = max(Results)
            return BestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        Results = self.AlphaBetaAlgorithm(gameState, 0, self.depth-1, float("-inf"), float("inf"))
        return Results[1]

    def AlphaBetaAlgorithm(self, gameState, Index, Depth, Alpha, Beta):
        #Determine number of agents
        NumberOfAgents = gameState.getNumAgents()

        #We won the game, terminate 
        if gameState.isWin():
            #print("We won!")
            return (self.evaluationFunction(gameState), 'Stop')

        #We lost the game, terminate
        if(gameState.isLose()):
            #print("We lost!")
            return (self.evaluationFunction(gameState), 'Stop')

        #Pacman's turn
        if(Index == 0):
            return self.Maximizer(gameState, Index, Depth, Alpha, Beta)
        
        #Last Ghost
        if Index == NumberOfAgents:
            return self.AlphaBetaAlgorithm(gameState, 0, Depth-1, Alpha, Beta)

        #Opponent's turn
        if Index > 0:
            return self.Minimizer(gameState, Index,Depth, Alpha, Beta)

    def Minimizer(self, gameState, Index, Depth, Alpha, Beta,):
        #Setting the initial best move to stop
        #If it is not updated to something else, it knows to stop.
        BestMove = 'Stop'

        #Setting the initial value
        InitialValue = float("inf")

        #Number of agents
        NumberOfAgents = gameState.getNumAgents()

        #Possible legal moves based on index
        PossibleMoves = []
        for Action in gameState.getLegalActions(Index):
            if Action != 'Stop':
                PossibleMoves.append(Action)

        #Are we in a state where we are onto the last ghost with a depth of 0
        if(Index + 1 == NumberOfAgents and Depth == 0):
            for Action in PossibleMoves:
              Successors = (self.evaluationFunction(gameState.generateSuccessor(Index, Action)), Action)
              #Is the InitialValue greater than the first value of successors
              #On first state, it should be
              if(InitialValue > Successors[0]):

                #Update InitialValue for next test.
                InitialValue = Successors[0]

                #Set the BestMove to the corresponding value
                BestMove = Action

                #Is the Alpha value passed greater than the InitialValue
                #In first run, it should be.
                if(Alpha > InitialValue):
                  return (InitialValue, BestMove)

                #Since we are in Minimizer, set the Beta equal to the lowest between the Beta value, and the Selected Value (InitialValue)
                Beta = min(InitialValue, Beta)
            #Return the updated InitialValue and the BestMove
            return (InitialValue, BestMove)
        #We are NOT on the lastr ghost, so lets continue
        else:
            for Action in PossibleMoves:
                #Pass it to the main function
                #However, increase index by 1 so we know it is a different agent
                Successors = self.AlphaBetaAlgorithm(gameState.generateSuccessor(Index, Action), Index+1, Depth, Alpha, Beta)
                #Is the InitialValue greater than the first of Successors?
                #On first run, it should be.
                if( InitialValue > Successors[0]):

                    #Update the Best Move since it is greater
                    BestMove = Action

                    #Update the InitialValue since it is greater for next test.
                    InitialValue = Successors[0]

                    #Is this updated InitialValue greater than the Alpha value passed?
                    #On first run, it should be.
                    if InitialValue < Alpha:
                        return (InitialValue, BestMove)

                    #Since we are in the minimizer, update the beta value to whatever is lower between the Initialvalue and Beta
                    Beta = min(InitialValue, Beta)
            #Return the InitialValue and the BestMove
            return (InitialValue, BestMove)

    def Maximizer(self, gameState, Index, Depth, Alpha, Beta):
      #Setting the initial best move to stop
        #If it is not updated to something else, it knows to stop.
        BestMove = 'Stop'

        #Setting the initial value
        InitialValue = float("-inf")

        #Number of agents
        NumberOfAgents = gameState.getNumAgents()

        #Possible legal moves based on index
        PossibleMoves = []
        for Action in gameState.getLegalActions(Index):
            if Action != 'Stop':
                PossibleMoves.append(Action)

        for Action in PossibleMoves:
            #Pass it to the main function
            #However, increase index by 1 so we know it is a different agent
            Successors = self.AlphaBetaAlgorithm(gameState.generateSuccessor(Index, Action), Index+1, Depth, Alpha, Beta)

            #Is the first value of the Successors less than that of the Initial Value?
            if(InitialValue < Successors[0]):

                #Update the BestMove since this is a move that maximizes utility
                BestMove = Action

                #Update the InitialValue for next test
                InitialValue = Successors[0]

                #Is the Beta value less than that of the InitialMove?
                #If so, return the InitialValue and the BestMove
                if(Beta < InitialValue):
                    return(InitialValue, BestMove)

                #Since we have concluded that the InitialValue is less than that of the first Successor, we should also update Alpha since we are a maximizer
                #We set it to the greater between the InitialValue and Alpha
                Alpha = max(Alpha, InitialValue)
        return(InitialValue, BestMove)
			
class ExpectimaxAgent(MultiAgentSearchAgent):
        """
            Your expectimax agent (question 4)
        """

        def getAction(self, gameState):
                """
                    Returns the expectimax action using self.depth and self.evaluationFunction

                    All ghosts should be modeled as choosing uniformly at random from their
                    legal moves.
                """

                Results = self.ExpectiMaxSearch(gameState, 1, 0)
                return Results

        #ExpectiMax search algorithm
        #Based on wikipedia pseudo code: 
        #https://en.wikipedia.org/wiki/Expectiminimax_tree
        #I am keeping track of index depth as well.
        def ExpectiMaxSearch(self, gameState, Depth, Index):

            if(Depth > self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)

            #Possible legal moves based on index
            PossibleMoves = []
            for Action in gameState.getLegalActions(Index):
                if Action != 'Stop':
                    PossibleMoves.append(Action)

            NumberOfAgents = gameState.getNumAgents()

            if(Index + 1 >= NumberOfAgents):
                Results = [self.ExpectiMaxSearch( gameState.generateSuccessor(Index, Action), Depth+1, 0) for Action in PossibleMoves]
            else:
                Results = [self.ExpectiMaxSearch( gameState.generateSuccessor(Index, Action), Depth, Index + 1) for Action in PossibleMoves]

            #If it is our first move (i.e. depth 1, Index 0)
            if Index == 0 and Depth == 1:

                #Get the best move based on results for ExpectiMaxSearch
                BestMove = max(Results)
                
                #Get the best possible indexes
                BestIndexes = [Index for Index in range(len(Results)) if Results[Index] == BestMove]
                #return the soonest one
                return PossibleMoves[BestIndexes[0]]
            
            if(Index != 0):
                #If the index is NOT 0, return the average of the Results
                #For the "other" party
                BestMove = sum(Results)/len(Results)
            else:
                #Otherwise, return the max
                BestMove = max(Results)
            return BestMove

def betterEvaluationFunction(currentGameState):
        """
            Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
            evaluation function (question 5).

            DESCRIPTION: <write something here so we know what you did>
        """
        #Get where the pellets are, in a list format
        Pelletslist = currentGameState.getFood().asList();

        #Get current position of Pacman
        PacmanPosition = currentGameState.getPacmanPosition();
        
        #Set Initial Food Distance to 0
        FoodDistance = 0

        #Initial Pacman Score set to 0
        PacmanScore = 0

        #Initial Ghost Score set to 0
        GhostScore = 0

        #Set the maximum score for a game
        MaxScore = 1000

        #Ghost States
        GhostStates = currentGameState.getGhostStates()

        #Initial Ghost White Timer is empty
        #Keeps the timers for the ghosts of they are vulnerable
        GhostWhiteTimer = []

        #Current Game Score
        CurrentGameScore = currentGameState.getScore()

        #Ghost's Punishable Distance from Pacman
        #If a ghost is this distance from pacman, he/she is punished
        GhostPunishedDistance = 3

        #We ate all of the Pellets
        #So we get the max score
        if(len(Pelletslist) == 0):
            PacmanScore = MaxScore

        for Pellets in Pelletslist:
            #Using built in manhattanDistance instead of my own 
            #Because the built in one calculates position on its own
            #I do not need to explicitly state it
            FoodDistance += util.manhattanDistance(PacmanPosition, Pellets)

        #Populate GhostWhiteTimer with the current Scared timers for all ghosts
        for GhostState in GhostStates:
            GhostWhiteTimer.append(GhostState.scaredTimer)

        #If the timer for the first element is greater than 0, give the ghosts 10 points.
        if(GhostWhiteTimer[0] > 0):
            GhostScore += 100

        for GhostState in GhostStates:
            GhostPosition = GhostState.getPosition()
            GhostDistance = util.manhattanDistance(PacmanPosition, GhostPosition)
            GhostScaredTimer = GhostState.scaredTimer

            #Is the Ghost's Distance from Pacman  is greater than the Ghost's Scared Timer, increase score 
            if( GhostDistance > GhostScaredTimer):
                GhostScore += 1/GhostDistance
            else:
                #Kept here to stop issue where GhostScore was not updated
                GhostScore = GhostScore

            #If the Ghost's Diostance from Pacman is greater than GhostPunishedDistance and the GhostScaredTimer is equal to 0, decrease score
            if( GhostDistance < GhostPunishedDistance and GhostScaredTimer == 0):
                GhostScore -= 1/(GhostPunishedDistance - GhostDistance)
            else: 
                #Kept here to stop issue where GhostScore was not updated
                GhostScore = GhostScore

        #Pacman's Score in relation to himself
        #Pacman's Score in relation to the Ghost's performance
        PacmanScore += (1 / (2 + len(Pelletslist)) + 1) / (2 + FoodDistance) + CurrentGameScore + GhostScore

        return PacmanScore

# Abbreviation
better = betterEvaluationFunction

