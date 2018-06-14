# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    #print("Starting Depth First Search (DFS).")

    #Defining Start State
    Start_State = problem.getStartState()

    #Defining empty Visited_Nodes Array
    Visited_Nodes = []

    #Defining Stack, as Fringe
    #Stack uses LIFO so it is ideal for DFS
    Fringe = util.Stack()

    #Adding Start_State to Queue as instructed in slides
    #I am pushing the Start_State, Action_Array and 0 for cost, DFS doesn't consider cost so we are not concerned with it
    Fringe.push((Start_State, [], 0))

    while not Fringe.isEmpty(): #Nodes still in queue
    	#Setting Current_Position, Action_Array and Total_Cost equal to last position in Fringe using pop()
        Current_Position, Action_Array, Total_Cost = Fringe.pop()

        #Is the current position the goal?
        #I had to add this to pass the autograder
        #Originally, it was in the for loop, but as instructed in Piazza, I moved it here.
        if problem.isGoalState(Current_Position):
        	return Action_Array

       	#Is the current position in the visited nodes?
       	#We do not want to visit places we have already visited.
        if Current_Position not in Visited_Nodes:
        	Visited_Nodes.append(Current_Position)

        	#Generating a list of successors based on current position
        	Successors = problem.getSuccessors(Current_Position)

        	for next_state, Action, Cost in Successors:
        		if next_state not in Visited_Nodes:
        			#next_state is not in Visited Nodes, therefore it is an unexplored node.
        			#Adding it to the fringe so we can move in that direction in next iteration
        			Fringe.push((next_state, Action_Array + [Action], Cost))
        		else:
        			continue
    [] #return empty aray if no solution was found

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    #print("Starting Breath First Search (BFS).")

    #Defining Start State
    Start_State = problem.getStartState()

    #Defining empty Visited_Nodes Array
    Visited_Nodes = []

    #Defining Queue, as Fringe
    #Stack uses FIFO so it is ideal for DFS
    Fringe = util.Queue()

    #Adding Start_State to Queue as instructed in slides
    #I am pushing the Start_State, Action_Array and 0 for cost, DFS doesn't consider cost so we are not concerned with it
    Fringe.push((Start_State, [], 0))

    while not Fringe.isEmpty(): #Nodes still in queue
    	#Setting Current_Position, Action_Array and Total_Cost equal to last position in Fringe using pop()
        Current_Position, Action_Array, Total_Cost = Fringe.pop()

        #Is the current position the goal?
        #I had to add this to pass the autograder
        #Originally, it was in the for loop, but as instructed in Piazza, I moved it here.
        if problem.isGoalState(Current_Position):
        	return Action_Array

       	#Is the current position in the visited nodes?
       	#We do not want to visit places we have already visited.
        if Current_Position not in Visited_Nodes:
        	Visited_Nodes.append(Current_Position)

        	#Generating a list of successors based on current position
        	Successors = problem.getSuccessors(Current_Position)

        	for next_state, Action, Cost in Successors:
        		if next_state not in Visited_Nodes:
        			#next_state is not in Visited Nodes, therefore it is an unexplored node.
        			#Adding it to the fringe so we can move in that direction in next iteration
        			Fringe.push((next_state, Action_Array + [Action], Cost))
        		else:
        			continue
    [] #return empty aray if no solution was found


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    #print("Starting Uniform Cost Search (UCS).")

    #Defining Start State
    Start_State = problem.getStartState()

    #Defining empty Visited_Nodes Array
    Visited_Nodes = []

    #Defining Priority, as Fringe
    Fringe = util.PriorityQueue()

    #Adding Start_State to Queue as instructed in slides
    #I am pushing the starting state, Actions, and cost for initial position.
    #Fringe also has a field for Total Cost, setting that to 0 as well.
    Fringe.push((Start_State, [], 0), 0)

    while not Fringe.isEmpty(): #Nodes still in queue
    	#Setting Current_Position, Action_Array and Action_Cost equal to last position in Fringe using pop()
        Current_Position, Action_Array, Action_Cost = Fringe.pop()

        #Is the current position the goal?
        #I had to add this to pass the autograder
        #Originally, it was in the for loop, but as instructed in Piazza, I moved it here.
        if problem.isGoalState(Current_Position):
        	return Action_Array

       	#Is the current position in the visited nodes?
       	#We do not want to visit places we have already visited.
        if Current_Position not in Visited_Nodes:
        	Visited_Nodes.append(Current_Position)

        	#Generating a list of successors based on current position
        	Successors = problem.getSuccessors(Current_Position)

        	for next_state, Action, Cost in Successors:
        		if next_state not in Visited_Nodes:
        			#next_state is not in Visited Nodes, therefore it is an unexplored node.
        			#Adding it to the fringe so we can move in that direction in next iteration
        			Temp_Cost = Action_Cost + Cost
        			Fringe.push((next_state, Action_Array + [Action], Temp_Cost), Temp_Cost)
        		else:
        			continue
    [] #return empty aray if no solution was found
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    #print("Starting A* Search.")

    #Defining Start State
    Start_State = problem.getStartState()

    #Defining empty Visited_Nodes Array
    Visited_Nodes = []

    #Defining Priority, as Fringe
    Fringe = util.PriorityQueue()

    #Adding Start_State to Queue as instructed in slides
    #I am pushing the starting state, Actions, and cost for initial position.
    #Fringe also has a field for Total Cost, setting that to 0 as well.
    Fringe.push((Start_State, [], 0), 0)

    while not Fringe.isEmpty(): #Nodes still in queue
    	#Setting Current_Position, Action_Array and Action_Cost equal to last position in Fringe using pop()
        Current_Position, Action_Array, Action_Cost = Fringe.pop()

        #Is the current position the goal?
        #I had to add this to pass the autograder
        #Originally, it was in the for loop, but as instructed in Piazza, I moved it here.
        if problem.isGoalState(Current_Position):
        	return Action_Array

       	#Is the current position in the visited nodes?
       	#We do not want to visit places we have already visited.
        if Current_Position not in Visited_Nodes:
        	Visited_Nodes.append(Current_Position)

        	#Generating a list of successors based on current position
        	Successors = problem.getSuccessors(Current_Position)

        	for next_state, Action, Cost in Successors:
        		if next_state not in Visited_Nodes:
        			#next_state is not in Visited Nodes, therefore it is an unexplored node.
        			#Adding it to the fringe so we can move in that direction in next iteration

        			#Cost_To_Go is the heuristic cost of the "next_state" provided
        			#This is provided by the problem
        			Cost_To_Go = heuristic(next_state, problem)

        			#Incurred_Cost is the cost so far
        			Incurred_Cost = Action_Cost + Cost

        			Fringe.push((next_state, Action_Array + [Action], Incurred_Cost), Incurred_Cost + Cost_To_Go)
        		else:
        			continue
    [] #return empty aray if no solution was found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
