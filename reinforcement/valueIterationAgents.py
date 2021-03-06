# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here

        #Grab State from MDP
        State = self.mdp.getStates()[2]

        #Grab the NextState from TransitionStates based on the current state
        NextState = mdp.getTransitionStatesAndProbs(State, mdp.getPossibleActions(State)[0])

        #List of all States
        States = self.mdp.getStates()


        #MDP States
        MDPStates = self.mdp.getStates()

        for i in range(0, self.iterations):

          TemporaryValue = util.Counter()

          for State in MDPStates:

          	#Is the state a terminal state?
            if self.mdp.isTerminal(State):
            	TemporaryValue[State] = 0

            else:
            	MaxPossibleValue = float("-inf")

            	for Action in self.mdp.getPossibleActions(State):
            		#Score is initially 0
            		Score = 0

            		for NextState, Probability in self.mdp.getTransitionStatesAndProbs(State, Action):
            			#Bellman equation
            			Score += Probability * (self.mdp.getReward(State, Action, NextState) + (self.discount*self.values[NextState]))

            		#Set max possible value to whichever is greater, old value or current score.
            		MaxPossibleValue = max(Score, MaxPossibleValue)

            		#Update the temporary value of the state (for next iteration)
            		TemporaryValue[State] = MaxPossibleValue
          self.values = TemporaryValue


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        #InitialScore is 0
        TotalScore = 0

        for NextState, Probability in self.mdp.getTransitionStatesAndProbs(state,action):
        	#Bellman Equation
          TotalScore += Probability * (self.mdp.getReward(state, action, NextState) + (self.discount*self.values[NextState]))
        return TotalScore

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        #Is the current state the termination state? 
        #If so, done.
        if self.mdp.isTerminal(state):
          return None

        #Initial Value is -infinity
        Value = float("-inf")

        #No policy intially
        Policy = None

        #Iterate through the possible actions for the given state
        for Action in self.mdp.getPossibleActions(state):

        	#Calculate temporar q value for state
        	TemporaryValue = self.computeQValueFromValues(state, Action)

        	#If the temporaryvalue is greater or equal to the Value, then update it.
        	#AND update the action as the policy
        	if TemporaryValue >= Value:

        		#Updating Policy
        		Policy = Action

        		#Updating Value for Q-value
        		Value = TemporaryValue
        	else:
        		#Otherwise, continue.
        		continue
        return Policy

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
