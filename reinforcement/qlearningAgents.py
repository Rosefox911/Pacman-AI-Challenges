# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qvals = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """

        #List of QValues
        QValues = self.qvals

        #CurrentQValue based on state and action
        CurrentQValue = (state, action)

        if CurrentQValue not in QValues:
        	#Initialize them as 0
        	self.qvals[(state, action)] = 0.0

        return self.qvals[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        #Possible, legal, actions
        PossibleActions = self.getLegalActions(state)

        #Returning 0 as specified if no legal actions possible
        if len(PossibleActions) == 0:
        	return 0.0

        #Temporary value initialization
        TemporaryValue = util.Counter()

        for Action in PossibleActions:
        	#Update TemporaryValue recursively for all possible moves
        	#Set it to the q value for the provided action and state
        	TemporaryValue[Action] = self.getQValue(state, Action)

        #Has to be argmax, cannot be normal max.
        return TemporaryValue[TemporaryValue.argMax()] 

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        #Possible, legal, actions
        PossibleActions = self.getLegalActions(state)

        #Returning 0 as specified if no legal actions possible
        if len(PossibleActions) == 0:
        	return 0.0

        #Temporary value initialization
        TemporaryValue = util.Counter()

        for Action in PossibleActions:
        	#Update TemporaryValue recursively for all possible moves
        	#Set it to the q value for the provided action and state
        	TemporaryValue[Action] = self.getQValue(state, Action)

        #Has to be argmax, cannot be normal max.
        return TemporaryValue.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        #Possible, legal, actions
        PossibleActions = self.getLegalActions(state)

        #Returning None as specified if no legal actions possible
        if len(PossibleActions) == 0:
        	return None

        elif(util.flipCoin(self.epsilon)):
        	#Picking Random Action as specified
        	action = random.choice(PossibleActions)
        else:
        	action = self.computeActionFromQValues(state)
        return action


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        #Value for Next State
        QValueForNextState = self.computeValueFromQValues(nextState)

        #Current QValue for current state
        QValueForCurrentState = self.getQValue(state,action)

        #Q-Learning Formula
        #http://docs.goodai.com/brainsimulator/guides/discreteqlearning/Q-update.png
        self.qvals[(state,action)] =  ((1-self.alpha) * QValueForCurrentState) + self.alpha * (reward + self.discount * QValueForNextState)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        PacmanFeatures = self.featExtractor.getFeatures(state, action)
        return PacmanFeatures.__mul__(self.weights)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        #Grab Pacman Features
        PacmanFeatures = self.featExtractor.getFeatures(state, action)

        #Set the value equal to QValue formula
    	Value = (reward + (self.discount * self.getValue(nextState)) - self.getQValue(state, action))
    	#Update weights for each feature
    	for Feature in PacmanFeatures:
        	self.weights[Feature] = self.weights[Feature] + (self.alpha * Value * PacmanFeatures[Feature])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            #print(self.weights)
            pass
