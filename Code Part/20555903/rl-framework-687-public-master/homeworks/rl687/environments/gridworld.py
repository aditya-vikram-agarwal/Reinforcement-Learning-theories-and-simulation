import numpy as np
from .skeleton import Environment

class Gridworld(Environment):
    """
    The Gridworld as described in the lecture notes of the 687 course material. 
    
    Actions: up (0), down (1), left (2), right (3)
    
    Environment Dynamics: With probability 0.8 the robot moves in the specified
        direction. With probability 0.05 it gets confused and veers to the
        right -- it moves +90 degrees from where it attempted to move, e.g., 
        with probability 0.05, moving up will result in the robot moving right.
        With probability 0.05 it gets confused and veers to the left -- moves
        -90 degrees from where it attempted to move, e.g., with probability 
        0.05, moving right will result in the robot moving down. With 
        probability 0.1 the robot temporarily breaks and does not move at all. 
        If the movement defined by these dynamics would cause the agent to 
        exit the grid (e.g., move up when next to the top wall), then the
        agent does not move. The robot starts in the top left corner, and the 
        process ends in the bottom right corner.
        
    Rewards: -10 for entering the state with water
            +10 for entering the goal state
            0 everywhere else
        
    
    
    """

    def __init__(self, startState=1, endState=23, shape=(5,5), obstacles=[12, 17], waterStates=[7, 17, 21], problem_number = 1):
        self.startState = startState
        self.endState = endState
        self.shape = shape
        self.obstacles = obstacles
        self.waterStates = waterStates
        self.discount_gamma = 0.9
        self.currentState = startState
        self.current_action = 0
        self.problem_number = problem_number
        
    @property
    def name(self):
        pass
        
    @property
    def reward(self):
        pass

    @property
    def action(self):
        if self.problem_number == 1:
            self.current_action = np.random.randint(4)
            return self.current_action
        elif self.problem_number == 2:
            if self.currentState <= 4 or self.currentState >= 7 and self.currentState <= 9 or self.currentState == 13 or self.currentState == 17 or self.currentState >= 21:
                self.current_action = 3
            elif self.currentState == 6 or self.currentState == 11 or self.currentState == 15 or self.currentState == 19:
                self.current_action = 0
            elif self.currentState == 16 or self.currentState == 20:
                self.current_action = 2
            elif self.currentState == 12 or self.currentState == 5 or self.currentState == 10 or self.currentState == 14 or self.currentState == 18:
                self.current_action = 1
            return self.current_action

    @property
    def isEnd(self):
        if(self.currentState == self.endState):
            return True
        else:
            return False

    @property
    def state(self):
        return self.currentState

    @property
    def gamma(self):
        pass

    def step(self, action):
        
        probability = np.random.randint(100)
        if(probability >= 95):
            # veer left
            if(action == 0):
                newaction = 2
            elif(action == 2):
                newaction = 1
            elif(action == 1):
                newaction = 3
            elif(action == 3):
                newaction = 0
        elif(probability >= 90):
            # veer right
            if(action == 0):
                newaction = 3
            elif(action == 2):
                newaction = 0
            elif(action == 1):
                newaction = 2
            elif(action == 3):
                newaction = 1
        elif(probability >= 80):
            # go nowhere
            return self.currentState
        elif(probability < 80):
            if(action == 0):
                newaction = 0
            elif(action == 2):
                newaction = 2
            elif(action == 1):
                newaction = 1
            elif(action == 3):
                newaction = 3

        if(newaction == 0):
            if(self.currentState <= 5 or self.currentState == 21):
                return self.currentState
            elif(self.currentState >= 6 and self.currentState <= 12 or self.currentState >= 22 and self.currentState <= 23):
                self.currentState = self.currentState - 5
            elif(self.currentState >= 13 and self.currentState <= 20):
                self.currentState = self.currentState - 4
        elif(newaction == 1):
            if(self.currentState >= 19 or self.currentState == 8):
                return self.currentState
            elif(self.currentState >= 1 and self.currentState <= 7 or self.currentState >= 17 and self.currentState <= 18):
                self.currentState = self.currentState + 5
            elif(self.currentState >= 9 and self.currentState <= 16):
                self.currentState = self.currentState + 4
        elif(newaction == 2):
            if(self.currentState == 1 or self.currentState == 6 or self.currentState == 11 or self.currentState == 15 or self.currentState == 19 or self.currentState == 13 or self.currentState == 17):
                return self.currentState
            else:
                self.currentState = self.currentState - 1
        elif(newaction == 3):
            if(self.currentState == 5 or self.currentState == 10 or self.currentState == 14 or self.currentState == 18 or self.currentState == 23 or self.currentState == 12 or self.currentState == 16):
                return self.currentState
            else:
                self.currentState = self.currentState + 1

        return self.currentState

    def reset(self):
        self.current_action = 0
        self.currentState = self.startState
        
    def R(self, _state):
        """
        reward function
        
        output:
            reward -- the reward resulting in the agent being in a particular state
        """
        
        if(_state in self.waterStates):
            return -10
        elif(_state == self.endState):
            return +10
        else:
            return 0
        
