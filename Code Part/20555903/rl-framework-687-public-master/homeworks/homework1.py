import numpy as np
import random
import matplotlib.pyplot as plt
from rl687.environments.gridworld import Gridworld

np.random.seed(687)

def problemA():
    """
    Have the agent uniformly randomly select actions. Run 10,000 episodes.
    Report the mean, standard deviation, maximum, and minimum of the observed 
    discounted returns.
    """
    returns = np.zeros(10000)
    gridworld = Gridworld()
    for i in range(10000):
        reward_sum = 0
        startState = gridworld.startState
        for t in range(10000):
            action = gridworld.action
            newState = gridworld.step(action)
            reward_sum += pow(gridworld.discount_gamma, t) * gridworld.R(newState)
            if(gridworld.isEnd == True):
                break
        returns[i] = reward_sum
        gridworld.reset()
    
    print('----------------------Printing result for Problem A------------------------------------')
    print('Mean: %f' % np.mean(returns))
    print('Standard Deviation: %f' % np.std(returns))
    print('Maximum: %f' % np.max(returns))
    print('Minimum: %f' % np.min(returns))
    return returns

def problemC():
    """
    Run the optimal policy that you found for 10,000 episodes. Repor the 
    mean, standard deviation, maximum, and minimum of the observed 
    discounted returns
    """
    returns = np.zeros(10000)
    gridworld = Gridworld(problem_number = 2)
    for i in range(10000):
        reward_sum = 0
        startState = gridworld.startState
        for t in range(10000):
            action = gridworld.action
            newState = gridworld.step(action)
            reward_sum += pow(gridworld.discount_gamma, t) * gridworld.R(newState)
            if(gridworld.isEnd == True):
                break
        returns[i] = reward_sum
        gridworld.reset()
    
    
    print('----------------------Printing result for Problem C------------------------------------')
    print('Mean: %f' % np.mean(returns))
    print('Standard Deviation: %f' % np.std(returns))
    print('Maximum: %f' % np.max(returns))
    print('Minimum: %f' % np.min(returns))
    return returns
    
def problemD(returns_random_policy, returns_optimal_policy):
    """
    Plot the distribution of returns for both the random policy and the optimal policy using 10,000 trials each
    """

    # x axis values 
    population = returns_random_policy
    k = 1000
    x = random.sample(list(population), k)
    x.sort()
    # corresponding y axis values 
    y = [yy / k for yy in range(1, k + 1, 1)] 

    # plotting the points  
    plt.plot(y, x) 

    # naming the x axis 
    plt.xlabel('τ') 
    # naming the y axis 
    plt.ylabel('Q(τ)') 

    # giving a title to my graph 
    plt.title('Distribution of returns for random policy (1000 samples) \n and 10,000 trials.') 

    # function to show the plot 
    plt.show()
    
    
    
    # x axis values 
    population = returns_optimal_policy
    k = 1000
    x = random.sample(list(population), k)
    x.sort()
    # corresponding y axis values 
    y = [yy / k for yy in range(1, k + 1, 1)] 

    # plotting the points  
    plt.plot(y, x) 

    # naming the x axis 
    plt.xlabel('τ') 
    # naming the y axis 
    plt.ylabel('Q(τ)') 

    # giving a title to my graph 
    plt.title('Distribution of returns for optimal policy (1000 samples) \n and 10,000 trials.') 

    # function to show the plot 
    plt.show()
    
    
    print('----------------------Printing plot for Problem D------------------------------------')
    print('----------------------Finished Printing all results------------------------------------')
      
        
    
def problemE():
    """
    Simulation to empirically estimate the probability that S19 = 21 given that S8 = 18 for uniform random policy
    """
    result = 0
    num_tests = 10000
    # returns = np.zeros(10000)
    gridworld = Gridworld(startState = 18, problem_number = 1)
    for i in range(num_tests):
        # reward_sum = 0
        startState = gridworld.startState
        for t in range(8, 20):
            if(gridworld.isEnd == True):
                break
            elif(t == 19 and newState == 21):
                result += 1
                break
            action = gridworld.action
            newState = gridworld.step(action)
            # reward_sum += pow(gridworld.discount_gamma, t) * gridworld.R(newState)
        # returns[i] = reward_sum
        gridworld.reset()
        
    probability = result / num_tests
    
    
    print('----------------------Printing result for Problem E------------------------------------')
    print('Probability that S19 = 21 given that S8 = 18 for uniform random policy: %f' % probability)
    
def main():
    print("Hello world")
    returns_random_policy = problemA()
    returns_optimal_policy = problemC()
    problemE()
    problemD(returns_random_policy, returns_optimal_policy)

main()

