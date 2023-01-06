#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import numpy as np

from environment import MountainCar, GridWorld

"""
Please read: THE ENVIRONMENT INTERFACE

In this homework, we provide the environment (either MountainCar or GridWorld) 
to you. The environment returns states, represented as 1D numpy arrays, rewards, 
and a Boolean flag indicating whether the episode has terminated. The environment 
accepts actions, represented as integers.

The only file you need to modify/read is this one. We describe the environment 
interface below.

class Environment: # either MountainCar or GridWorld

    def __init__(self, mode, debug=False):
        Initialize the environment with the mode, which can be either "raw" 
        (for the raw state representation) or "tile" (for the tiled state 
        representation). The raw state representation contains the position and 
        velocity; the tile representation contains zeroes for the non-active 
        tile indices and ones for the active indices. GridWorld must be used in 
        tile mode. The debug flag will log additional information for you; 
        make sure that this is turned off when you submit to the autograder.

        self.state_space = an integer representing the size of the state vector
        self.action_space = an integer representing the range for the valid actions

        You should make use of env.state_space and env.action_space when creating 
        your weight matrix.

    def reset(self):
        Resets the environment to initial conditions. Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the initial state.
    
    def step(self, action):
        Updates itself based on the action taken. The action parameter is an 
        integer in the range [0, 1, ..., self.action_space). Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the new state that the agent is in after taking its 
                        specified action.
            
            (2) reward : A float indicating the reward received at this step.

            (3) done : A Boolean flag indicating whether the episode has 
                        terminated; if this is True, you should reset the 
                        environment and move on to the next episode.
    
    def render(self, mode="human"):
        Renders the environment at the current step. Only supported for MountainCar.


For example, for the GridWorld environment, you could do:

    env = GridWorld(mode="tile")

Then, you can initialize your weight matrix to all zeroes with shape 
(env.action_space, env.state_space+1) (if you choose to fold the bias term in). 
Note that the states returned by the environment do *not* have the bias term 
folded in.
"""

def parse_args() -> tuple:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate
    
    Usage:
        env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)

    args = parser.parse_args()

    return args.env, args.mode, args.weight_out, args.returns_out, args.episodes, args.max_iterations, args.epsilon, args.gamma, args.learning_rate


if __name__ == "__main__":

    env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()

    if env_type == "mc":
        #env = None # Replace me!
        env = MountainCar(mode = mode)
    elif env_type == "gw":
        #env = None # Replace me!
        env = GridWorld(mode = mode)
    else: raise Exception(f"Invalid environment type {env_type}")


    weight = np.zeros((env.action_space, env.state_space + 1))  # if choose to fold the bias term in
    returns = []
    for episode in range(episodes):

        # Get the initial state by calling env.reset()
        s = env.reset() # initial state
        r_ls = []

        for iteration in range(max_iterations): # #of states per
            if iteration == 0:
                s_i = np.concatenate(([1],s),axis = 0) #np.append([1], s, axis=0)
            #s_i = np.concatenate(([1],s),axis = 0)


            # Select an action based on the state via the epsilon-greedy strategy
            a_star = np.argmax(weight @ s_i) #with epsilon
            #a_star = [i for i in range(env.action_space) if (weight @ s_i)[i] == np.max(weight @ s_i)][0]
            #a = np.random.randint(0,env.action_space,size = 1)[0] #with (1-epsilon)
            #np.random.seed(iteration)
            #p = []
            #for num in range(env.action_space):
            #    rand_p = epsilon/env.action_space
            #    p.append(rand_p)
            #p.append(1-epsilon)
            #actions = [i for i in range(env.action_space)]
            #actions.append(a_star)
            #a = np.random.choice(actions,p = p)
            if np.random.rand() > epsilon:
                a = a_star
            else:
                a = np.random.int(env.action_space)


            # Take a step in the environment with this action, and get the 
            # returned next state, reward, and done flag
            s_i_next,r_i,d_flag = env.step(a) # regarding s_i
            #s_i_next = np.append([1], s_i_next, axis=0)
            s_i_next = np.concatenate(([1],s_i_next), axis=0)
            r_ls.append(r_i)

            # Using the original state, the action, the next state, and 
            # the reward, update the parameters. Don't forget to update the 
            # bias term!
            q_value = weight @ s_i
            q_appr = q_value[a]
            q_true = r_i + gamma * max(weight @ s_i_next)
            #dq = np.zeros((env.action_space, env.state_space + 1))
            #dq[a] = s_i
            #grad_i = (q_appr - q_true) * dq
            #weight = weight - lr * grad_i
            grad_i = (q_appr - q_true) * s_i
            weight[a] = weight[a] - lr * grad_i

            # Remember to break out of this inner loop if the environment signals done!
            if d_flag == True:
                break
            else:
                s_i = s_i_next
                #s = s_i_next[1:]
            pass

        returns.append(sum(r_ls)) #if total rewards (the returns) refer to sum rewards

    #print(weight)
    #print(returns)

    # Save your weights and returns. The reference solution uses
    #weight_out =
    np.savetxt(weight_out,weight, fmt="%.18e", delimiter=" ")
    #returns_out =
    np.savetxt(returns_out,returns, fmt="%.18e", delimiter=" ")

