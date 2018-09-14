"""Example experiment for CMPUT 366 Fall 2019

This experiment uses the rl_step() function.

Runs a random agent in a 1D environment. Runs 10 (num_runs) iterations of
100 episodes, and reports the final average reward. Each episode is capped at
100 steps (max_steps).
"""

import numpy as np
from random import randint
from envBase import envBase
from smartAgent import smartAgent
from rl_glue import RLGlue
import matplotlib.pyplot as plt

def experiment1(rlg, num_runs, max_steps, num_arms):
    totalRewardsAtEachStep = np.zeros(max_steps)
    #totalRewardsAtEachStep.zeros(max_step)
    rewards = np.zeros(num_runs)
    ####################################
    epsilon = 0
    initalReward = 5
    stepSize = 0.1
    ####################################
    for run in range(num_runs):
        rewardAtEachStep = np.array([])
        rlg.rl_init()
        bestArm = rlg.rl_env_message("num_arms "+str(num_arms))
        rlg.rl_env_message("max_steps "+str(max_steps))
        rlg.rl_agent_message("num_arms "+str(num_arms))
        rlg.rl_agent_message("epsilon "+str(epsilon))
        rlg.rl_agent_message("initalReward "+str(initalReward))
        rlg.rl_agent_message("stepSize "+str(stepSize))
        rlg.rl_start()
        for x in range(max_steps):
            reward, state, action, is_terminal = rlg.rl_step()
            rewardAtEachStep = np.append(rewardAtEachStep, state[1])
            #totalRewardsAtEachStep[x] += (1/(run+1))*(rewardAtEachStep[x] - totalRewardsAtEachStep[x])
        #print("Adding lists:")
        #print(rewardAtEachStep)
        #print(bestArm)        
        for x in range(0,np.size(rewardAtEachStep)):
            #print("Comparing if "+str(rewardAtEachStep[x])+" == "+str(bestArm))
            if(rewardAtEachStep[x] == bestArm):
                totalRewardsAtEachStep[x] += 1
                
        #print(totalRewardsAtEachStep)
            #totalRewardsAtEachStep[x] = totalRewardsAtEachStep[x]/num_runs
            #totalRewardsAtEachStep[x] += (1/(run+1))*(rewardAtEachStep[x] - totalRewardsAtEachStep[x])
        #print("result")
        #print(state)
        
        #rewards[run] = rlg.total_reward()
        print("Completed run: "+str(run))
    return totalRewardsAtEachStep, rewards.mean()

def experiment2(rlg, num_runs, max_steps, num_arms):
    totalRewardsAtEachStep = np.zeros(max_steps)
    #totalRewardsAtEachStep = np.zeros(max_steps)
    rewards = np.zeros(num_runs)
    ####################################
    epsilon = 0.1
    initalReward = 0
    stepSize = 0.1
    ####################################
    for run in range(num_runs):
        rewardAtEachStep = np.array([])
        rlg.rl_init()
        bestArm = rlg.rl_env_message("num_arms "+str(num_arms))
        rlg.rl_env_message("max_steps "+str(max_steps))
        rlg.rl_agent_message("num_arms "+str(num_arms))
        rlg.rl_agent_message("epsilon "+str(epsilon))
        rlg.rl_agent_message("initalReward "+str(initalReward))
        rlg.rl_agent_message("stepSize "+str(stepSize))
        rlg.rl_start()
        for x in range(max_steps):
            reward, state, action, is_terminal = rlg.rl_step()
            rewardAtEachStep = np.append(rewardAtEachStep, state[1])
            #totalRewardsAtEachStep[x] += (1/(run+1))*(rewardAtEachStep[x] - totalRewardsAtEachStep[x])
        #print("Adding lists:")
        #print(rewardAtEachStep)
        #print(bestArm)        
        for x in range(0,np.size(rewardAtEachStep)):
            #print("Comparing if "+str(rewardAtEachStep[x])+" == "+str(bestArm))
            if(rewardAtEachStep[x] == bestArm):
                totalRewardsAtEachStep[x] += 1
                
        #print(totalRewardsAtEachStep)
            #totalRewardsAtEachStep[x] = totalRewardsAtEachStep[x]/num_runs
            #totalRewardsAtEachStep[x] += (1/(run+1))*(rewardAtEachStep[x] - totalRewardsAtEachStep[x])
        #print("result")
        #print(state)
        
        #rewards[run] = rlg.total_reward()
        print("Completed run: "+str(run))
    return totalRewardsAtEachStep, rewards.mean()


def main():
    ##################################################################
    max_steps = 1000 # max number of steps in an episode
    num_runs = 2000  # number of repetitions of the experiment
    num_arms = 10 # number of bandit arms
    ##################################################################
    # Create and pass agent and environment objects to RLGlue
    agent = smartAgent()
    environment = envBase()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore

    #totalRewardsAtEachStep, result = experiment1(rlglue, num_runs, max_steps, num_arms)
    #print("experiment1 average reward: {}\n".format(result))
    
    
    #tempLenList = np.size(totalRewardsAtEachStep)
    #for i in range(0,tempLenList):
        #totalRewardsAtEachStep[i] = totalRewardsAtEachStep[i]/(num_runs)    
    
    #plt.plot(totalRewardsAtEachStep,'c', label='Experiment 1 (E = 0)')  
    
    totalRewardsAtEachStep, result = experiment2(rlglue, num_runs, max_steps, num_arms)
    print("experiment2 average reward: {}\n".format(result))    
    
    tempLenList = np.size(totalRewardsAtEachStep)
    for i in range(0,tempLenList):
        totalRewardsAtEachStep[i] = totalRewardsAtEachStep[i]/(num_runs)
    ####################################################################
    plt.plot(totalRewardsAtEachStep,'k', label='Experiment 2 (E = 0.1)')  
    plt.title("Experiments Rewards")
    plt.xlabel("Step number")
    plt.ylabel("% Optimal Choice")
    #plt.ylim(0,1)
    plt.legend(loc='lower right') 
    plt.show()    
    ####################################################################
    
if __name__ == '__main__':
    main()
