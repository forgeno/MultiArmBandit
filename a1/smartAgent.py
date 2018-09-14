import numpy as np
from rl_glue import BaseAgent
import random


class smartAgent(BaseAgent): 
    """
    simple random agent, which moves left or right randomly in a 2D world

    Note: inheret from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""

        # Your agent may need to remember what the action taken was.
        # In this case the variable is not used.
        self.currentState = None

        # Your agent may have a policy for choosing actions. This agent will
        # choose action 0 with probability self.prob0 and action 1 with
        # probability 1-self.prob0
        self.stepSize = None

        self.initalReward = None
        
        self.epsilon = None
        
        self.num_arms = None
        
        self.rewardArms = np.array([])
        
    def agent_init(self):
        """Initialize agent variables."""
        pass

    def _choose_action(self, state):
        """
        Convenience function.

        You are free to define whatever internal convenience functions
        you want, you just need to make sure that the RLGlue interface
        functions are also defined as well.
        """
        
        # 0 = explore mode
        # 1 = exploit mode
        state[0] = 0  #Default is in explore mode
        chanceOfExploit = float(random.randrange(0, 100))/100
        #print("Chance of exploring is: "+str(chanceOfExploit)+" compared to the needed: "+str(self.epsilon))
        if(chanceOfExploit > self.epsilon):
            #print("Exploit mode!")
            state[0] = 1 #exploit mode
            checkRepeat = np.where(self.rewardArms == self.rewardArms.max())[0]
            #print("checkRepeat: "+str(checkRepeat))
            if(np.size(checkRepeat) > 1):
                pickRandom = random.randint(0,np.size(checkRepeat)-1)
                state[1] = checkRepeat[pickRandom]
            else:
                state[1] = checkRepeat[0]
            state[1] = np.argmax(self.rewardArms)
            #print("Selecting arm: "+str(state[1]))
        else:
            #print("Explore mode!")
            state[1] = random.randint(0,(self.num_arms-1)) #if arms are 10 then we must select arms between 0-9
        
        
        #print(state)
        return state

    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (state observation): The agent's current state

        Returns:
            The first action the agent takes.
        """
        
        # This agent doesn't care what state it's in, it always chooses
        # to move left or right randomly according to self.probLeft
        self.rewardArms = np.array([])
        for i in range(0, self.num_arms):
            self.rewardArms = np.append(self.rewardArms,self.initalReward)
        #print(self.rewardArms)
        self.currentState = self._choose_action(state)
        return self.currentState

    def agent_step(self, reward, state):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (state observation): The agent's current state
        Returns:
            The action the agent is taking.
        """
        # Updates table of rewards
        self.rewardArms[state[1]] += self.stepSize*(state[2] - self.rewardArms[state[1]]) #Formula for calculating average reward of each arm
        #print(self.rewardArms)
        #print(self.rewardArms)
        self.currentState = self._choose_action(state)

        return self.currentState

    def agent_end(self, reward):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # random agent doesn't care about reward
        pass

    def agent_message(self, message):
        if 'epsilon' in message:
            self.epsilon = float(message.split()[1])
        if 'initalReward' in message:
            self.initalReward = int(message.split()[1])
        if 'stepSize' in message:
            self.stepSize = float(message.split()[1])          
        if 'num_arms' in message:
            self.num_arms = int(message.split()[1])        