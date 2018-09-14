from rl_glue import BaseEnvironment
import random
import decimal
import numpy as np

class envBase(BaseEnvironment):
    """
    Example 1-Dimensional environment
    """

    def __init__(self):
        """Declare environment variables."""

        # state we are in currently
        self.currentState = [1,0,0] #0 means we are currently starting in exploit mode starting with arm 0

        # possible actions
        self.actions = [-1, 1]
        
        self.num_arms = None
        
        self.max_steps = None
        
        self.armRewardChance = None
        

    def env_init(self):
        """
        Initialize environment variables.
        """
        pass

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        

        return self.currentState

    def env_step(self, state):
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        #print("Pulling arm: "+str(state[1]))
        state[2] = np.random.normal(self.armRewardChance[state[1]])
            
        self.currentState = state
        terminal = False

        return state[2], self.currentState, terminal

    def env_message(self, message):
        if 'num_arms' in message:
            self.num_arms = int(message.split()[1])
            self.armRewardChance = np.random.normal(0,1,self.num_arms)
            bestArm = np.argmax(self.armRewardChance)
            return bestArm
            #print(self.armRewardChance)
            
        if 'max_steps' in message:
            self.max_steps = int(message.split()[1])
