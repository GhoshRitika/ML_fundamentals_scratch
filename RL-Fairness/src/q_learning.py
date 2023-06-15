import numpy as np
import src.random


class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      alpha - (float) The weighting to give current rewards in estimating Q. This 
        should range [0,1], where 0 means "don't change the Q estimate based on 
        current reward" 
      gamma - (float) This is the weight given to expected future rewards when 
        estimating Q(s,a). It should be in the range [0,1]. Here 0 means "don't
        incorporate estimates of future rewards into the reestimate of Q(s,a)"

      See page 131 of Sutton and Barto's Reinforcement Learning book for
        pseudocode and for definitions of alpha, gamma, epsilon 
        (http://incompleteideas.net/book/RLbook2020.pdf).  
    """

    def __init__(self, epsilon=0.2, alpha=0.5, gamma=0.5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, env, steps=1000, num_bins=100):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.

        See page 131 of Sutton and Barto's book Reinforcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. Choose actions with
        an epsilon-greedy approach Note that unlike the pseudocode, we are
        looping over a total number of steps, and not a total number of
        episodes. This allows us to ensure that all of our trials have the same
        number of steps--and thus roughly the same amount of computation time.

        See (https://www.gymlibrary.ml) for examples of how to use the OpenAI
        Gym Environment interface.

        In every step of the fit() function, you should sample
            two random numbers using functions from `src.random`.
            1.  First, use either `src.random.rand()` or `src.random.uniform()`
                to decide whether to explore or exploit.
            2. Then, use `src.random.choice` or `src.random.randint` to decide
                which action to choose. Even when exploiting, you should make a
                call to `src.random` to break (possible) ties.

        Please don't use `np.random` functions; use the ones from `src.random`!
        Please do not use `env.action_space.sample()`!

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "done" returned
            from env.step() is True.
          - In addition to resetting the environment, calling env.reset() will
            return the environment's initial state.
          - When choosing to exploit the best action rather than exploring,
            do not use np.argmax: it will deterministically break ties by
            choosing the lowest index of among the tied values. Instead,
            please *randomly choose* one of those tied-for-the-largest values.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://www.gymlibrary.ml/content/api/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          avg_rewards - (np.array) A 1D sequence of averaged rewards of length
            `num_bins`. Let s = int(np.ceil(steps / `num_bins`)), then
            rewards[0] contains the average reward over the first s steps,
            rewards[1] contains the average of the next s steps, etc.
        """
        # set up rewards list, Q(s, a) table
        n_actions, n_states = env.action_space.n, env.observation_space.n
        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)
        avg_rewards = np.zeros([num_bins])
        all_rewards=[]
        # reset environment before your first action
        env.reset()
        j=0
        k=0
        total_rewards=0
        s = int(np.ceil(steps / num_bins))

        for i in range(steps):
          pick=src.random.uniform()
          if pick<=0.5:
            A=src.random.choice(n_actions)
          else:
            max=np.argwhere(self.Q==np.amax(self.Q))
            max_index=src.random.randint(len(max))
            A=max[max_index][0]
         
          A=int(A)
          val,R,done,time=env.step(A)

          self.Q[A]+=self.alpha *(R-self.gamma * self.Q[A]-self.Q[A])
          if j==s-1:
            avg_rewards[k]=total_rewards/s
            k+=1
            j=0
            R=0
          else:
            total_rewards+=R
            j+=1
          if done==True:
            env.reset()

        state_action_values = np.tile(self.Q, (n_states, 1))
        return state_action_values, avg_rewards



    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. During prediction, you
            should only "exploit."
          - In addition to resetting the environment, calling env.reset() will
            return the environment's initial state
          - You should use a loop to predict over each step in an episode until
            it terminates by returning `done=True`.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://www.gymlibrary.ml/content/api/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        n_actions, n_states = env.action_space.n, env.observation_space.n

        states= np.array([0])
        actions = np.array([])
        rewards = np.array([])
        env.reset()

        max=np.argwhere(self.Q==np.amax(self.Q))
        max_index=src.random.randint(len(max))
        A=max[max_index][0]
        A=int(A)
        val,R,done,time=env.step(A)
        A=np.append(actions,A)
        rewards=np.append(rewards,R)

        return states,actions,rewards

