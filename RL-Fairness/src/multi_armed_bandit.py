import numpy as np
import src.random


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000, num_bins=100):
        """
        Trains the MultiArmedBandit on an OpenAI Gym environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. For the step size, use
        1/N, where N is the number of times the current action has been
        performed. (This is the version of Bandits we saw in lecture before
        we introduced alpha). Use an epsilon-greedy approach to pick actions.

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
          - When choosing to exploit the best action rather than exploring,
            do not use np.argmax: it will deterministically break ties by
            choosing the lowest index of among the tied values. Instead,
            please *randomly choose* one of those tied-for-the-largest values.
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          avg_R - (np.array) A 1D sequence of averaged rewards of length
            `num_bins`. Let s = int(np.ceil(steps / `num_bins`)), then
            rewards[0] contains the average reward over the first s steps,
            rewards[1] contains the average of the next s steps, etc.
        """
        # set up Q function, rewards
        n_actions, n_states = env.action_space.n, env.observation_space.n
        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)
        avg_R = np.zeros([num_bins])

        all_rewards = []
        env.reset()
        j=0
        k=0
        total_R=0

        s = int(np.ceil(steps / num_bins))

        for i in range(steps):
          if i ==0:
           A=src.random.choice(n_actions)
          else:
            pick=src.random.uniform()
            if pick<=0.5:
              A=src.random.choice(n_actions)
            else:
              max=np.argwhere(self.Q==np.amax(self.Q))
              max_index=src.random.randint(len(max))
              A=max[max_index][0]
         
          A=int(A)

          val,R,done,time=env.step(A)
          self.N[A]=self.N[A]+1
          self.Q[A]=self.Q[A]+1/self.N[A]*(R-self.Q[A])

          if j==s-1:
            avg_R[k]=total_R/s
            k+=1
            j=0
            R=0
          else:
            total_R+=R
            j+=1
          if done==True:
            env.reset()
        state_action_values = np.tile(self.Q, (n_states, 1))
        return state_action_values, avg_R


    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. During prediction, you
            should only "exploit."
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
        # reset environment before your first action
        env.reset()
        states= np.array([])
        actions = np.array([])
        rewards = np.array([])

        max=np.argwhere(self.Q==np.amax(self.Q))
        max_index=src.random.randint(len(max))
        A=max[max_index][0]

        A=int(A)
        val,R,done,time=env.step(A)

        states=np.append(states,A)
        actions=np.append(actions,A)
        rewards=np.append(rewards,R)

        return states,actions,rewards
