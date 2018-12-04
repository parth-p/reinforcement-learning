# reinforcement-learning </br>
1. cartpole-qn : 
  - Uses Q-learning method to train cartpole balancing problem in OpenAI environment. 
  - It creates a dictionary and stores q-values for all states and all corresponding actions.
  - Here bins are used to discretize the values of the observations for every state.
  - There are 4 observations (position, velocity, angular velocity, theta) and 2 actions (to move left and right).
  - While training, for each iteration we update 'q[state][action] += alpha*(reward + gamma*maxq_val_newstate - q[state][action])'
