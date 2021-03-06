# Reinforcement-learning </br>
## 1. cartpole-qn :
* Uses Q-learning method to train cartpole balancing problem in OpenAI environment. 
* It creates a dictionary and stores q-values for all states and all corresponding actions.
* Here bins are used to discretize the values of the observations for every state.
* There are 4 observations (position, velocity, angular velocity, theta) and 2 actions (to move left and right).
* While training, for each iteration we update`q[state][action] += alpha*(reward + gamma*maxq_val_newstate - q[state][action])`.

## 2. cartpole-dqn :
* It is a pytorch implementation of Deep Q Networks to train cartpole balancing problem in OpenAI environment.
* It uses replay buffer to save state, action, reward and next state which is then used to train the network.
* It is a simple network of 3 layers and final layer consists of 2 neurons.
* Update policy :
~~~~
expected_q_value = reward + gamma * next_q_value * (1 - done)
loss = (q_value - expected_q_value.data).pow(2).mean()
~~~~
* For implementing Double DQN, 2 models are initialized and update policy changes to :
~~~~
q_values = model1(state)
next_q_values = model1(next_state)
next_q_values2 = model2(next_state)
q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
next_q_value = next_q_values2.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
expected_q_value = reward + gamma * next_q_value * (1 - done)
loss = (q_value - expected_q_value.data).pow(2).mean()
~~~~
* And model 1 is synchronized with model 2 periodically

## 3. atari-dqn :
* It is a pytorch implementation of Deep Q Networks to train atari games in OpenAI environment.
* Here observations are frrames of 210x160x3 but we reshape them to 80x80 frames.
* Input to network is 4 such frames stacked together and output is 6 actions [0, 1,.....5] and the network used here is convolutional neural network.
* For double DQN, update policy is similar as in above case.

## 4. dueling-DQN :
* It is a pytorch implementation of Dueling DQN to train cartpole problem in OpenAI environment.
![alt text](dueling.png "Network")
* Here prioritized memory replay is used and while update, the samples taken from memory are in prioritized order given by the error at that step. Data structure used here is Sum tree.
* It works for atari problems as well if linear layers are converted to cnn and network is changed accordingly.
