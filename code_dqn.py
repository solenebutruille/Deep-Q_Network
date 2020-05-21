import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

class DQN:

    REPLAY_MEMORY_SIZE = 10000 	#N number of tuples in experience replay  
    EPSILON = 0.9     					#Y epsilon of epsilon-greedy exploation (0.5)
    EPSILON_DECAY =  0.99 			#N exponential decay multiplier for epsilon (0.99) 0.999
    EPISODES_NUM = 2000 				#N number of episodes to train on. Ideally shouldn't take longer than 2000 (same) 
    MINIBATCH_SIZE = 40 			  #Y size of minibatch sampled from the experience replay (10)
    DISCOUNT_FACTOR = 1.0			  #N MDP's gamma (0.9) 0.95
    TARGET_UPDATE_FREQ = 100  	#N number of steps (not episodes) after which to update the target networks (100) 1000
    LOG_DIR = './logs' 					# directory wherein logging takes place 
    HIDDEN1_SIZE = 256 					#Y size of hidden layer 1 (128)
    HIDDEN2_SIZE = 128 					#Y size of hidden layer 2 (128)
    LEARNING_RATE = 0.0001 				# learning rate and other parameters for SGD/RMSProp/Adam (0.0001)  

    #Needs to plot the results
    x_axis = []
    y_axis = []
    mean_values = np.zeros(100)
         
    # Create and initialize the environment
    def __init__(self, env):
        self.env = gym.make(env) 
        assert len(self.env.observation_space.shape) == 1
        self.input_size = self.env.observation_space.shape[0]		# In case of cartpole, 4 state features
        self.output_size = self.env.action_space.n					# In case of cartpole, 2 actions (right/left)  

    # Create the Q-network
    def initialize_network(self): 

      ############################################################
      # Design your q-network here.
      # 
      # Add hidden layers and the output layer. For instance:
      # 
      # with tf.name_scope('output'):
      #	W_n = tf.Variable(
      # 			 tf.truncated_normal([self.HIDDEN_n-1_SIZE, self.output_size], 
      # 			 stddev=0.01), name='W_n')
      # 	b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')
      # 	self.Q = tf.matmul(h_n-1, W_n) + b_n
      #
      #############################################################
 
      X_input = Input((self.input_size,)) 
      X = Dense(512, input_shape=(self.input_size,), activation="relu", kernel_initializer='he_uniform')(X_input)
      X = Dense(self.HIDDEN1_SIZE, activation="relu", kernel_initializer='he_uniform')(X)
      X = Dense(self.HIDDEN2_SIZE, activation="relu", kernel_initializer='he_uniform')(X)
      X = Dense(self.output_size, activation="linear", kernel_initializer='he_uniform')(X)

      ############################################################
      # Next, compute the loss.
      #
      # First, compute the q-values. Note that you need to calculate these
      # for the actions in the (s,a,s',r) tuples from the experience replay's minibatch
      #
      # Next, compute the l2 loss between these estimated q-values and 
      # the target (which is computed using the frozen target network)
      #
      ############################################################
  
      loss = "mse"

      ############################################################
      # Finally, choose a gradient descent algorithm : SGD/RMSProp/Adam. 
      #
      # For instance:
      # optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
      # global_step = tf.Variable(0, name='global_step', trainable=False)
      # self.train_op = optimizer.minimize(self.loss, global_step=global_step)
      #
      ############################################################

      gradient = RMSprop(lr=self.LEARNING_RATE, rho=0.95, epsilon=0.01)

      self.model = Model(inputs = X_input, outputs = X, name='DQN_CartPole')
      self.model.compile(loss=loss, optimizer=gradient, metrics=["accuracy"])

      ############################################################ 
    
    def train(self, episodes_num=EPISODES_NUM):

      # Initialize summary for TensorBoard
      summary_writer = tf.summary.FileWriter(self.LOG_DIR)
      summary = tf.Summary()
      # Alternatively, you could use animated real-time plots from matplotlib
      # (https://stackoverflow.com/a/24228275/3284912)

      # Initialize the TF session
      self.session = tf.Session()
      self.session.run(tf.global_variables_initializer())

      ############################################################
      # Initialize other variables (like the replay memory)
      ############################################################

      cumulative_episode = 0
      self.memory = []
      epsi = self.EPSILON

      ############################################################
      # Main training loop
      #
      # In each episode,
      #	pick the action for the given state,
      #	perform a 'step' in the environment to get the reward and next state,
      #	update the replay buffer, sample a random minibatch from the replay buffer,
      # perform Q-learning, update the target network, if required. 
      #
      ############################################################

      for episode in range(self.EPISODES_NUM):
        
        state = self.env.reset()
        state = np.reshape(state, [1, self.input_size]) 

        ############################################################
        # Episode-specific initializations go here.
        ############################################################ 

        done = False
        episode_length = 0 

        ############################################################

        while not done:   

          ############################################################
          # Pick the next action using epsilon greedy and and execute it
          ############################################################

          if random.uniform(0, 1) > self.EPSILON :   
            action = np.argmax(self.model.predict(state))  
          else:  
            action = self.env.action_space.sample() 

          ############################################################
          # Step in the environment.  
          ############################################################

          next_state, reward, done, _ = self.env.step(action)
          next_state = np.reshape(next_state, [1, self.input_size]) 
          if done and episode_length != 200:
               reward = -100 

          ############################################################
          # Update the (limited) replay buffer.
          #
          # Note : when the replay buffer is full, you'll need to
          # remove an entry to accommodate a new one.
          ############################################################
          
          if len(self.memory) >= self.REPLAY_MEMORY_SIZE:
              indice = random.randint(0, len(self.memory)-1) 
              self.memory[indice] = [state, action, reward, next_state, done]
          else:
              self.memory.append((state, action, reward, next_state, done)) 
            
          state = next_state
          episode_length += 1 

          ############################################################
          # Sample a random minibatch and perform Q-learning (fetch max Q at s')
          #
          # Remember, the target (r + gamma * max Q) is computed
          # with the help of the target network.
          # Compute this target and pass it to the network for computing
          # and minimizing the loss with the current estimates
          #
          ############################################################

          if len(self.memory) >= self.TARGET_UPDATE_FREQ: 
            
            self.EPSILON *= self.EPSILON_DECAY
            # Taking random sample from memory to do the minibatch
            minibatch = random.sample(self.memory, min(len(self.memory), self.MINIBATCH_SIZE))

            states = np.zeros((self.MINIBATCH_SIZE, self.input_size))
            next_states = np.zeros((self.MINIBATCH_SIZE, self.input_size))
            actions, rewards, dones = [], [], []

            # assignement of values
            for i in range(self.MINIBATCH_SIZE):
                states[i] = minibatch[i][0]
                actions.append(minibatch[i][1])
                rewards.append(minibatch[i][2])
                next_states[i] = minibatch[i][3]
                dones.append(minibatch[i][4])

          ############################################################
            # Update target weights.
            #
            # Something along the lines of:
          # if total_steps % self.TARGET_UPDATE_FREQ == 0:
          # 	target_weights = self.session.run(self.weights)
          ############################################################ 
            target = self.model.predict(states)
            target_next = self.model.predict(next_states)

            #Updating q table
            for i in range(self.MINIBATCH_SIZE): 
                if dones[i]: target[i][actions[i]] = rewards[i]
                else: target[i][actions[i]] = rewards[i] + self.DISCOUNT_FACTOR * (np.amax(target_next[i]))

            # Train the Neural Network with batches
            self.model.fit(states, target, batch_size=self.MINIBATCH_SIZE, verbose=0) 

          ############################################################
          # Break out of the loop if the episode ends  (already in the while loop)
          ############################################################

        ############################################################
        # Logging.
        #
        # Very important. This is what gives an idea of how good the current
        # experiment is, and if one should terminate and re-run with new parameters
        # The earlier you learn how to read and visualize experiment logs quickly,
        # the faster you'll be able to prototype and learn.
        #
        # Use any debugging information you think you need.
        ############################################################
        if episode_length >= 195:  cumulative_episode += 1
        else:  cumulative_episode = 0

        print("Training: Episode: %d/%d, Length: %d, Cumulative episodes: %d" % (episode, self.EPISODES_NUM, episode_length, cumulative_episode)) 
        summary.value.add(tag="episode length", simple_value=episode_length)
        summary_writer.add_summary(summary, episode)
        
        self.x_axis.append(episode)
        self.mean_values[episode%100] = episode_length
        self.y_axis.append(np.mean(self.mean_values))

        if cumulative_episode >= 100:
            print("Problem solved")
            break 
      title = "DQN of cartpole-v0, ε = {} minibatchsize = {} hidden layer size : {}/{}".format(epsi, self.MINIBATCH_SIZE, self.HIDDEN1_SIZE, self.HIDDEN2_SIZE)
      plt.figure(num=None, figsize=(14, 8))
      plt.bar(self.x_axis, self.y_axis, 1.0)
      plt.title(title)     
      plt.ylabel("Versus average total reward of a 100-episode window")
      plt.xlabel("Learning curve of episodes")  
      plt.savefig("img.png")

    # Simple function to visually 'test' a policy
    def playPolicy(self):

      done = False
      steps = 0
      state = self.env.reset()

      # we assume the CartPole task to be solved if the pole remains upright for 200 steps
      while not done and steps < 200:
        state = np.reshape(state, [1, self.input_size]) 
        #not working on collab
   #     self.env.render()
        action = np.argmax(self.model.predict(state)) 
        state, _, done, _ = self.env.step(action)
        steps += 1

      return steps  

if __name__ == "__main__": 
    # Create and initialize the model
    dqn = DQN('CartPole-v0')
    dqn.initialize_network()

    print("\nStarting training...\n")
    dqn.train() 
    print("\nFinished training...\nCheck out some demonstrations\n")

    # Visualize the learned behaviour for a few episodes 
    results = []
    for i in range(50):
      episode_length = dqn.playPolicy()
      print("Test steps = ", episode_length)
      results.append(episode_length)
    print("Mean steps = ", sum(results) / len(results))	

    print("\nFinished.")
    print("\nCiao, and hasta la vista...\n")