import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tqdm import trange
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler

WEIGHT_PATH = "./weights/"


def lr_schedule(epoch, lr):
    # Example learning rate decay schedule
    if epoch < 50:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def display_boards(env, n=5):
    
    fig,axs=plt.subplots(1,min(len(env.boards), n), figsize=(5,5))
    for ax, board in zip(axs, env.boards):
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.imshow(board, origin="lower")

class QModel(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        super(QModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        with tf.device('/GPU:0'): 

            x = self.conv1(inputs)
            x = self.conv2(x)
            x = self.flatten(x)
            x = self.dense1(x)
            return self.output_layer(x)
        
class QModel2(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        super(QModel2, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape= input_shape)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)    


class DQNAgent:
    def __init__(self, input_shape, num_actions, gamma=0.9, max_buffer_size = 50000, epsilon_initial=1.0, epsilon_final=0.1, epsilon_decay=0.95):
        self.epsilon = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.max_buffer_size = max_buffer_size
        self.gamma = gamma
        self.num_actions = num_actions
        self.q_model = QModel(input_shape, num_actions)
        self.target_q_model = QModel(input_shape, num_actions)
        self.target_q_model.set_weights(self.q_model.get_weights())
        self.learning_rate = 1e-4
        self.optimizer_q = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.replay_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)
        
    def to_channels(self, state):
        
        num_boards = state.shape[0]
        board_dim = state.shape[1]
        return np.transpose(state, (0,3,1,2)).reshape(num_boards, 4, board_dim, board_dim)

    def select_actions_exploration(self, states):
        with tf.device('/GPU:0'):
            actions = []
            for state in states:
                self.update_epsilon()
                if np.random.rand() < self.epsilon:
                    actions.append(np.random.randint(self.num_actions))  # Choose random action
                else:
                    
                    q_values = self.q_model.predict(np.expand_dims(state, axis=0), verbose=None)[0]
                    actions.append(np.argmax(q_values))  # Choose best action
            return np.array(actions)


    def select_actions_exploitation(self, states):
        with tf.device('/GPU:0'):
            q_values = self.q_model.predict(np.array(states), verbose=None)  # Predict Q-values for all states
            print("q_values: ",q_values)
            actions = np.argmax(q_values, axis=1)  # Choose the action with the maximum Q-value for each state
            print("actions: ",actions)
            return actions

    # select action based on softmax policy for all states
    def select_actions_softmax(self, states):
        with tf.device('/GPU:0'):
            q_values = self.q_model.predict(np.array(states), verbose=None)  # Predict Q-values for all states
            probs = np.exp(q_values) / np.sum(np.exp(q_values), axis=1, keepdims=True)
            actions = np.random.choice(self.num_actions, p=probs[0], size=len(states))
            return actions



    def store_transition(self, state, action, reward, next_state, done):
        
        with tf.device('/GPU:0'): 
            self.replay_buffer['states'].append(state)
            self.replay_buffer['actions'].append(action)
            self.replay_buffer['rewards'].append(reward)
            self.replay_buffer['next_states'].append(next_state)
            self.replay_buffer['dones'].append(done)

            excess = len(self.replay_buffer['states']) - self.max_buffer_size
            if excess > 0:
                del self.replay_buffer['states'][:excess]
                del self.replay_buffer['actions'][:excess]
                del self.replay_buffer['rewards'][:excess]
                del self.replay_buffer['next_states'][:excess]
                del self.replay_buffer['dones'][:excess]


    def experience_replay_q(self, batch_size):
        with tf.device('/GPU:0'): 

            buffer_size = len(self.replay_buffer['states'])
            indices = np.arange(buffer_size)
            sampled_indices = np.random.choice(indices, size=min(len(indices),batch_size), replace=False)

            states = np.array(self.replay_buffer['states'])[sampled_indices]
            actions = np.array(self.replay_buffer['actions'])[sampled_indices]
            rewards = np.array(self.replay_buffer['rewards'])[sampled_indices]
            next_states = np.array(self.replay_buffer['next_states'])[sampled_indices]
            dones = np.array(self.replay_buffer['dones'])[sampled_indices]

            # Compute target Q-values
            targets = rewards.copy()  # Initialize targets with rewards
            non_terminal_indices = np.where(dones == 0)[0]
            if len(non_terminal_indices) > 0:
                non_terminal_next_states = next_states[non_terminal_indices]
                non_terminal_q_values = self.target_q_model.predict(non_terminal_next_states)
                max_q_values = np.amax(non_terminal_q_values, axis=1, keepdims=True)
                targets[non_terminal_indices] += self.gamma * max_q_values

            # shape: (batch_size, 1) : reward of the current state and the max expected reward of the next state
            # targets = rewards + self.gamma * tf.reduce_max(self.target_q_model(next_states), axis=1, keepdims=True)
            with tf.GradientTape() as tape:
                q_values = tf.gather(self.q_model(states), actions, axis=1, batch_dims=1)
                loss = self.mse(q_values, targets)
            gradients = tape.gradient(loss, self.q_model.trainable_variables)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
            gradients = [tf.convert_to_tensor(g) for g in clipped_gradients]
            self.optimizer_q.apply_gradients(zip(gradients, self.q_model.trainable_variables))
            return loss


    def update_target_q_model(self):
        with tf.device('/GPU:0'): 
            self.target_q_model.set_weights(self.q_model.get_weights())


    def save_weights(self, q_model_path):
        self.q_model.save_weights(q_model_path)
        # self.v_model.save_weights(v_model_path)

    def load_weights(self, q_model_path):
        try :
            self.q_model.load_weights(q_model_path)
            # self.v_model.load_weights(v_model_path)
        except:
            print("The weights do not exist")
            return
        
        
    def train(self, env, num_epochs, batch_size):
        with tf.device('/GPU:0'): 
            for epoch in range(num_epochs):
                self.learning_rate = lr_schedule(epoch, self.learning_rate)
                self.optimizer_q = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
                state = env.to_state()
                fruit_location_before = np.argwhere(env.boards == env.FRUIT)
                actions = self.select_actions_softmax(state).reshape(-1, 1)
                rewards = env.move(actions)
                next_state = env.to_state()
                fruit_location_after = np.argwhere(env.boards == env.FRUIT)

                for i in range(env.n_boards):
                    done = 0 if np.array_equal(fruit_location_before[i], fruit_location_after[i]) else 1
                    self.store_transition(state[i], actions[i], rewards[i].numpy(), next_state[i], done)
                
                # Experience replay for Q-learning to update the DQN model
                loss = self.experience_replay_q(batch_size)
                print(f"Epoch {epoch + 1}/{num_epochs} => loss: {loss.numpy()}, Reward Mean: {np.mean(rewards)}")

                if (epoch + 1) % 50 == 0:
                    self.update_target_q_model()            
                    self.save_weights(WEIGHT_PATH+"q_model.h5")
                    

    def play(self, env, steps=1000):
        with tf.device('/GPU:0'):
            # load the weights
            # self.load_weights(WEIGHT_PATH+"q_model.h5", WEIGHT_PATH+"v_model.h5") 
            fruits = np.zeros(env.n_boards, dtype=int)

            rewards = np.zeros(env.n_boards, dtype=float)[:,None]
            for _ in trange(steps):
                print("iteration: ", _)
                # update the coordination of fruits in each board
                # fruit_before = np.argwhere(env.boards == env.FRUIT)
                state = env.to_state()
                actions = self.select_actions_exploitation(state).reshape(-1, 1)
                reward = env.move(actions)
                # fruit_after = np.argwhere(env.boards == env.FRUIT)
                # diff = [np.array_equal(fruit_before[i], fruit_after[i]) for i in range(env.n_boards)]
                # increment the fruit count in boards that have different fruit locations
                # fruits = np.array([fruits[i] + 1 if not diff[i] else fruits[i] for i in range(env.n_boards)])
                
                # print(fruits)

                rewards = rewards + reward
                print("rewards: ", rewards)
                display_boards(env, 5)

