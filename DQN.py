import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import trange
import numpy as np
import random
import os
import matplotlib.pyplot as plt


WEIGHT_PATH = "./weights/"

class QModel(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        super(QModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        with tf.device('/GPU:0'): 

            x = self.conv1(inputs)
            x = self.conv2(x)
            x = self.flatten(x)
            x = self.dense1(x)
            return self.output_layer(x)

class ValueModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(ValueModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        with tf.device('/GPU:0'): 
        
            x = self.conv1(inputs)
            x = self.flatten(x)
            x = self.dense1(x)
            return self.output_layer(x)

class DQNAgent:
    def __init__(self, input_shape, num_actions, gamma=0.9, max_buffer_size = 10000, epsilon_initial=1.0, epsilon_final=0.1, epsilon_decay=0.99):
        self.epsilon = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.max_buffer_size = max_buffer_size
        self.gamma = gamma
        self.num_actions = num_actions
        self.q_model = QModel(input_shape, num_actions)
        self.target_q_model = QModel(input_shape, num_actions)
        self.target_q_model.set_weights(self.q_model.get_weights())
        self.optimizer_q = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.replay_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': []
        }

        self.v_model = ValueModel(input_shape)
        self.optimizer_v = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)
        
        
    def select_actions(self, states):
        with tf.device('/GPU:0'):
            actions = []
            for state in states:
                if np.random.rand() < self.epsilon:
                    actions.append(np.random.randint(self.num_actions))  # Choose random action
                else:
                    q_values = self.q_model.predict(np.expand_dims(state, axis=0))[0]
                    actions.append(np.argmax(q_values))  # Choose best action
            return np.array(actions)



    def store_transition(self, state, action, reward, next_state):
        with tf.device('/GPU:0'): 
            self.replay_buffer['states'].append(state)
            self.replay_buffer['actions'].append(action)
            self.replay_buffer['rewards'].append(reward)
            self.replay_buffer['next_states'].append(next_state)

            excess = len(self.replay_buffer['states']) - self.max_buffer_size
            if excess > 0:
                del self.replay_buffer['states'][:excess]
                del self.replay_buffer['actions'][:excess]
                del self.replay_buffer['rewards'][:excess]
                del self.replay_buffer['next_states'][:excess]


    def experience_replay_q(self, batch_size):
        with tf.device('/GPU:0'): 

            buffer_size = len(self.replay_buffer['states'])
            indices = np.arange(buffer_size)
            sampled_indices = np.random.choice(indices, size=min(len(indices),batch_size), replace=False)

            states = np.array(self.replay_buffer['states'])[sampled_indices]
            actions = np.array(self.replay_buffer['actions'])[sampled_indices]
            rewards = np.array(self.replay_buffer['rewards'])[sampled_indices]
            next_states = np.array(self.replay_buffer['next_states'])[sampled_indices]

            targets = tf.reduce_mean(rewards + self.gamma * tf.reduce_max(self.target_q_model(next_states), axis=1, keepdims=True), axis=0)
            
            with tf.GradientTape() as tape:
                q_values = self.q_model(states)
                selected_q_values = tf.reduce_sum(tf.one_hot(actions, self.num_actions) * q_values, axis=1)
                loss = tf.reduce_mean(tf.square(selected_q_values - targets))
            gradients = tape.gradient(loss, self.q_model.trainable_variables)
            gradients = [tf.convert_to_tensor(g) for g in gradients]  # Convert NumPy arrays to TensorFlow tensors
            self.optimizer_q.apply_gradients(zip(gradients, self.q_model.trainable_variables))
            return loss

    def experience_replay_v(self, batch_size):
        with tf.device('/GPU:0'): 
            buffer_size = len(self.replay_buffer['states'])
            indices = np.arange(buffer_size)
            sampled_indices = np.random.choice(indices, size=batch_size, replace=False)

            states = np.array(self.replay_buffer['states'])[sampled_indices]
            rewards = np.array(self.replay_buffer['rewards'])[sampled_indices]

            with tf.GradientTape() as tape:
                v_values = self.v_model(states)
                loss = tf.reduce_mean(tf.square(rewards - v_values[:, 0]))

            gradients = tape.gradient(loss, self.v_model.trainable_variables)
            self.optimizer_v.apply_gradients(zip(gradients, self.v_model.trainable_variables))

    def update_target_q_model(self):
        with tf.device('/GPU:0'): 
        
            self.target_q_model.set_weights(self.q_model.get_weights())


    def save_weights(self, q_model_path, v_model_path):
        self.q_model.save_weights(q_model_path)
        self.v_model.save_weights(v_model_path)

    def load_weights(self, q_model_path, v_model_path):
        try :
            self.q_model.load_weights(q_model_path)
            self.v_model.load_weights(v_model_path)
        except:
            print("The weights do not exist")
            return
        
        
    def train(self, env, num_epochs, batch_size):
        with tf.device('/GPU:0'): 
            for epoch in range(num_epochs):
                state = env.to_state()

                actions = self.select_actions(state).reshape(-1, 1)
                rewards = env.move(actions)
                for i in range(env.n_boards):
                    self.store_transition(state[i], actions[i], rewards[i].numpy(), env.to_state()[i])

                # Experience replay for Q-learning to update the DQN model
                loss = self.experience_replay_q(batch_size)
                # self.experience_replay_v(batch_size)
                print(f"Epoch {epoch}/{num_epochs} => loss: {loss.numpy()}, Reward Mean: {np.mean(rewards)}")

                if epoch % 10 == 0:
                    self.update_target_q_model()            
                    self.save_weights(WEIGHT_PATH+"q_model.h5", WEIGHT_PATH+"v_model.h5")
                    

    def play(self, env, steps=1000):
        with tf.device('/GPU:0'):         
            rewards = np.zeros(env.n_boards, dtype=float)
            for step in trange(steps):
                state = env.to_state()
                actions = self.select_actions(state).reshape(-1, 1)
                rewards = rewards + env.move(actions)
                print("rewards: ", rewards)
