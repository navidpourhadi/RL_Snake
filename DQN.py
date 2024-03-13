import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import trange
import numpy as np
import random

class QModel(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        super(QModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        with tf.device('/GPU:0'):  # Use '/GPU:0' or '/GPU:1' depending on your setup

            x = self.conv1(inputs)
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
        with tf.device('/GPU:0'):  # Use '/GPU:0' or '/GPU:1' depending on your setup
        
            x = self.conv1(inputs)
            x = self.flatten(x)
            x = self.dense1(x)
            return self.output_layer(x)

class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.num_actions = num_actions
        self.q_model = QModel(input_shape, num_actions)
        self.target_q_model = QModel(input_shape, num_actions)
        self.target_q_model.set_weights(self.q_model.get_weights())
        self.optimizer_q = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.gamma = 0.9
        self.replay_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': []
        }

        self.v_model = ValueModel(input_shape)
        self.optimizer_v = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def select_action(self, state):
        with tf.device('/GPU:0'):  # Use '/GPU:0' or '/GPU:1' depending on your setup
            q_values = self.q_model.predict(tf.expand_dims(state, axis=0))[0]
            return np.argmax(q_values)        
        
    def store_transition(self, state, action, reward, next_state):
        with tf.device('/GPU:0'):  # Use '/GPU:0' or '/GPU:1' depending on your setup

            self.replay_buffer['states'].append(state)
            self.replay_buffer['actions'].append(action)
            self.replay_buffer['rewards'].append(reward)
            self.replay_buffer['next_states'].append(next_state)

    def experience_replay_q(self, batch_size):
        with tf.device('/GPU:0'):  # Use '/GPU:0' or '/GPU:1' depending on your setup

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



    def experience_replay_v(self, batch_size):
        with tf.device('/GPU:0'):  # Use '/GPU:0' or '/GPU:1' depending on your setup
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
        with tf.device('/GPU:0'):  # Use '/GPU:0' or '/GPU:1' depending on your setup
        
            self.target_q_model.set_weights(self.q_model.get_weights())
