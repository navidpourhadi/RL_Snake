import tensorflow as tf
from tqdm import trange
import numpy as np
import utils
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras import regularizers
import time
from collections import deque
import random

WEIGHT_PATH = "./weights/"
MAX_BUFFER_SIZE = 20000


class DoubleDQNAgent:
    def __init__(self, input_shape, num_actions, gamma=0.9, epsilon_initial=1.0, epsilon_final=0.1, epsilon_decay=0.99, lr_initial=1e-4):
        self.input_shape = input_shape
        self.epsilon = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.num_actions = num_actions
        self.lr_initial = lr_initial
        self.q_model = self.build_q_model(input_shape, num_actions)
        self.target_q_model = self.build_q_model(input_shape, num_actions)
        self.target_q_model.set_weights(self.q_model.get_weights())
        self.optimizer_q = tf.keras.optimizers.Adam(learning_rate=self.lr_initial)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)

        print(self.q_model.summary())


    def build_q_model(self, input_shape, num_actions):
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2), strides=2),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
            MaxPooling2D((2, 2), strides=2),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.2),
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(num_actions, activation='linear')
        ])
        return model


    def update_epsilon(self):
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)


    def select_actions_exploration(self, states):
        self.update_epsilon()
        random_actions = np.random.randint(0, self.num_actions, size=len(states))
        greedy_actions = np.argmax(self.q_model.predict(states, verbose=0), axis=1)
        mask = np.random.rand(len(states)) < self.epsilon
        actions = np.where(mask, random_actions, greedy_actions)
        return actions

    def select_actions_exploitation(self, states):
        actions = np.argmax(self.q_model.predict(states, verbose=0), axis=1)
        return actions

    def select_actions_exploitation_target(self, states):
        actions = np.argmax(self.target_q_model.predict(states, verbose=0), axis=1)
        return actions



    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def experience_replay_q(self, batch_size):
        batch = random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        targets = rewards.copy()

        non_terminal_indices = np.where(dones == 0)[0]
        if len(non_terminal_indices) > 0:
            non_terminal_next_states = next_states[non_terminal_indices]
            target_q_values = self.target_q_model.predict(non_terminal_next_states, verbose=0)
            max_actions = np.argmax(self.q_model.predict(non_terminal_next_states, verbose=0), axis=1)
            targets[non_terminal_indices] += self.gamma * target_q_values[np.arange(len(target_q_values)), max_actions].reshape(-1, 1)

        with tf.GradientTape() as tape:
            q_values = tf.gather(self.q_model(states), actions, axis=1, batch_dims=1)
            loss = self.mse(q_values, targets)

        gradients = tape.gradient(loss, self.q_model.trainable_variables)
        self.optimizer_q.apply_gradients(zip(gradients, self.q_model.trainable_variables))

    def update_target_q_model(self):
        self.target_q_model.set_weights(self.q_model.get_weights())



    def save_weights(self, file_prefix="ddqn_"):
        q_model_path = WEIGHT_PATH + file_prefix + str(self.input_shape[0]) +"_q_model.h5"
        target_q_model_path = WEIGHT_PATH + file_prefix + str(self.input_shape[0]) +"_target_q_model.h5"
        self.q_model.save_weights(q_model_path)
        self.target_q_model.save_weights(target_q_model_path)
        print("Weights saved.")

    def load_weights(self, file_prefix="ddqn_"):
        q_model_path = WEIGHT_PATH + file_prefix + str(self.input_shape[0]) + "_q_model.h5"
        target_q_model_path = WEIGHT_PATH + file_prefix + str(self.input_shape[0]) +"_target_q_model.h5"
        self.q_model.load_weights(q_model_path)
        self.target_q_model.load_weights(target_q_model_path)
        print("Weights loaded.")


    def train(self, env, num_epochs, batch_size):
        for epoch in range(num_epochs):
            T1 = time.time()
            # self.update_learning_rate(epoch)

            states = env.to_state()
            # actions = np.array([self.select_actions_exploration([state])[0] for state in states]).reshape(-1,1)
            actions = self.select_actions_exploration(states).reshape(-1, 1)
            rewards = env.move(actions)
            next_states = env.to_state()

            for i in range(env.n_boards):
                done = 0 if np.array_equal(states[i], next_states[i]) else 1
                self.store_transition(states[i], actions[i], rewards[i], next_states[i], done)
            self.experience_replay_q(batch_size)
            T2 = time.time()
            
            print("Epoch: {}/{} | Epsilon: {:.3f} | Time: {:.3f}s".format(epoch + 1, num_epochs, self.epsilon, T2 - T1))
            # print("Epoch: {}/{} | Epsilon: {:.3f}".format(epoch + 1, num_epochs, self.epsilon))

            if (epoch + 1) % 20 == 0:
                self.update_target_q_model()

        self.save_weights("Temp_DDQN")        


    def play(self, env, steps=1000):

        ask_loading = input("DO you want to load the final weights of the model for the play? (y/n): ")

        if ask_loading == "y":
            file_name_prefix = input("Enter the weight file name prefix:  (e.g. filename: 'Temp_DDQN_15_dqn_model.h5' then prefix: 'Temp_DDQN') ")
            self.load_weights(file_name_prefix)

        rewards = np.zeros(env.n_boards, dtype=float)[:, None]

        for _ in trange(steps):
            states = env.to_state()
            # actions = [self.select_actions_exploitation([state])[0] for state in states].reshape(-1, 1)
            actions = self.select_actions_exploitation(states).reshape(-1, 1)
            reward = env.move(actions)

            rewards = rewards + reward
            print("Step: {}/{} | Reward: {:.3f}".format(_, steps, np.mean(rewards)))
            utils.display_boards(env, 5)


    def play_target(self, env, steps=1000):
        # fruits = np.zeros(env.n_boards, dtype=int)
        rewards = np.zeros(env.n_boards, dtype=float)[:, None]

        for _ in trange(steps):
            states = env.to_state()
            # actions = [self.select_actions_exploitation([state])[0] for state in states].reshape(-1, 1)
            actions = self.select_actions_exploitation_target(states).reshape(-1, 1)
            reward = env.move(actions)

            rewards = rewards + reward
            if _ % 10 == 0:
                print("Step: {}/{} | Reward: {:.3f}".format(_, steps, np.mean(rewards)))
                utils.display_boards(env, 5)
        print("Mean Reward: {:.3f}".format(np.mean(rewards)))                
