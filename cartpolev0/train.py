import random
import os
import json
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import argparse


from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


def map_state_to_discrete_states(state, ranges):
    x, v, a, v_a = state
    i = map_position(x, ranges)
    j = map_velocity(v, ranges)
    k = map_angle(a, ranges)
    l = map_angle_velocity(v_a, ranges)
    return [i, j, k, l]


def map_velocity(x, ranges):
    step = 2.3 / (ranges - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x < r:
            return i
    return ranges - 1


def map_position(x, ranges):
    step = 2.3 / (ranges - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x <= r:
            return i
    return ranges - 1


def map_angle(x, ranges):
    step = 2.3 / (ranges - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x <= r:
            return i
    return ranges - 1


def map_angle_velocity(x, ranges):
    step = 2.3 / (ranges - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x <= r:
            return i
    return ranges - 1


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole(args):
    ranges = args.ranges
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    print(dqn_solver.model.summary())
    run = 0
    total_steps = 0
    value = 0
    while run < args.total_steps:
        run += 1
        state = env.reset()
        state = map_state_to_discrete_states(state, ranges)
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = map_state_to_discrete_states(state_next, ranges)
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                value = score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()
    model_name = 'model_' + str(args.total_steps) + '_' + str(ranges)
    model_json = dqn_solver.model.to_json()
    with open(os.path.join('models', model_name + '.json'), 'w') as json_file:
        json.dump(model_json, json_file)
    dqn_solver.model.save_weights(os.path.join('models', model_name + '.h5'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', dest='total_steps', type=int, required=True)
    parser.add_argument('--ranges', dest='ranges', type=int, required=True)
    args = parser.parse_args()
    cartpole(args)
