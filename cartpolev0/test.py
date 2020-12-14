import random
import os
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import argparse
import json
from keras.models import model_from_json
from time import sleep
from cartpole import CartPoleEnv
import sys


from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

def map_state_to_discrete_states(state, no_intervals):
    x, v, a, v_a = state
    i = map_position(x, no_intervals)
    j = map_velocity(v, no_intervals)
    k = map_angle(a, no_intervals)
    l = map_angle_velocity(v_a, no_intervals)
    return [i, j, k, l]


def map_velocity(x, no_intervals):
    step = 2.3 / (no_intervals - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x < r:
            return i
    return no_intervals - 1


def map_position(x, no_intervals):
    step = 2.3 / (no_intervals - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x <= r:
            return i
    return no_intervals - 1


def map_angle(x, no_intervals):
    step = 2.3 / (no_intervals - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x <= r:
            return i
    return no_intervals - 1


def map_angle_velocity(x, no_intervals):
    step = 2.3 / (no_intervals - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x <= r:
            return i
    return no_intervals - 1



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
        print(q_values)
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

def check(state):
    x, v, a, v_a = state
    if x >= 0.4 and v >= 0.5:
        print('unsafe')
    elif x <= -0.4 and v <= -0.5:
        print('unsafe')
    elif a >= 0.0523599 and v_a >= 0.4:
        print('unsafe')
    elif a <= -0.0523599 and v_a <= -0.4:
        print('unsafe')
    

def check_discrete_state(state, r):
    x, v, a, v_a = state
    if x >= map_position(0.4, r) and v >= map_velocity(0.5, r):
        print('unsafe high position')
    elif x <= map_position(-0.4, r) and v <= map_velocity(-0.5, r):
        print('unsafe low position')
    elif a >= map_angle(0.0523599, r) and v_a >= map_angle_velocity(0.4, r):
        print('unsafe high angle')
    elif a <= map_angle(-0.0523599, r) and v_a <= map_angle_velocity(-0.4, r):
        print('unsafe low angle')
    

def cartpole(args):
    filename = 'models/model_' + str(args.total_steps) 
    filename = filename + '_' + str(args.no_intervals) if args.discrete else filename
    with open(filename + '.json', 'r') as f:
        model_json = json.load(f)

    model = model_from_json(model_json)
    model.load_weights(filename + '.h5')
    print(model.summary())

    score_logger = ScoreLogger(ENV_NAME)
    #env = gym.make(ENV_NAME)
    #env = gym.make(ENV_NAME)
    env = CartPoleEnv()
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    seed = np.random.randint(0, 200000)
    print('SEED', seed, '\n')
    #env.seed(seed)
    if args.x0:
        init = [args.x0, args.x1, args.x2, args.x3]
        state = env.reset(init)
    else:
        state = env.reset()
    #state = [0.02595507, 0.03749801, -0.00619774, -0.03514691]
    #state = [-0.03246117, 0.00904449, -0.02069081, -0.00731908]
    print('initial state', state)
    no_intervals = args.no_intervals

    step = 0
    timesteps = args.timesteps if args.timesteps else sys.maxsize
    arr = []
    while step <= timesteps:
        state = map_state_to_discrete_states(state, no_intervals) if args.discrete else state
        check_discrete_state(state, no_intervals)
        state = np.reshape(state, [1, observation_space])
        arr.extend([env.render(mode="rgb_array")])
        out = model.predict(state)[0]
        action = np.argmax(out)
        print('State:', step, state, 'Action:', action)
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        state = state_next
        sleep(0.1)
        step += 1
        if terminal:
            print(terminal, state, step)
            print("Score: " + str(step))
            value = score_logger.add_score(step, 1)
            break
            state = env.reset()
            print('initial state', state)
            #state = [0.02595507, 0.03749801, -0.00619774, -0.03514691]
            step = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', dest='total_steps', type=int, required=True)
    parser.add_argument('--no_intervals', dest='no_intervals', type=int)
    parser.add_argument('--timesteps', dest='timesteps', type=int)
    parser.add_argument('--x0', type=float)
    parser.add_argument('--x1', type=float)
    parser.add_argument('--x2', type=float)
    parser.add_argument('--x3', type=float)
    parser.add_argument('--discrete', action="store_true")
    parser.add_argument('--no-discrete', action="store_true")
    args = parser.parse_args()
    cartpole(args)
