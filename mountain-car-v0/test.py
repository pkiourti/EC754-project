import argparse
import os

import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from time import time
import keras as K
from mountain_car import MountainCarEnv
from time import sleep


def play_model(actor, initial_state):
    state = env.reset(initial_state)
    print(state)
    score = 0
    done = False
    images = []
    R = 0
    t = 0
    step = 0
    while not done:
        env.render()
        state = np.reshape(state, [-1, env.observation_space.shape[0]])
        print(step, state)
        action = actor.predict(state)
        nextState, reward, done, _ = env.step(np.argmax(action))
        state = nextState
        score += reward
        sleep(0.1)
        step += 1
        if done:
            return score
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', dest='total_steps', type=int, required=True)
    parser.add_argument('--no_intervals', dest='no_intervals', type=int)
    parser.add_argument('--x0', type=float)
    parser.add_argument('--x1', type=float)
    parser.add_argument('--x2', type=float)
    parser.add_argument('--x3', type=float)
    parser.add_argument('--discrete', action="store_true")
    parser.add_argument('--no-discrete', action="store_true")
    args = parser.parse_args()

    total_steps = args.total_steps
    no_intervals = args.no_intervals
    model = os.path.join('models', 'discrete') if args.discrete else 'models'
    model += os.path.join(str(total_steps), 'model_' + str(total_steps) + '_' + str(no_intervals) + '.h5')
    env = MountainCarEnv()
    initial_state = None
    if args.x0 is not None and args.x1 is not None and args.x2 is not None and args.x3 is not None:
        initial_state = [args.x0, args.x1, args.x2, args.x3]
    testScores = []
    actor = K.models.load_model('{}'.format(model))
    print(actor.summary())
    print("Saved model loaded from '{}'".format(model))
    score = play_model(actor, initial_state)
    print("Score: {}".format(score))
