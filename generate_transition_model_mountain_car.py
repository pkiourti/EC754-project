import gym
import numpy as np
import time
from time import sleep
from tqdm import tqdm
import random
import argparse
import os
from mountain_car import MountainCarEnv
import json


def parser():
    parser = argparse.ArgumentParser('create a transition model from mdp')
    parser.add_argument('--no_intervals', type=int, required=True)
    parser.add_argument('--episodes', type=int, required=True)
    return parser


def map_position(x, no_intervals):
    step1 = 1.8 / (no_intervals - 1)
    for i, r in enumerate(np.arange(-1.2, 0.6, step1)):
        if x <= r:
            return i
    return no_intervals - 1


def map_velocity(x, no_intervals):
    step2 = 0.14 / (no_intervals - 1)
    for i, r in enumerate(np.arange(-0.07, 0.07, step2)):
        if x <= r:
            return i
    return no_intervals - 1


def map_state_to_a_number(state, no_intervals):
    x, v = state
    i = map_position(x, no_intervals)
    j = map_velocity(v, no_intervals)
    return i*no_intervals + j


def map_number_to_state(number, no_intervals):
    l = number % no_intervals
    k = (number // no_intervals) % no_intervals
    return i, j


def run_policy(env, initial_state, no_episodes, no_intervals, a):
    action_space = env.action_space.n
    state = env.reset(initial_state)
    episodes = 0
    step = 0
    while episodes < no_episodes:
        current_discrete = map_state_to_a_number(state, no_intervals)
        action = a if step == 0 else np.random.randint(0, action_space) 
        #print(action)
        state, reward, terminal, info = env.step(action)
        step += 1
        next_discrete = map_state_to_a_number(state, no_intervals)
        #print('current_discrete', current_discrete)#, next_discrete, action)
        discrete_states[action, current_discrete, next_discrete] += 1
        if terminal:
            state = env.reset(initial_state)
            episodes += 1
            step = 0


if __name__ == '__main__':
    args = parser().parse_args()
    no_intervals = args.no_intervals
    episodes = args.episodes
    
    env = MountainCarEnv()
    action_space = env.action_space.n
    
    no_discrete_states = no_intervals * no_intervals
    discrete_states = np.zeros((action_space, no_discrete_states, no_discrete_states), dtype='uint32')
    step1 = 1.8 / (no_intervals - 1)
    step2 = 0.14 / (no_intervals - 1)
    for i, x_bound in enumerate(np.arange(-1.2, 0.6, step1)):
        x = -1.2 if x_bound == -1.2 else random.uniform(x_bound - step1, x_bound)
        for j, v_bound in enumerate(np.arange(-0.07, 0.07, step2)):
            v = -0.07 if v_bound == -0.07 else random.uniform(v_bound - step2, v_bound)
            for action in range(action_space):
                initial_state = [x, v]
                run_policy(env, initial_state, episodes, no_intervals, action)
    

    transitions = []
    for a in range(action_space):
        transitions_a = []
        for i in range(len(discrete_states[a])):
            arr = discrete_states[a][i]
            transitions_a.extend([np.where(arr>0)[0].tolist()])
        transitions.extend([transitions_a])

    np.save(os.path.join('mdps', 'mountain_car', 'mdp_' + str(no_intervals) + '_' + str(episodes) + '.npy'), transitions)
