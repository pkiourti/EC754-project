import gym
import numpy as np
import time
from time import sleep
from tqdm import tqdm
import random
import argparse
import os
from cartpole import CartPoleEnv
import json


def parser():
    parser = argparse.ArgumentParser('create a transition model from mdp')
    parser.add_argument('--no_intervals', type=int, required=True)
    parser.add_argument('--episodes', type=int, required=True)
    return parser


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


def map_angle(x, intervals):
    step = 2.3 / (no_intervals - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x <= r:
            return i
    return no_intervals - 1


def map_angle_velocity(x, intervals):
    step = 2.3 / (no_intervals - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x <= r:
            return i
    return no_intervals - 1


def map_state_to_a_number(state, no_intervals):
    x, v, a, v_a = state
    i = map_position(x, no_intervals)
    j = map_velocity(v, no_intervals)
    k = map_angle(a, no_intervals)
    l = map_angle_velocity(v_a, no_intervals)
    return i*no_intervals*no_intervals*no_intervals + \
           j*no_intervals*no_intervals + \
           k*no_intervals + l


def map_number_to_state(number, no_intervals):
    l = number % no_intervals
    k = (number // no_intervals) % no_intervals
    j = (number // (no_intervals * no_intervals)) % no_intervals
    i = (number // (no_intervals * no_intervals * no_intervals)) % no_intervals
    return i, j, k, l


def map_continuous_state_to_discrete(number, no_intervals):
    step = 2.3 / (no_intervals - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if number == i:
            return (r - step + r) / 2


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
    
    ENV_NAME = "CartPole-v1"
    #env = gym.make(ENV_NAME)
    env = CartPoleEnv()
    action_space = env.action_space.n
    #env.seed(np.random.randint(0, 10000))
    
    no_discrete_states = no_intervals * no_intervals * no_intervals * no_intervals
    discrete_states = np.zeros((action_space, no_discrete_states, no_discrete_states), dtype='uint32')
    
    step = 2.3 / (no_intervals - 1)
    for x_bound in np.arange(-1.1, 1.2 + step, step):
        x = random.uniform(x_bound - step, x_bound)
        for v_bound in np.arange(-1.1, 1.2 + step, step):
            v = random.uniform(v_bound - step, v_bound)
            for a_bound in np.arange(-1.1, 1.2 + step, step):
                a = random.uniform(a_bound - step, a_bound)
                for v_a_bound in np.arange(-1.1, 1.2 + step, step):
                    for action in range(action_space):
                        v_a = random.uniform(v_a_bound - step, v_a_bound)
                        initial_state = [x, v, a, v_a]
                        run_policy(env, initial_state, episodes, no_intervals, action)
    

    transitions0 = []
    transitions = []
    for a in range(action_space):
        transitions_a = []
        for i in range(len(discrete_states[a])):
            arr = discrete_states[a][i]
            #if len(arr[arr > 0]):
            #print(np.where(arr > 0)[0])
            transitions_a.extend([np.where(arr>0)[0].tolist()])
        transitions.extend([transitions_a])
        #print(len(transitions_a))
        #print(len(transitions))
        #print(len(transitions))
    #print(transitions)
    #filename = os.path.join('mdps_correct', 'cartpole', 'mdp_' + str(no_intervals) + '_' + str(episodes) + '.json')
    #f = open(filename, 'w')
    #json.dump(transitions, f)
    #f.close()

    np.save(os.path.join('mdps_correct', 'cartpole', 'mdp_' + str(no_intervals) + '_' + str(episodes) + '.npy'), transitions)

    #for a in range(action_space):
    #    det_transitions[a] = np.argmax(discrete_states[a], axis=1)
    #np.save(os.path.join('mdps_correct', 'cartpole', 'mdp_' + str(no_intervals) + '_' + str(episodes) + '.npy'), discrete_states)
    #for a in range(action_space):
    #    for i, transition in enumerate(det_transitions[a]):
    #        if i == transition:
    #            v = np.argsort(discrete_states[a][i])[-2]
    #            print('i, transition, v', i, transition, v)
    #            det_transitions[a][i] = v
                #x, v, a, v_a = map_number_to_state(transition, no_intervals)
                #x_int = map_continuous_state_to_discrete(x, no_intervals)
                #v_int = map_continuous_state_to_discrete(v, no_intervals)
                #a_int = map_continuous_state_to_discrete(a, no_intervals)
                #v_a_int = map_continuous_state_to_discrete(v_a, no_intervals)
                #if v_int < 0 and v_a_int < 0:
                #    x = x + 1
                #elif v_int > 0 and v_a_int > 0:
                #    x = x - 1
                #elif v_int < 0 and v_a_int < 0:
                #    x = x - 1
                #elif v_int < 0 and v_a_int < 0:
                #    x = x + 1
                #map_state_to_a_number([x, v, a, v_a], no_intervals)

    #np.save(os.path.join('mdps_correct', 'cartpole', 'mdp_' + str(no_intervals) + '_' + str(episodes) + '.npy'), det_transitions)
   #for i, x_bound in enumerate(np.arange(-1.1, 1.2 + step, step)):
   #    x = random.uniform(-2.4, x_bound) if x_bound == -1.1 else random.uniform(x_bound - step, x_bound)
   #    for j, v_bound in enumerate(np.arange(-1.1, 1.2 + step, step)):
   #        v = random.uniform(-2.4, v_bound) if v_bound == -1.1 else random.uniform(v_bound - step, v_bound)
   #        for k, a_bound in enumerate(np.arange(-1.1, 1.2 + step, step)):
   #            a = random.uniform(-2.4, a_bound) if a_bound == -1.1 else random.uniform(a_bound - step, a_bound)
   #            for l, v_a_bound in enumerate(np.arange(-1.1, 1.2 + step, step)):
   #                for action in range(action_space):
   #                    v_a = random.uniform(-2.4, v_a_bound) if v_a_bound == -1.1 else random.uniform(v_a_bound - step, v_a_bound)
   #                    initial_state = [x, v, a, v_a]
   #                    run_policy(env, initial_state, episodes, no_intervals, action)
    
