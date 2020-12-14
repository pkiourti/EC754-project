import os
import numpy as np
from keras.models import model_from_json
import argparse
import json
import itertools


def parser():
    parser = argparse.ArgumentParser('automatically encode nn into smt')
    parser.add_argument('--x0', type=float, required=True)
    parser.add_argument('--x1', type=float, required=True)
    parser.add_argument('--x2', type=float, required=True)
    parser.add_argument('--x3', type=float, required=True)
    parser.add_argument('--model_id', type=int, required=True)
    parser.add_argument('--timesteps', type=int, required=True)
    parser.add_argument('--mdp', type=str, required=True)
    parser.add_argument('--no_intervals', type=int, required=True)
    return parser


def get_output(model, initial_state):
    q_values = model.predict(initial_state)
    #print('output', q_values.shape)
    return np.argmax(q_values, axis=1)


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


def get_initial_state(no, x0, x1, x2, x3):
    #x = -0.03246117
    #v = 0.00904449
    #a = -0.02069081
    #v_a = -0.00731908
    print(x0, x1, x2, x3)
    x_d = map_position(x0, no)
    v_d = map_velocity(x1, no)
    a_d = map_velocity(x2, no)
    v_a_d = map_angle_velocity(x3, no)
    return [x_d, v_d, a_d, v_a_d]


def constraints_x_high(no):
    x1 = 0.4
    v1 = 0.5
    return [map_position(x1, no), map_velocity(v1, no)]


def constraints_x_low(no):
    x2 = -0.4
    v2 = -0.5
    return [map_position(x2, no), map_velocity(v2, no)]


def constraints_a_high(no):
    a1 = 0.0523599
    v_a1 = 0.4
    return [map_angle(a1, no), map_angle_velocity(v_a1, no)]


def constraints_a_low(no):
    a2 = -0.0523599
    v_a2 = -0.4
    return [map_angle(a2, no), map_angle_velocity(v_a2, no)]


def map_to_discrete(x, no_intervals):
    step = 2.3 / (no_intervals - 1)
    for i, r in enumerate(np.arange(-1.1, 1.2, step)):
        if x <= r:
            return i
    return no_intervals - 1


def declare_state_variables(f, TIMESTEPS, num_states):
    #transitions = state_to_states_transition[t]
    #print('transitions', transitions)
    print('Declaring state and action variables...')
    for t_idx, t in enumerate(range(TIMESTEPS)):
        for n in range(num_states[t]):
            #print('n', n)
            f.write('(declare-const x0_' + str(t_idx) +'_' + str(n) + ' Int)\n')
            f.write('(declare-const x1_' + str(t_idx) +'_' + str(n) + ' Int)\n')
            f.write('(declare-const x2_' + str(t_idx) +'_' + str(n) + ' Int)\n')
            f.write('(declare-const x3_' + str(t_idx) +'_' + str(n) + ' Int)\n')
            f.write('(declare-const a_' + str(t_idx) +'_'+str(n) + '  Int)\n')


def declare_neural_network_variables(f, model, TIMESTEPS, num_states):
    print('Declaring neural network variables...')
    w = []
    b = []
    for idx, layer in enumerate(model.layers):
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]

        # define variables
        input_shape, output_shape = w.shape
        for i in range(input_shape):
            for j in range(output_shape):
                f.write('(declare-const l' + str(idx) + '_w_x' + str(i) + '_' + str(j) + ' Real)\n')

        output_shape = b.shape[0]
        for j in range(output_shape):
            f.write('(declare-const l' + str(idx) + '_b_x_' + str(j) + ' Real)\n')

        _, output_shape = w.shape
        for t_idx, t in enumerate(range(TIMESTEPS)):
            all_states = num_states[t]
            for n in range(all_states):
                for j in range(output_shape):
                    f.write('(declare-const l' + str(idx) + '_out_x' + str(j) + '_' + str(t_idx) + '_' + str(n) + ' Real)\n')
                    f.write('(declare-const l' + str(idx) + '_act_out_x' + str(j) + '_' + str(t_idx)  + '_' + str(n) + ' Real)\n')

        # declare discrete states
        if idx == 0:
            for t_idx, t in enumerate(range(TIMESTEPS)):
                for n in range(num_states[t]):
                    #print('n', n)
                    #print('declare', t_idx, n)
                #for i in range(input_shape):
                #    smt_file.write('(declare-const discrete_inter_x' + str(i) + '_' + str(t) + ' Int)\n')
                    f.write('(declare-const discrete_x_' + str(t_idx) + '_' + str(n) + ' Int)\n')
            for t_idx, t in enumerate(range(TIMESTEPS - 1)):
                for n in range(num_states[t_idx+1]):
                    f.write('(declare-const next_discrete_x_' + str(t_idx) + '_' + str(n) + ' Int)\n')


def feed_forward(f, model, t, num_states):
    print('time:', t, 'creating assert statements for feed forward')
    for idx, layer in enumerate(model.layers):
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]
        input_shape, output_shape = w.shape
        all_states = num_states[t]
        for n in range(all_states):
            for j in range(output_shape):
                var = 'l' + str(idx) + '_out_x' + str(j) + '_' + str(t) + '_' + str(n)
                f.write('(assert (= ' + var + ' ')
                operands = []
                for i in range(input_shape):
                    right_var1 = 'x' + str(i) + '_' + str(t) + '_' + str(n) if idx == 0 else 'l' + str(idx - 1) + '_act_out_x' + str(i) + '_' + str(t) + '_' + str(n)
                    right_var2 = 'l' + str(idx) + '_w_x' + str(i) + '_' + str(j) 
                    operands.extend(['(+ (* ' + right_var1 + ' ' + right_var2 + '))'])
                f.write('(+ ') 
                for i in range(len(operands) - 1):
                    f.write('(+ ' + operands[i] + ' ')
                par = ''.join([')' for _ in range(len(operands) - 1)])
                f.write(operands[i+1] + par)
                right_var3 = 'l' + str(idx) + '_b_x_' + str(j) 
                f.write(' ' + right_var3 + ')))\n')
            if idx < len(model.layers) - 1:
                for j in range(output_shape):
                    var = 'l' + str(idx) + '_act_out_x' + str(j) + '_' + str(t) + '_' + str(n)
                    right_var = 'l' + str(idx) + '_out_x' + str(j) + '_' + str(t) + '_' + str(n)
                    f.write('(assert (= ' + var + ' (ite (> ' + right_var + ' 0) ' + right_var + ' 0)))\n')
            elif idx == len(model.layers) - 1:
                for j in range(output_shape):
                    var = 'l' + str(idx) + '_act_out_x' + str(j) + '_' + str(t) + '_' + str(n)
                    right_var = 'l' + str(idx) + '_out_x' + str(j) + '_' + str(t) + '_' + str(n)
                    f.write('(assert (= ' + var + ' ' + right_var + '))\n')
            action0 = 'l'+str(len(model.layers) - 1) + '_act_out_x0_' + str(t) + '_' + str(n)
            action1 = 'l'+str(len(model.layers) - 1) + '_act_out_x1_' + str(t) + '_' + str(n)
            f.write('(assert (= a_' + str(t) + '_' + str(n) + ' (ite (> ' + action0 + ' ' + action1 + ') 0 1)))\n')


def set_weights(f, model):
    print('Setting neural network weights...')
    for idx, layer in enumerate(model.layers):
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]

        # set weights
        input_shape, output_shape = w.shape
        for i in range(input_shape):
            for j in range(output_shape):
                # set value w[i, j] of layer idx to l{idx}_w_x{i}_{j}
                var = 'l' + str(idx) + '_w_x' + str(i) + '_' + str(j)
                f.write('(assert (= ' + var + ' ' + '%f' % w[i, j] + '))\n')

        # set biases
        output_shape = b.shape[0]
        for j in range(output_shape):
            f.write('(assert (= l' + str(idx) + '_b_x_' + str(j) + ' ' + str(b[j]) + '))\n')


def set_initial_state(f, initial_state):
    #print('Initial state', initial_state)
    print('Setting initial state...')
    f.write('(assert (= x0_0_0 ' + str(initial_state[0]) + '))\n')
    f.write('(assert (= x1_0_0 ' + str(initial_state[1]) + '))\n')
    f.write('(assert (= x2_0_0 ' + str(initial_state[2]) + '))\n')
    f.write('(assert (= x3_0_0 ' + str(initial_state[3]) + '))\n')


# TODO
def map_discrete_state_to_one_num(f, no_intervals, t, num_states):
    all_states = num_states[t]
    for n in range(all_states):
        operands = []
        operands.extend(['(* ' + str(no_intervals) + ' (* ' + str(no_intervals) + ' (* ' + str(no_intervals) + ' x0_' + str(t) + '_' + str(n) + ')))'])
        operands.extend(['(* ' + str(no_intervals) + ' (* ' + str(no_intervals) + ' x1_' + str(t) + '_' + str(n) + '))'])
        operands.extend(['(* ' + str(no_intervals) + ' x2_' + str(t) + '_' + str(n) + ')'])
        operands.extend(['x3_' + str(t) + '_' + str(n)])
        f.write('(assert (= discrete_x_' + str(t) + '_' + str(n) + ' ')
        for i in range(len(operands) - 1):
            f.write('(+ ' + operands[i] + ' ')
        par = ''.join([')' for _ in range(len(operands) - 1)])
        f.write(operands[i+1] + par + '))\n')
        
        f.write('')


def find_next_state_from_array(f, mdp, t, num_states, state_to_states_transition):
    print('time:', t, 'creating assert statements for transitioning to next states')
    transitions = state_to_states_transition[t]
    #print('transitions', transitions)
    next_states = 0
    for n in range(num_states[t]):
        #print('n', n)
        for idx_next in range(transitions[n]):
            f.write('(assert (= next_discrete_x_' + str(t) + '_' + str(next_states + idx_next))
            f.write(' (ite (= a_' + str(t) + '_' + str(n) + ' 0) ')
            for a in range(2):
                conditions = []
                ranges = []
                for i, r in enumerate(mdp[a]):
                    if len(r) != transitions[n]:
                        continue
                    #if r == 0:
                    #    continue
                    conditions.extend(['(= discrete_x_' + str(t) + '_' + str(n) + ' ' + str(i) + ')'])
                    ranges.extend([r[idx_next]])
                #conditions = conditions[0:5]
                #ranges = ranges[0:5]
                for i, _ in enumerate(conditions):
                    f.write('(ite ' + conditions[i] + ' ' + str(ranges[i]) + ' ')
                f.write('discrete_x_' + str(t) + '_' + str(n))
                f.write(''.join([')' for _ in range(len(conditions))]))
                f.write(' ')

            f.write(')))\n')
        next_states += transitions[n]


def set_next_state(f, no_intervals, t, num_states):
    print('time:', t, 'creating assert statements for setting next states')
    next_t = t + 1
    all_states = num_states[next_t]
    for n in range(all_states):
        var0 = 'x0_' + str(next_t) + '_' + str(n)
        var1 = 'x1_' + str(next_t) + '_' + str(n)
        var2 = 'x2_' + str(next_t) + '_' + str(n)
        var3 = 'x3_' + str(next_t) + '_' + str(n)
        f.write('(assert (= ' + var3 + ' (mod next_discrete_x_' + str(t) + '_' + str(n) + ' ' + str(no_intervals) + ')))\n')
        f.write('(assert (= ' + var2 + ' (mod (/ (- next_discrete_x_' + str(t) + '_' + str(n) + ' ' + var3 + ') ' + str(no_intervals) + ' ) ' + str(no_intervals) + ')))\n')
        f.write('(assert (= ' + var1 + ' (mod (/ (/ (- (- next_discrete_x_' + str(t) + '_' + str(n) + ' ' + var3 + ') ' + var2 + ') ' + str(no_intervals) + ') ' + str(no_intervals) + ') ' + str(no_intervals) + ')))\n')
        f.write('(assert (= ' + var0 + ' (mod (/ (/ (/ (- (- (- next_discrete_x_' + str(t) + '_' + str(n) + ' ' + var3 + ') ' + var2 + ') ' + var1 + ') ' + str(no_intervals) + ') ' + str(no_intervals) + ') ' + str(no_intervals) + ') ' + str(no_intervals) + ')))\n')


def map_to_one_number(states, r):
    arr = []
    for s in states:
        x, v, a, v_a = s[0], s[1], s[2], s[3]
        arr.extend([x*r*r*r + v*r*r + a*r + v_a])
    return arr


def map_to_discrete_numbers(states, r):
    arr = []
    for s in states:
        v_a = s % r
        a = (s // r) % r
        v = (s // (r * r)) % r
        x = (s // (r * r * r)) % r
        arr.extend([[x, v, a, v_a]])
    return arr


def find_states_at_each_timestep(model, mdp, timesteps, initial_states, r):
    all_states = []
    states = initial_states
    all_states = [[states]]
    state_to_states_transition = []
    for i in range(timesteps-1):
        states = np.reshape(np.asarray(states), (-1, 4))
        actions = get_output(model, states)
        discrete_states = map_to_one_number(states, r)
        #print('Iteration', i, 'states', len(states), 'discrete_states', len(discrete_states), 'actions', len(actions))
        next_states = [mdp[actions[idx]][s] for idx, s in enumerate(discrete_states)]
        state_to_states_transition.append([len(states) for states in next_states])
        next_states = list(itertools.chain(*next_states))
        states = map_to_discrete_numbers(next_states, r)
        all_states.extend([states])

    num_states = []
    for i, s in enumerate(all_states):
        num_states.extend([len(s)]) 
    np.save('num_states.npy', num_states)
    np.save('state_to_states_transition.npy', state_to_states_transition)
    return num_states, state_to_states_transition
    


if __name__ == '__main__':
    args = parser().parse_args()
    env = 'cartpole' 
    initial_x0 = args.x0
    initial_x1 = args.x1
    initial_x2 = args.x2
    initial_x3 = args.x3
    model_id = args.model_id
    no_intervals = args.no_intervals
    step = 2.3 / (no_intervals - 1)
    TIMESTEPS = args.timesteps
    folder = os.path.join(env+'-0', 'models')
    arch = os.path.join(folder, 'model_' + str(model_id) +  '_' + str(no_intervals) + '.json')
    weights = os.path.join(folder, 'model_' + str(model_id) + '_' + str(no_intervals) + '.h5')

    with open(arch, 'r') as f:
        arch_json = json.load(f)

    model = model_from_json(arch_json)
    model.load_weights(weights)
    mdp = np.load(os.path.join('mdps', 'cartpole', args.mdp), allow_pickle=True)
    mdp = mdp.tolist()

    debug_folder = os.path.join('formulas', 'x0_'+str(initial_x0) +  
                                            'x1_'+str(initial_x1) + 
                                            'x2_'+str(initial_x2) + 
                                            'x3_'+str(initial_x3))
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    smt_file = open(os.path.join(debug_folder, 'cartpole_' + str(model_id) + '_' + str(TIMESTEPS) + '_' + args.mdp + '.smt2'), 'w')
    smt_file.write('(set-logic QF_NIRA)\n')

    initial_state = get_initial_state(no_intervals, initial_x0, initial_x1, initial_x2, initial_x3)
    num_states, state_to_states_transition = find_states_at_each_timestep(model, mdp, TIMESTEPS, initial_state, no_intervals)
    declare_state_variables(smt_file, TIMESTEPS, num_states)
    declare_neural_network_variables(smt_file, model, TIMESTEPS, num_states)

    set_initial_state(smt_file, initial_state)

    set_weights(smt_file, model)

    for t_idx, t in enumerate(range(TIMESTEPS-1)):
        feed_forward(smt_file, model, t, num_states)

        # map discrete state in one number
        map_discrete_state_to_one_num(smt_file, no_intervals, t, num_states)
        
        # next discrete comes from mdp
        find_next_state_from_array(smt_file, mdp, t, num_states, state_to_states_transition)
        set_next_state(smt_file, no_intervals, t, num_states)

    x_high = constraints_x_high(no_intervals)
    x_low = constraints_x_low(no_intervals)
    a_high = constraints_a_high(no_intervals)
    a_low = constraints_a_low(no_intervals)
    t = TIMESTEPS - 1
    print('time:', t, 'creating assert statements for checking the last states')
    for n in range(num_states[-1]):
        smt_file.write('(assert (or')
        smt_file.write(' (and (>= x0_' + str(t) + '_' + str(n) + ' ' + str(x_high[0]) + ') (>= x1_' + str(t) + '_' + str(n) + ' ' + str(x_high[1]) + '))')
        smt_file.write(' (and (<= x0_' + str(t) + '_' + str(n) + ' ' + str(x_low[0]) + ') (<= x1_' + str(t) + '_' + str(n) + ' ' + str(x_low[1]) + '))')
        smt_file.write(' (and (>= x2_' + str(t) + '_' + str(n) + ' ' + str(a_high[0]) + ') (>= x3_' + str(t) + '_' + str(n) + ' ' + str(a_high[1]) + '))')
        smt_file.write(' (and (<= x2_' + str(t) + '_' + str(n) + ' ' + str(a_low[0]) + ') (<= x3_' + str(t) + '_' + str(n) + ' ' + str(a_low[1]) + '))')
        smt_file.write('))\n')

    smt_file.write('(check-sat)\n')
    smt_file.write('(get-model)\n')
    smt_file.close()
