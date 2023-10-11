from math import sqrt
import numpy as np
import numpy.random as random

# This code is based on https://github.com/palladiun/SuccessorRepresentation
def onehot(value, length_of_vec):
    '''
    making one hot vector, two arg needs
    value : position of a hot element
    length_of_vec : length of one hot vector
    '''
    vec = np.zeros(length_of_vec)
    vec[value] = 1
    return vec

def rel_action(sr_matrix):
    '''
    return relative value of action in each state
    sr_matrix : M_Q sr matrix
    '''
    state_size = sr_matrix.shape[1]
    unit_vector = np.ones([state_size])
    action_values = np.matmul(sr_matrix, unit_vector)
    
    action_values_sum = np.sum(action_values, axis = 0)
    action_values_each_state = action_values/action_values_sum
    return action_values_each_state

# Xavier Weight Initilazation 
# This code is based on https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
def weight_init(n_input, n_output):
    lower = -(1.0 /sqrt(n_input))
    upper = 1.0 / sqrt(n_input)
    numbers = random.rand(n_input * n_output)
    scaled = lower + numbers*(upper - lower)
    w_matrix = np.array(scaled).reshape([n_input, n_output])
    return np.abs(w_matrix) # Uniform distribution [0, 1/sqrt(n_i)]

def he_init(n_input, n_output):
    std = sqrt(2.0/n_input)
    numbers = random.rand(n_input * n_output)
    scaled = numbers * std
    w_matrix = np.array(scaled).reshape([n_input, n_output])
    return np.abs(w_matrix)

def uniform(n_input, n_output):
    std = 0.1
    numbers = random.rand(n_input * n_output)
    scaled = numbers * std
    w_matrix = np.array(scaled).reshape([n_input, n_output])
    return np.abs(w_matrix)

def V_error_calculation(V_ground_truth, V_estimates):
    V_true = np.array(V_ground_truth)
    V_est = np.array(V_estimates)
    errors = (V_true - V_est) ** 2
    return np.mean(errors[:-1]) # exclude error from last state from 1D maze

def mat_error(sr_mat_history, sf_mat_history):
    total_trials = len(sr_mat_history)
    sf_error_history = []
    for trial in range(total_trials):
        each_sr = sr_mat_history[trial]
        each_sf = sf_mat_history[trial]
        total_epi = len(each_sr)
        sf_error_of_episodes = []
        for epi in range(total_epi):
            sr_mat = each_sr[epi][:-1, :]
            sf_mat = each_sf[epi][:-1, :]
            sf_errors = np.mean((sr_mat - sf_mat) ** 2)
            sf_error_of_episodes.append(sf_errors)
        sf_error_history.append(sf_error_of_episodes)
    return sf_error_history

def my_argmax(Qarray):
    max_index = np.where(Qarray == Qarray.max())[0]
    if len(max_index) == 1:
        return np.argmax(Qarray)
    else:
        return np.random.randint(len(Qarray))

def dV_depi(v_error):
    v_error_ahead = v_error[1:]
    v_error_ahead = np.append(v_error_ahead, v_error_ahead[-1])
    dV_de = -(v_error_ahead - v_error)
    return dV_de

def dstep_depi(step_length):
    step_length_ahead = step_length[1:]
    step_length_ahead = np.append(step_length_ahead, step_length_ahead[-1])
    ds_de = step_length_ahead - step_length
    return ds_de
