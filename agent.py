# This code is based on https://github.com/palladiun/SuccessorRepresentation.

import numpy as np
import utils

class SFAgent():
    def __init__(self, featvec_size, action_size, \
                alpha_r = 0.1, alpha_w = 0.1, gamma = 0.95, weight_init = "eye"):
        self.featvec_size = featvec_size
        self.action_size = action_size
        self.sf_size = featvec_size
        self.r_vector = np.zeros(self.featvec_size) # expected position of reward
        self.alpha_r = alpha_r # learning rate for reward vector W
        self.alpha_w = alpha_w # learning rate for M matrix
        self.gamma = gamma # discount rate
        if weight_init == "eye":
            self.w_matrix = np.eye(self.featvec_size)
        elif weight_init == "zero":
            self.w_matrix = np.zeros((self.featvec_size, self.sf_size))
        elif weight_init == "random":
            self.w_matrix = utils.weight_init(self.featvec_size, self.sf_size)
        elif weight_init == "He":
            self.w_matrix = utils.he_init(self.featvec_size, self.sf_size)
        elif weight_init == "uni":
            self.w_matrix = utils.uniform(self.featvec_size, self.sf_size)
        else:
            print("weight initialization problem.")


    def estimated_sf_vec(self, featvec):
        return self.w_matrix @ featvec 
    
    @property
    def estimated_SR(self):
        feature_matrix = np.eye(self.featvec_size)
        return np.matmul(self.w_matrix, feature_matrix).T
    
    def update_w(self, current_exp):
        s_t = current_exp[0]
        s_t_1 = current_exp[2]
        sf_s_t = self.estimated_sf_vec(s_t)
        sf_s_t_1 = self.estimated_sf_vec(s_t_1)
        done = current_exp[4]
        if done:
            delta_in = self.alpha_w * (s_t + self.gamma*s_t_1 - sf_s_t) 
        else:
            delta_in = self.alpha_w * (s_t + self.gamma*sf_s_t_1 - sf_s_t)
        delta_W = np.outer(delta_in, s_t)
        self.w_matrix += delta_W
        return delta_W

    def update_r_vector(self, current_exp):
        s_t_1 = current_exp[2]
        reward = current_exp[3]
        delta_in = self.alpha_r * (reward - np.matmul(self.r_vector, s_t_1))
        delta_r_vector = delta_in * s_t_1
        self.r_vector += delta_r_vector
        return delta_r_vector


    def V_estimates(self, featvec, goal = None):
        goal = self.r_vector
        sf_vec = self.estimated_sf_vec(featvec)
        V_state = np.matmul(sf_vec, goal)
        return V_state
    
    @property
    def V_vector_estimated(self):
        return np.matmul(self.estimated_SR, self.r_vector)

    def Q_estimates(self, next_state, reward):
        '''
        estimate Q value depends on estimated value of next state
        '''
        V = self.V_estimates(next_state)
        Qvalue = reward + self.gamma * V
        return Qvalue

class SRAgent():
    def __init__(self, state_size, action_size, alpha_r, alpha_m, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.r_vector = np.zeros([state_size]) # expected position of reward
        self.alpha_r = alpha_r # learning rate for reward vector W
        self.alpha_m = alpha_m # learning rate for M matrix
        self.gamma = gamma # discount rate
        self.sr_matrix = np.eye(state_size)


    def V_estimates(self, next_state, goal = None):
        goal = self.r_vector
        V_next_state = np.matmul(self.sr_matrix[next_state, :], goal) 
        return V_next_state
    
    @property
    def V_vector_estimated(self):
        return np.matmul(self.sr_matrix, self.r_vector)

    def Q_estimates(self, next_state, reward):
        '''
        estimate Q value depends on estimated value of next state
        '''
        V = self.V_estimates(next_state)
        Qvalue = reward + self.gamma * V
        return Qvalue

    def update_r_vector(self, current_exp):
        '''
        reward positon vector updated by TD error rule.
        current_exp: [state, action, state_next, reward, done]
                     in experience list [-1] element.
        '''
        state_next = current_exp[2] 
        reward = current_exp[3] # here the agent receive information for reward
        error = reward - self.r_vector[state_next] # update the belief of agent for reward
        self.r_vector[state_next] += self.alpha_r * error # delta(prediction error) update
        return error

    def update_sr(self, current_exp):
        '''
        SARSA TD learning rule, in branching git sarsa
        for another TD rule such as Q learning, make another branch
        current_exp : [state, action, state_next, reward, done]
                      in experience list [-1] element.
        '''
        state = current_exp[0]
        state_next = current_exp[2]
        done = current_exp[4]
        I = utils.onehot(state, self.state_size)
        if done:
            td_error = (I + self.gamma * utils.onehot(state_next, self.state_size) - \
                self.sr_matrix[state, :])
        else:
            td_error = (I + self.gamma * self.sr_matrix[state_next, :] \
                - self.sr_matrix[state, :])
        self.sr_matrix[state, :] += self.alpha_m * td_error
        return td_error


class Qagent():
    def __init__(self, state_size, action_size, alpha, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.v_learning = np.zeros([state_size])
        self.alpha = alpha # learning rate for q learner
        self.gamma = gamma # discount rate
        

    def update_V(self, current_exp):
        state = current_exp[0]
        state_next = current_exp[2]
        reward = current_exp[3]
        done = current_exp[4]
        if done:
            td_error = (self.gamma*reward - self.v_learning[state])
        else:
            td_error = (reward + self.gamma * self.v_learning[state_next] - \
                self.v_learning[state])
        self.v_learning[state] += self.alpha * td_error
        return td_error

    @property
    def V_vector_estimated(self):
        return self.v_learning 

    def Q_estimates(self, next_state, reward):
        '''
        estimate Q value depends on estimated value of next state
        '''
        V = self.v_learning[next_state]
        Qvalue = reward + self.gamma * V
        return Qvalue