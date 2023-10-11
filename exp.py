# load library
import copy
import numpy as np
from tqdm import tqdm

# custom library
from env import Simple1DMaze
from agent import Qagent, SRAgent, SFAgent
import utils


def QL_1D(episodes = 500,
          alpha_q = 0.1,
          gamma = 0.95,
          corridor_size = 5, 
          explora = False, 
          epsilon_dic = None):
    experiences = []
    step_lengths = []
    max_step_length = corridor_size * 50
    
    V_vector_estimated_history = []
    V_error_history = []
    V_ground_truth = [gamma ** (corridor_size - (i+1)) for i in range(corridor_size)]
    
    maze = Simple1DMaze(corridor_size, obs_mode="index")
    agent = Qagent(maze.corridor_size, maze.action_size, alpha_q, gamma)
    for episode in tqdm(range(episodes), desc="episodes"):
        agent_start = [0]
        goal_pos = [maze.corridor_size - 1]
        
        maze.reset(agent_pos=agent_start, goal_pos=goal_pos)
        state = maze.observation
       
        if explora:
            #epsilon decay
            epsilon = 0.9 * (epsilon_dic ** episode) + 0.1
        else:
            epsilon = epsilon_dic

        #step_idx = 0 for while loop
        
        for step_idx in range(max_step_length):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(maze.action_size)
            else:
                Qvalues = np.zeros(maze.action_size)
                for action in maze.action_set:
                    simulated_next_state, simulated_reward = maze.simulate(action)
                    Qvalues[action] = agent.Q_estimates(simulated_next_state, \
                                                        simulated_reward)
                action = utils.my_argmax(Qvalues)
            #action = 1
            reward = maze.step(action)
            state_next = maze.observation
            done = maze.done
            experiences.append([state, action, state_next, reward, done])
            state = state_next
            agent.update_V(experiences[-1])
            if maze.done:
                break
            #step_idx += 1
        
        step_lengths.append(step_idx+1)
        
        V_vector_estimated_history.append(agent.V_vector_estimated)
        V_error_history.append(utils.V_error_calculation(V_ground_truth,
                                             agent.V_vector_estimated))


    return step_lengths, V_vector_estimated_history, V_error_history


def SR_1D(episodes = 500,
          alpha_m = 0.1,
          alpha_r = 0.1,
          gamma = 0.95,
          corridor_size = 5, 
          explora = False, 
          epsilon_dic = None):
    experiences = []
    step_lengths = []
    lifetime_R_errors = []
    sr_mat_history = []
    V_vector_estimated_history = []
    V_error_history = []

    max_step_length = corridor_size * 50
    V_ground_truth = [gamma ** (corridor_size - (i+1)) for i in range(corridor_size)]

    
    maze = Simple1DMaze(corridor_size, obs_mode="index")
    agent = SRAgent(maze.corridor_size, maze.action_size, alpha_r, alpha_m, gamma)
    for episode in tqdm(range(episodes), desc="episodes"):
        agent_start = [0]
        goal_pos = [maze.corridor_size - 1]
        
        maze.reset(agent_pos=agent_start, goal_pos=goal_pos)
        state = maze.observation
        reward_error = []

        if explora:
            #epsilon decay
            epsilon = 0.9 * (epsilon_dic ** episode) + 0.1
        else:
            epsilon = epsilon_dic

        # step_idx = 0  for while loop
        # while True:
        for step_idx in range(max_step_length):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(maze.action_size)
            else:
                Qvalues = np.zeros(maze.action_size)
                for action in maze.action_set:
                    simulated_next_state, simulated_reward = maze.simulate(action)
                    Qvalues[action] = agent.Q_estimates(simulated_next_state, \
                                                        simulated_reward)
                action = utils.my_argmax(Qvalues)
            #action = 1
            reward = maze.step(action)
            state_next = maze.observation
            done = maze.done
            experiences.append([state, action, state_next, reward, done])
            state = state_next
            if step_idx >= 0:
                agent.update_sr(experiences[-1])
                delta_r_vector = agent.update_r_vector(experiences[-1])
                reward_error.append(np.mean(np.abs(delta_r_vector)))
            if maze.done:
                agent.update_sr(experiences[-1])
                reward_error.append(np.mean(np.abs(delta_r_vector)))
                break
            #step_idx += 1
        
        step_lengths.append(step_idx+1)
        lifetime_R_errors.append(np.mean(reward_error))
        V_vector_estimated_history.append(agent.V_vector_estimated)
        V_error_history.append(utils.V_error_calculation(V_ground_truth,
                                             agent.V_vector_estimated))
        sr_mat_history.append(copy.deepcopy(agent.sr_matrix))


    return step_lengths, sr_mat_history, V_vector_estimated_history, \
        V_error_history



def SF_1D(episodes = 500,
          alpha_w = 0.1,
          alpha_r = 0.1,
          gamma = 0.95,
          corridor_size = 5, 
          weight_init = "eye",
          explora = False, 
          epsilon_dic = None):
    experiences = []
    step_lengths = []
    lifetime_R_errors = []
    sf_mat_history = []
    V_vector_estimated_history = []
    V_error_history = []

    max_step_length = corridor_size * 50
    V_ground_truth = [gamma ** (corridor_size - (i+1)) for i in range(corridor_size)]

    maze = Simple1DMaze(corridor_size, obs_mode="onehot")
    agent = SFAgent(maze.corridor_size, maze.action_size, alpha_r, alpha_w, gamma, weight_init=weight_init)
    for episode in tqdm(range(episodes), desc="episodes"):
        agent_start = [0]
        goal_pos = [maze.corridor_size - 1]
        
        maze.reset(agent_pos=agent_start, goal_pos=goal_pos)
        state = maze.observation
        reward_error = []

        if explora:
            #epsilon decay
            epsilon = 0.9 * (epsilon_dic ** episode) + 0.1
        else:
            epsilon = epsilon_dic
        
        # step_idx = 0 for while loop
        #while True:
        for step_idx in range(max_step_length):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(maze.action_size)
            else:
                Qvalues = np.zeros(maze.action_size)
                for action in maze.action_set:
                    simulated_next_state, simulated_reward = maze.simulate(action)
                    Qvalues[action] = agent.Q_estimates(simulated_next_state, \
                                                        simulated_reward)
                action = utils.my_argmax(Qvalues)
            #action = 1
            reward = maze.step(action)
            state_next = maze.observation
            done = maze.done
            experiences.append([state, action, state_next, reward, done])
            state = state_next
            if step_idx >= 0:
                agent.update_w(experiences[-1])
                delta_r_vector = agent.update_r_vector(experiences[-1])
                reward_error.append(np.mean(np.abs(delta_r_vector)))
            if maze.done:
                agent.update_w(experiences[-1])
                reward_error.append(np.mean(np.abs(delta_r_vector)))
                break
            #step_idx += 1
        
        step_lengths.append(step_idx+1)
        lifetime_R_errors.append(np.mean(reward_error))
        V_vector_estimated_history.append(agent.V_vector_estimated)
        V_error_history.append(utils.V_error_calculation(V_ground_truth,
                                             agent.V_vector_estimated))
        sf_mat_history.append(copy.deepcopy(agent.estimated_SR))

    return step_lengths, sf_mat_history, V_vector_estimated_history, \
        V_error_history
