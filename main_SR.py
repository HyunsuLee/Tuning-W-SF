# load library
import numpy as np
from tqdm import tqdm
import pickle

# custom library
from exp import SR_1D
from utils import dstep_depi, dV_depi

corridor_size_list = [2, 3, 4, 5, 10, 11, 13, 15, 17, 19, 25, 30, 50, 75, 100] 

save_dic_path = "./simulated_results/20210729/SR/"


gamma = 0.95
alpha_r = 0.1
alpha_m = 0.1
alpha_key_str = 'alpha=0.1'

epsilon = 0.95 #intial epsilon


total_trials = 10



for corridor_size in tqdm(corridor_size_list, desc="corridor size"):
    episodes = corridor_size * 100

    
    SR_exp = {}
        
    for trial in tqdm(range(total_trials), desc="trial of SR"):
        step_length, sr_mat_history, V_est, V_error_history = \
            SR_1D(episodes = episodes, alpha_r = alpha_r, gamma = gamma, \
            corridor_size = corridor_size, epsilon_dic = epsilon, explora= True)
        if trial == 0:
            trials_step_lengths = [step_length]
            trials_matrices = [sr_mat_history]
            trials_V_est = [V_est]
            trials_V_error_history = [V_error_history]
            trials_dV_history = [dV_depi(V_error_history)]
            trials_dStep_history = [dstep_depi(step_length)]
        else:
            trials_step_lengths.append(step_length)
            trials_matrices.append(sr_mat_history)
            trials_V_est.append(V_est)
            trials_V_error_history.append(V_error_history)
            trials_dV_history.append(dV_depi(V_error_history))
            trials_dStep_history.append(dstep_depi(step_length))


    sr_mean_step_lengths = np.mean(np.array(trials_step_lengths), axis =0)
    sr_std_step_lengths = np.std(np.array(trials_step_lengths), axis = 0)
    sr_mean_matrices = np.mean(np.array(trials_matrices), axis = 0)
    sr_std_matrices = np.std(np.array(trials_matrices), axis=0)
    sr_mean_V_est = np.mean(np.array(trials_V_est), axis=0)
    sr_std_V_est = np.std(np.array(trials_V_est), axis=0)
    sr_mean_V_error_history = np.mean(np.array(trials_V_error_history), axis = 0)
    sr_std_V_error_history = np.std(np.array(trials_V_error_history), axis = 0)
    sr_mean_dV = np.mean(np.array(trials_dV_history), axis = 0)
    sr_std_dV = np.std(np.array(trials_dV_history), axis = 0)
    sr_mean_dStep = np.mean(np.array(trials_dStep_history), axis = 0)
    sr_std_dStep = np.std(np.array(trials_dStep_history), axis = 0)


    SR_exp[alpha_key_str] = {'mean of steps': sr_mean_step_lengths,
                'std of steps': sr_std_step_lengths,
                'SR matrix': sr_mean_matrices,
                'std of SR matrix': sr_std_matrices,
                'mean of V estimates': sr_mean_V_est,
                'std of V estimated': sr_std_V_est,
                'mean of V error': sr_mean_V_error_history,
                'std of V error': sr_std_V_error_history,
                'mean of dV': sr_mean_dV,
                'std of dV': sr_std_dV,
                'mean of dStep': sr_mean_dStep,
                'std of dStep': sr_std_dStep}

    save_file_name = '1DimMaze_'+str(corridor_size)+'states.pkl'
    with open(save_dic_path + save_file_name, 'wb') as f:
        pickle.dump(SR_exp, f, pickle.HIGHEST_PROTOCOL)


print('\n\n')

if __name__=='__main__':
    pass
