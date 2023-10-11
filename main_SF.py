# load library
import numpy as np
from tqdm import tqdm
import pickle

# custom library
from exp import SF_1D
from utils import dstep_depi, dV_depi

corridor_size_list = [2, 3, 4, 5, 10, 11, 13, 15, 17, 19, 25, 30, 50, 75, 100] 

gamma = 0.95
alpha_r = 0.1
alpha_w_list = [0.1]

epsilon = 0.95 #intial epsilon


total_trials = 10



weight_init_set = ["eye", "zero", "random", "He", "uni"] # eye, zero, random,(He, uni did for revision at 20221212)
for weight_init in tqdm(weight_init_set, desc="weight init exp"):
        
    save_fig_path = "./simulated_results/20210729/" + weight_init +"/"


    for corridor_size in tqdm(corridor_size_list, desc="corridor size"):
        episodes = corridor_size * 100

        corridor_size_str = str(corridor_size) + "states"
        SF_exp = {'metadata': {'alpha': alpha_w_list}}

        for alpha_w in tqdm(alpha_w_list, desc = "alpha test"):
            
            for trial in tqdm(range(total_trials), desc="trial of SF"):
                step_length, sf_mat_history, V_est, V_error_history = \
                    SF_1D(episodes = episodes, alpha_r = alpha_r, gamma = gamma, \
                        corridor_size = corridor_size, weight_init=weight_init,\
                         epsilon_dic = epsilon, explora= True)
                if trial == 0:
                    trials_step_lengths = [step_length]
                    trials_matrices = [sf_mat_history]
                    trials_V_est = [V_est]
                    trials_V_error_history = [V_error_history]
                    trials_dV_history = [dV_depi(V_error_history)]
                    trials_dStep_history = [dstep_depi(step_length)]
                else:
                    trials_step_lengths.append(step_length)
                    trials_matrices.append(sf_mat_history)
                    trials_V_est.append(V_est)
                    trials_V_error_history.append(V_error_history)
                    trials_dV_history.append(dV_depi(V_error_history))
                    trials_dStep_history.append(dstep_depi(step_length))


            sf_mean_step_lengths = np.mean(np.array(trials_step_lengths), axis =0)
            sf_std_step_lengths = np.std(np.array(trials_step_lengths), axis = 0)
            sf_mean_matrices = np.mean(np.array(trials_matrices), axis = 0)
            sf_std_matrices = np.std(np.array(trials_matrices), axis = 0)
            sf_mean_V_est = np.mean(np.array(trials_V_est), axis=0)
            sf_std_V_est = np.std(np.array(trials_V_est), axis=0)
            sf_mean_V_error_history = np.mean(np.array(trials_V_error_history), axis = 0)
            sf_std_V_error_history = np.std(np.array(trials_V_error_history), axis = 0)
            sf_mean_dV = np.mean(np.array(trials_dV_history), axis = 0)
            sf_std_dV = np.std(np.array(trials_dV_history), axis = 0)
            sf_mean_dStep = np.mean(np.array(trials_dStep_history), axis = 0)
            sf_std_dStep = np.std(np.array(trials_dStep_history), axis = 0)
            
            alpha_w_str = 'alpha_w='+str(alpha_w)
            SF_exp[alpha_w_str] = {'mean of steps': sf_mean_step_lengths,
                        'std of steps': sf_std_step_lengths,
                        'SR matrix': sf_mean_matrices,
                        'std of SR matrix': sf_mean_matrices,
                        'mean of V estimates': sf_mean_V_est,
                        'std of V estimated': sf_std_V_est,
                        'mean of V error': sf_mean_V_error_history,
                        'std of V error': sf_std_V_error_history, 
                        'mean of dV': sf_mean_dV,
                        'std of dV': sf_std_dV,
                        'mean of dStep': sf_mean_dStep,
                        'std of dStep': sf_std_dStep}
                
        save_file_name = '1DimMaze_'+str(corridor_size)+'states.pkl'
        with open(save_fig_path + save_file_name, 'wb') as f:
            pickle.dump(SF_exp, f, pickle.HIGHEST_PROTOCOL)


print('\n\n')

if __name__=='__main__':
    pass