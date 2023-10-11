import pickle5 as pickle
import re

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#%matplotlib inline

save_fig_path = "./simulated_results/20210729/"
save_file_name = 'total_alpha0p1_re.pkl'

with open(save_fig_path+save_file_name, 'rb') as f:
    exp_dic = pickle.load(f)


agents_keys = list(exp_dic.keys()) 
corridor_size_keys = list(exp_dic[agents_keys[1]].keys())
data_types = list(exp_dic[agents_keys[1]][corridor_size_keys[0]].keys())

fig_agents = agents_keys[1:]
agents_label = ['SR', 'I', 'zero', 'Xavier', 'He', 'uniform']
fig_corridor = ['5states', '25states', '50states', '100states']
fig_int = [5, 25, 50, 100]

# whole corridor (except for 2)
corridor_size_int_list = []
for corridor_str in corridor_size_keys:
    reg_rule = re.compile("\d+") # digit one or more(+) and "states" text match
    matched_text = reg_rule.search(corridor_str)
    n_states_int = int(matched_text.group())
    corridor_size_int_list.append(n_states_int)
corridor_size_int_list = corridor_size_int_list[1:]
corridor_size_str_list = [str(x) + 'states' for x in corridor_size_int_list]


def step_length_fill(agent, corridor_size):
    step_length = exp_dic[agent][corridor_size][data_types[0]]
    upper = exp_dic[agent][corridor_size][data_types[0]] + \
            exp_dic[agent][corridor_size][data_types[1]]
    lower = exp_dic[agent][corridor_size][data_types[0]] - \
            exp_dic[agent][corridor_size][data_types[1]]
    epi_len_x = np.arange(len(exp_dic[agent][corridor_size][data_types[0]]))
    return step_length, upper, lower, epi_len_x

def step_length_mean(agent, corridor_size):
    return np.mean(exp_dic[agent][corridor_size][data_types[0]][:100])


def dstep_depi_fill(agent, corridor_size):
    ds_de = exp_dic[agent][corridor_size][data_types[10]]
    epi_len = len(exp_dic[agent][corridor_size][data_types[10]])
    epi_len_x = np.arange(epi_len)
    upper_bound = exp_dic[agent][corridor_size][data_types[10]] + \
        exp_dic[agent][corridor_size][data_types[11]]
    lower_bound = exp_dic[agent][corridor_size][data_types[10]] - \
        exp_dic[agent][corridor_size][data_types[11]]
    return ds_de, upper_bound, lower_bound, epi_len_x,

def std_dstep_depi(agent, corridor_size):
    ds_de = exp_dic[agent][corridor_size][data_types[10]][:100]
    return np.std(ds_de)

step_diff_std_list = []
step_mean_list = []
for corridor in corridor_size_str_list:
    std_corridor = []
    mean_corridor = []
    for agent in fig_agents:
        std_ds_de = std_dstep_depi(agent, corridor)
        mean = step_length_mean(agent, corridor)
        std_corridor.append(std_ds_de)
        mean_corridor.append(mean)
    step_diff_std_list.append(std_corridor)
    step_mean_list.append(mean_corridor)

step_diff_std_np = np.array(step_diff_std_list).transpose()
step_mean_np = np.array(step_mean_list).transpose()



fig = plt.figure(constrained_layout = True, figsize= (11, 4.5))
subfigs= fig.subfigures(1,2, wspace = 0.04, width_ratios = [3,1])

axsLeft = subfigs[0].subplots(2, 4, sharex = 'col')
subfigs[0].text(0, 0.96, 'A', fontsize = 14, fontweight = 'bold', va = 'bottom')


for idx_col, corridor_size in enumerate(fig_corridor):
    for idx_a, agent in enumerate(fig_agents):
        step_length, upper, lower, epi_len_x = step_length_fill(agent, corridor_size)
        axsLeft[0, idx_col].plot(step_length, label = agents_label[idx_a], alpha = 0.8)
        axsLeft[0, idx_col].fill_between(epi_len_x, lower, upper, alpha = 0.3)
        axsLeft[0, idx_col].set_title("{} cells".format(fig_int[idx_col]), fontsize = 11)
        axsLeft[0, idx_col].set_xscale('log')
        #axsLeft[idx_col].loglog()
        axsLeft[0, idx_col].ticklabel_format(style = 'sci', useMathText=True, axis = 'y', scilimits = (-1,2))

#axsLeft[0, 0].set_yticks([0, 3, 6, 9, 12, 15])

axsLeft[0, 0].set_ylabel('step length')
#axsLeft[0, 2].legend()


for idx_col, corridor_size in enumerate(fig_corridor):
    for idx_a, agent in enumerate(fig_agents):
        dstep_de, upper, lower, epi_len_x = dstep_depi_fill(agent, corridor_size)
        axsLeft[1, idx_col].plot(dstep_de, alpha =0.7, label = agents_label[idx_a])
        axsLeft[1, idx_col].fill_between(epi_len_x, lower, upper, alpha =0.3)
        axsLeft[1, idx_col].ticklabel_format(style = 'sci', useMathText=True, axis = 'y', scilimits = (-1,2))

axsLeft[1, 0].set_ylabel(r'$\frac{\Delta\mathrm{(step \enspace length)}}{\Delta\mathrm{episode}}$')
axsLeft[1, 1].set_xlabel('Episodes', x = .8)
axsLeft[1, 0].set_xticks([10, 100], minor= True)
axsLeft[0, 0].legend()

axsRight = subfigs[1].subplots(2,1)

subfigs[1].text(0, 0.96, 'B', fontsize = 14, fontweight = 'bold', va = 'bottom')

for idx_a, agent in enumerate(fig_agents):
    axsRight[0].plot(corridor_size_int_list, step_mean_np[idx_a], label = agents_label[idx_a],
        alpha = 0.7, marker = '.', markersize = 5)

for idx_a, agent in enumerate(fig_agents):
    axsRight[1].plot(corridor_size_int_list, step_diff_std_np[idx_a], label = agents_label[idx_a],
        alpha = 0.7, marker = '.', markersize = 5)

axsRight[0].legend()
#axsRight[0].set_yscale('log')
axsRight[0].ticklabel_format(style = 'sci', useMathText=True, axis = 'y', scilimits = (-1,2))
axsRight[1].ticklabel_format(style = 'sci', useMathText=True, axis = 'y', scilimits = (-1,2))
axsRight[0].set_ylabel(r'(step length)$_{mean \mathrm{\enspace of \enspace first \enspace 100 \enspace episodes}}$', fontsize = 9)
axsRight[1].set_xlabel('Number of cells')
axsRight[1].set_ylabel(r'SD of $\frac{\Delta\mathrm{(step \enspace length)}}{\Delta\mathrm{episode}}_{\mathrm{first \enspace 100 \enspace epi}}$', fontsize = 9)

plt.savefig('./images/Figure_step_length.png', dpi=600)
plt.close()

