import pickle5 as pickle
import re
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
plt.switch_backend('agg')
#%matplotlib inline

#matplotlib.rcParams['text.usetex'] = True
save_file_path = "./simulated_results/20210729/"
save_file_name = 'total_alpha0p1_re.pkl'

with open(save_file_path+save_file_name, 'rb') as f:
    exp_dic = pickle.load(f)

agents_keys = list(exp_dic.keys()) 
corridor_size_keys = list(exp_dic[agents_keys[1]].keys())
data_types = list(exp_dic[agents_keys[1]][corridor_size_keys[1]].keys())

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

def v_mean_last(agent, corridor_size):
    return np.mean(exp_dic[agent][corridor_size][data_types[6]][-100:])

#def v_std_last(agent, corridor_size):
#    return np.std(exp_dic[agent][corridor_size][data_types[6]][-100:])

v_mean_last_list = []
#v_std_last_list = []
for agent in fig_agents:
    v_mean_agent = []
    #v_std_agent = []
    for corridor_size in corridor_size_str_list:
        v_mean_agent.append(v_mean_last(agent, corridor_size))
#        v_std_agent.append(v_std_last(agent, corridor_size))
    v_mean_last_list.append(v_mean_agent)
#    v_std_last_list.append(v_std_agent)
    

def v_error(agent, corridor_size):
    return exp_dic[agent][corridor_size][data_types[6]]

def v_fill(agent, corridor_size):
    epi_len = len(exp_dic[agent][corridor_size][data_types[6]])
    epi_len_x = np.arange(epi_len)
    upper_bound = exp_dic[agent][corridor_size][data_types[6]] + \
        exp_dic[agent][corridor_size][data_types[7]]
    lower_bound = exp_dic[agent][corridor_size][data_types[6]] - \
        exp_dic[agent][corridor_size][data_types[7]]
    return epi_len_x, upper_bound, lower_bound


def dv_depi(agent, corridor_size):
    return exp_dic[agent][corridor_size][data_types[8]]

def dv_depi_fill(agent, corridor_size):
    epi_len = len(exp_dic[agent][corridor_size][data_types[8]])
    epi_len_x = np.arange(epi_len)
    upper_bound = exp_dic[agent][corridor_size][data_types[8]] + \
        exp_dic[agent][corridor_size][data_types[9]]
    lower_bound = exp_dic[agent][corridor_size][data_types[8]] - \
        exp_dic[agent][corridor_size][data_types[9]]
    return epi_len_x, upper_bound, lower_bound


def mean_dV_depi(agent, corridor_size):
    dV_de = exp_dic[agent][corridor_size][data_types[8]]
    return np.mean(dV_de[:10])
    

dv_mean_first_list = []
for agent in fig_agents:
    dv_mean_agent = []
    for corridor_size in corridor_size_str_list:
        dv_mean_agent.append(mean_dV_depi(agent, corridor_size))
    dv_mean_first_list.append(dv_mean_agent)



fig = plt.figure(constrained_layout = True, figsize = (11, 5))
subfigs = fig.subfigures(1,2,wspace = 0.04, width_ratios = [3,1])

axsLeft = subfigs[0].subplots(2,4, sharey = False, sharex= 'col')



for idx_col, corridor_size in enumerate(fig_corridor):
    for idx_a, agent in enumerate(fig_agents):
        v_error_value = v_error(agent, corridor_size)
        axsLeft[0, idx_col].plot(v_error_value, label = agents_label[idx_a], alpha = 0.8)
        
        epi_len_x, upper_bound,lower_bound = v_fill(agent, corridor_size)
        axsLeft[0, idx_col].fill_between(epi_len_x, v_error_value, upper_bound, alpha = 0.3)
    axsLeft[0, idx_col].set_title("{} cells".format(fig_int[idx_col]))
    axsLeft[0, idx_col].loglog()

axsLeft[0, 0].set_ylabel(r'MSE from $V^{*}$')
axsLeft[0, 0].legend(loc = 'lower left')

for idx_col, corridor_size in enumerate(fig_corridor):
    for idx_a, agent in enumerate(fig_agents):
        dv_de_value = dv_depi(agent, corridor_size)
        axsLeft[1, idx_col].plot(dv_de_value, alpha = 0.8, label = agents_label[idx_a])
        epi_len_x, upper_bound, lower_bound = dv_depi_fill(agent, corridor_size)
        axsLeft[1, idx_col].fill_between(epi_len_x, lower_bound, upper_bound, alpha = 0.3)
        axsLeft[1, idx_col].ticklabel_format(style = 'sci', useMathText=True, axis = 'y', scilimits = (0,0))


axsLeft[1, 0].set_ylabel(r'$-\frac{\Delta \mathrm{MSE}}{\Delta \mathrm{episode}}$')
axsLeft[1, 1].set_xlabel('Episodes', x = 1.)
axsLeft[1, 0].set_xticks([10, 100], minor= True)
subfigs[0].text(0, 0.95, 'A', fontsize = 14, fontweight = 'bold', va = 'bottom')


axsRight = subfigs[1].subplots(2,1, sharex = True)


for idx_f, agent in enumerate(fig_agents):
    axsRight[0].plot(corridor_size_int_list, v_mean_last_list[idx_f],\
         label = agents_label[idx_f], alpha = 0.8, marker = '.', markersize = 5)

axsRight[0].set_ylabel(r'MSE$_{mean \mathrm{\enspace of \enspace last \enspace 100 \enspace episodes}}$')

axsRight[0].legend(loc = 'lower right')

subfigs[1].text(0, 0.95, 'B', fontsize = 14, fontweight = 'bold', va = 'bottom')

for idx_f, agent in enumerate(fig_agents):
    axsRight[1].plot(corridor_size_int_list, dv_mean_first_list[idx_f], label = agents_label[idx_f],
        alpha = 0.8, marker = '.', markersize = 5)
axsRight[1].set_ylabel(r'$(-\frac{\Delta \mathrm{MSE}}{\Delta \mathrm{episode}})_{mean \mathrm{\enspace of \enspace first \enspace 10 \enspace episodes}}$')
axsRight[1].set_xlabel('Number of cells')
axsRight[0].set_yscale('log')
#axsRight[1].set_yscale('log')
axsRight[1].set_xticks([25, 50, 75, 100])
axsRight[1].ticklabel_format(style = 'scientific', useMathText=True, axis = 'y', scilimits = (0,0))


plt.savefig('./images/Figure_value_error.png', dpi= 600)
plt.close()
