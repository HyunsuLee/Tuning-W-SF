
import pickle5 as pickle
#from matplotlib.cm import get_cmap
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#%matplotlib inline

save_file_path = "./simulated_results/20210729/"
save_file_name = 'total_alpha0p1_re.pkl'

with open(save_file_path+save_file_name, 'rb') as f:
    exp_dic = pickle.load(f)

agents_keys = list(exp_dic.keys()) 
corridor_size_keys = list(exp_dic[agents_keys[0]].keys())
data_types = list(exp_dic[agents_keys[1]][corridor_size_keys[0]].keys())

fig_agents = agents_keys[1:]
agents_label = ['SR', 'I', 'zero', 'Xavier', 'He', 'Uniform']

#print(exp_dic[agents_keys[1]]['30states'][data_types[2]][:,:,14])
# datatype[2]: SR matrix (episodes, M, M')

state = 100
state_str = str(state)+'states'
mid_state = (state-1)//2
sel_epi = [10, 25, 50, 100, 300, 500] # number of state에 따라 조정

def sr_fill(agent, corridor_size):
    sr = exp_dic[agent][corridor_size][data_types[2]][idx_e,:,mid_state]
    upper_bound = exp_dic[agent][corridor_size][data_types[2]][idx_e,:,mid_state] + \
        exp_dic[agent][corridor_size][data_types[3]][idx_e,:,mid_state]
    lower_bound = exp_dic[agent][corridor_size][data_types[2]][idx_e,:,mid_state] - \
        exp_dic[agent][corridor_size][data_types[3]][idx_e,:,mid_state]
    return sr, upper_bound, lower_bound

x = np.arange(1,state+1,1)


fig = plt.figure(constrained_layout = True, figsize = (12, 8))
subfigs = fig.subfigures(1,2, wspace = 0.03, width_ratios = [2.5, 6])


# Figure 2a
axsLeft = subfigs[0].subplots(6,2, sharex = True, sharey = 'col')

colors = plt.cm.Oranges(np.linspace(0.3,1,len(sel_epi)))

subfigs[0].text(0,0.98, 'A', fontsize = 14, fontweight = 'bold')
subfigs[0].text(0.5,0.98, 'B', fontsize = 14, fontweight = 'bold')



# historical plot for each agent
for fig_row, agent in enumerate(fig_agents):
    for idx, idx_e in enumerate(sel_epi): 
        sr, upper, lower = sr_fill(agent, state_str)
        axsLeft[fig_row, 0].plot(x, sr, label = str(idx_e), color = colors[idx], alpha = 0.8)
        axsLeft[fig_row, 0].fill_between(x, lower, upper, color = colors[idx], alpha = 0.3)
    axsLeft[fig_row,0].set_ylabel(agents_label[fig_row])

axsLeft[3,0].set_xticks([20, 40, 60, 80, 100])

axsLeft[5,0].legend(fontsize = 7)
#loc = 'upper left', ncol = 2, 
#    bbox_to_anchor = (0., 1.02, 1., 0.3), mode ="expand", borderaxespad = 0.)

# comparing agent 
for fig_row, idx_e in enumerate(sel_epi):
    for idx_a, agent in enumerate(fig_agents):
        sr, upper, lower = sr_fill(agent, state_str)
        axsLeft[fig_row, 1].plot(x, sr, label = agents_label[idx_a], alpha = 0.7)
        axsLeft[fig_row, 1].fill_between(x, lower, upper, alpha = 0.3)
    axsLeft[fig_row,1].set_ylabel("{}th".format(idx_e))

axsLeft[5,1].legend(fontsize = 7)

# Figure 2b
axsRight = subfigs[1].subplots(6, 6, sharey= True, sharex=True)
subfigs[1].text(-0.01,0.98, 'C', fontsize = 14, fontweight = 'bold')
#subfigs[1].text(0.5, 0., 'n-th cell', ha = 'center')


min = np.min(exp_dic[fig_agents[1]][state_str][data_types[2]])
max = np.max(exp_dic[fig_agents[1]][state_str][data_types[2]])

# Figure 2c
for idx_row, agent in enumerate(fig_agents):
    for idx_col, idx_e in enumerate(sel_epi):
        mat_hist = exp_dic[agent][state_str][data_types[2]][idx_e, :, :]
        im = axsRight[idx_row, idx_col].imshow(mat_hist, vmin = min, vmax = max, 
                cmap = 'Blues')
        axsRight[0, idx_col].set_title("{}th episode".format(idx_e), fontsize = 11)
    axsRight[idx_row, 0].set_ylabel(agents_label[idx_row])
        

axsRight[5, 2].set_xlabel('n-th cell', fontsize = 12, x = 1.1)
axsRight[0, 0].set_xticks([20, 40, 60, 80, 100])
axsRight[0, 0].set_yticks([20, 40, 60, 80, 100])

fig.colorbar(im, ax = axsRight, location = 'right',
              shrink = 0.8, aspect = 50, pad = 0.01)

#fig.tight_layout()
plt.savefig('./images/Figure_SR.png', dpi = 600)

plt.close()