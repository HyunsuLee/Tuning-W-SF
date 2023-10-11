
import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')

save_file_path = "./simulated_results/20210729/"
save_file_name = 'total_alpha0p1_re.pkl'

with open(save_file_path+save_file_name, 'rb') as f:
    exp_dic = pickle.load(f)

agents_keys = list(exp_dic.keys()) 
corridor_size_keys = list(exp_dic[agents_keys[0]].keys())
data_types = list(exp_dic[agents_keys[1]][corridor_size_keys[0]].keys())

fig_agents = agents_keys[1:]
agents_label = ['SR', 'I', 'zero', 'Xavier', 'He', 'uniform']
fig_corridor = ['5states', '25states', '50states', '100states']
fig_int = [5, 25, 50, 100]
fig_legend = [str(x) + ' cells' for x in fig_int]

def norm_dist_mat(hist_mat1, hist_mat2):
    _, size, _ = hist_mat1.shape
    dis = np.abs(hist_mat1 - hist_mat2)
    sum = np.sum(dis, axis=(1,2))
    return sum, sum/(size**2)

dis_dic = {}
norm_dis_dic = {}

for state_n in fig_corridor:
    dis_dic[state_n] = {}
    norm_dis_dic[state_n] = {}
    for agent1 in fig_agents:
        dis_dic[state_n][agent1] = {}
        norm_dis_dic[state_n][agent1] = {}
        agent1_hist = exp_dic[agent1][state_n][data_types[2]][:,:,:]
        for agent2 in fig_agents:
            agent2_hist = exp_dic[agent2][state_n][data_types[2]][:,:,:]
            dis_sum, norm_dis = norm_dist_mat(agent1_hist, agent2_hist)
            dis_dic[state_n][agent1][agent2] = dis_sum
            norm_dis_dic[state_n][agent1][agent2] = norm_dis

def max_dist(dis_dic, agent1, agent2):
    max_epi = np.argmax(dis_dic['100states'][agent1][agent2])
    max_val = np.max(dis_dic['100states'][agent1][agent2])
    return max_epi, max_val


fig, axes = plt.subplots(6,6, constrained_layout = False, sharey=True, sharex=True)
fig.set_size_inches((10, 10))

# to add common x, y label
ax = fig.add_subplot(111, frameon=False)
ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
ax.set_xlabel("Relative episodes", fontsize = 12)
ax.set_ylabel("L1 distance", labelpad = 15, fontsize = 12)


colors = plt.cm.Greens(np.linspace(0.3,0.9,len(fig_corridor)))

for idx_row, agent_row in enumerate(fig_agents):
    for idx_col, agent_col in enumerate(fig_agents):
        for idx, state_n in enumerate(fig_corridor):
            state_epi = len(dis_dic[state_n][agent_row][agent_col])
            state_x = np.arange(0, 1,1/state_epi)
            if idx_row > idx_col:
                axes[idx_row, idx_col].plot(state_x, dis_dic[state_n][agent_row][agent_col], 
                    label = state_n, color = colors[idx])
            elif idx_row == idx_col:
                axes[idx_row, idx_col].annotate('NA', xy = (0.01, 10), va = "center", ha = "center")
                axes[idx_row, idx_col].plot(state_x, state_x, color = 'white')
            else:
                axes[idx_row, idx_col].set_visible(False)
                #axes[idx_row, idx_col].set_axis_off()
            axes[idx_row, idx_col].set_xscale('log')
            axes[idx_row, idx_col].set_yscale('log')
            #axes[idx_row, idx_col].set_title(agents_label[idx_col], fontsize = 12)
            axes[idx_row, 0].set_ylabel(agents_label[idx_row], fontsize =12)
            

axes[0,1].set_ylim((0.02,3000))
axes[2,1].legend(fig_legend, loc = 'upper right', bbox_to_anchor = (3.5, 1.0))
#fig.tight_layout()
plt.savefig('./images/Figure_dist.png', dpi = 600)
plt.close()


# for Supple, mean distance of one element in SR matrix
fig, axes = plt.subplots(6,6, constrained_layout = False, sharey=True, sharex=True)
fig.set_size_inches((10, 10))

# to add common x, y label
ax = fig.add_subplot(111, frameon=False)
ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
ax.set_xlabel("Relative episodes", fontsize = 12)
#ax.set_ylabel("L1 distance", labelpad = 15, fontsize = 12)

# change color key for state, agent color key와 혼동 방지
colors = plt.cm.Greens(np.linspace(0.3,0.9,len(fig_corridor)))

for idx_row, agent_row in enumerate(fig_agents):
    for idx_col, agent_col in enumerate(fig_agents):
        for idx, state_n in enumerate(fig_corridor):
            state_epi = len(norm_dis_dic[state_n][agent_row][agent_col])
            state_x = np.arange(0, 1,1/state_epi)
            if idx_row > idx_col:
                axes[idx_row, idx_col].plot(state_x, norm_dis_dic[state_n][agent_row][agent_col], 
                    label = state_n, color = colors[idx])
                
            elif idx_row == idx_col:
                axes[idx_row, idx_col].annotate('NA', xy = (0.01, 0.02), va = "center", ha = "center")
                axes[idx_row, idx_col].plot(state_x, state_x, color = 'white')
            else:
                axes[idx_row, idx_col].set_visible(False)
            axes[idx_row, idx_col].set_xscale('log')
            axes[idx_row, idx_col].set_yscale('log')
            #axes[0, idx_col].set_title(agents_label[idx_col], fontsize = 12)
            axes[idx_row, 0].set_ylabel(agents_label[idx_row], fontsize =12)
            

axes[0,1].set_ylim((0.001,0.5))
axes[2,1].legend(fig_legend, loc = 'upper right', bbox_to_anchor = (3.5, 1.0))
#fig.tight_layout()
plt.savefig('./images/Supple_dist.png', dpi = 600)
plt.close()