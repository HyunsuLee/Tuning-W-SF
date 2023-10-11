from sklearn.decomposition import PCA

import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')

PCA_save_path = './simulated_results/20210729/'
dic_file_name = 'PCA_for_sel_state_re.pkl'

with open(PCA_save_path + dic_file_name, 'rb') as f:
    PCA_dic = pickle.load(f)

fig_corridor = list(PCA_dic.keys())
fig_int = [5, 25, 50, 100]
fig_agents = list(PCA_dic[fig_corridor[0]].keys())
agents_label = ['SR', 'I', 'zero', 'Xavier', 'He', 'uniform']



fig, axes = plt.subplots(2,2, constrained_layout = True, sharex= False, sharey= False)

fig.set_size_inches((6, 6))

state_idx = 0
for idx_row in range(2):
    for idx_col in range(2):
        for idx_agent, agent in enumerate(fig_agents):
            state_n = fig_corridor[state_idx]
            state_str = str(fig_int[state_idx])
            PCA_to_plot = PCA_dic[state_n][agent]
            axes[idx_row, idx_col].plot(PCA_to_plot[:, 0], PCA_to_plot[:, 1],
                marker = ".", label = agents_label[idx_agent], alpha = 0.4)
            axes[idx_row, idx_col].annotate('*',
                xy = (PCA_to_plot[0, 0], PCA_to_plot[0, 1]),
                color = plt.cm.tab10(idx_agent))
            axes[idx_row, idx_col].set_title(state_str +' cells', fontsize = 10)
            

        state_idx += 1
    axes[0,1].legend(loc = 'lower right') #bbox_to_anchor = (1.0, 0.5)

axes[1, 0].set_ylabel('PC2', y = 1.1, labelpad = 1., fontsize = 12)
axes[1, 1].set_xlabel('PC1', x = -0.1, fontsize = 12)




plt.savefig('./images/Figure_PCA.png', dpi = 600)

plt.close()


## For supple figure
fig, axes = plt.subplots(2,2, constrained_layout = True, sharex= True, sharey= True)

fig.set_size_inches((6, 6))



state_idx = 0
for idx_row in range(2):
    for idx_col in range(2):
        for idx_agent, agent in enumerate(fig_agents):
            state_n = fig_corridor[state_idx]
            state_str = str(fig_int[state_idx])
            PCA_to_plot = PCA_dic[state_n][agent]
            axes[idx_row, idx_col].plot(PCA_to_plot[:, 0], PCA_to_plot[:, 1],
                marker = ".", label = agents_label[idx_agent], alpha = 0.4)
            axes[idx_row, idx_col].annotate('*',
                xy = (PCA_to_plot[0, 0], PCA_to_plot[0, 1]),
                color = plt.cm.tab10(idx_agent))
            axes[idx_row, idx_col].set_title(state_str +' cells')

        state_idx += 1
    axes[0,0].legend()

axes[1, 0].set_ylabel('PC2', y = 1.1, labelpad = 1., fontsize = 12)
axes[1, 1].set_xlabel('PC1', x = -0.1, fontsize = 12)


plt.savefig('./images/Supple_PCA_same_axis.png', dpi = 600)

plt.close()
