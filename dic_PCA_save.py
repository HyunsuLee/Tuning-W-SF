from sklearn.decomposition import PCA

import numpy as np
import pickle


save_file_path = "./simulated_results/20210729/"
save_file_name = 'total_alpha0p1_re.pkl'

with open(save_file_path+save_file_name, 'rb') as f:
    exp_dic = pickle.load(f)

agents_keys = list(exp_dic.keys()) 
corridor_size_keys = list(exp_dic[agents_keys[0]].keys())
data_types = list(exp_dic[agents_keys[1]][corridor_size_keys[0]].keys())

fig_agents = agents_keys[1:]

fig_corridor = ['5states', '25states', '50states', '100states']
fig_int = [5, 25, 50, 100]

PCA_dic = {}

for state_n in fig_corridor:
    SR_hist = exp_dic['SR'][state_n][data_types[2]][:,:,:]
    eye_hist = exp_dic['eye'][state_n][data_types[2]][:,:,:]
    zero_hist = exp_dic['zero'][state_n][data_types[2]][:,:,:]
    rand_hist = exp_dic['random'][state_n][data_types[2]][:,:,:] #Xavier method
    he_hist = exp_dic['He'][state_n][data_types[2]][:,:,:]
    uni_hist = exp_dic['uni'][state_n][data_types[2]][:,:,:]

    all = np.vstack((SR_hist, eye_hist, zero_hist, rand_hist, he_hist, uni_hist))
    epis, states, _ = all.shape
    re_all = all.reshape(-1, states ** 2)

    pca = PCA(n_components=2)

    pca.fit(re_all)
    SR_pcs = pca.transform(SR_hist.reshape(-1, states **2))
    eye_pcs = pca.transform(eye_hist.reshape(-1, states **2))
    zero_pcs = pca.transform(zero_hist.reshape(-1, states **2))
    rand_pcs = pca.transform(rand_hist.reshape(-1, states **2))
    he_pcs = pca.transform(he_hist.reshape(-1, states **2))
    uni_pcs = pca.transform(uni_hist.reshape(-1, states **2))

    PCA_dic[state_n] = {'SR': SR_pcs, 'eye': eye_pcs, 
                        'zero': zero_pcs, 'random': rand_pcs,
                        'He': he_pcs, 'uniform': uni_pcs}



dic_file_name = 'PCA_for_sel_state_re.pkl'
with open(save_file_path + dic_file_name, 'wb') as f:
    pickle.dump(PCA_dic, f, pickle.HIGHEST_PROTOCOL)

