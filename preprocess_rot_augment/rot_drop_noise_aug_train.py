import pickle
import numpy as np
import collections
from argoverse.map_representation.map_api import ArgoverseMap
import random


def give_idx(a_range):
    idx_select = np.random.binomial(1, 0.9, a_range-1)
    return np.concatenate([[0], np.where(idx_select==1)[0]+1])


x = np.load('train_crs_dist6_angle90.p', allow_pickle=True)
angles = np.arange(0, 360, 30)


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


def write_file():
    f = open("train_rot_aug.p", "wb")
    op = 'y'

    while op == 'y':
        pickle_list = []
        for idx in range(len(x)):
            if idx % 1000 == 0:
                print(idx)
            data_orig = x[idx]
            
            # aug2: dropout agents with prob 0.1
            a_range = len(data_orig['gt_preds'])

            for j in range(len(angles)):
                selected_agents = give_idx(a_range)
                new_data = dict()
                
                for key in ['city', 'orig', 'gt_preds', 'has_preds']:
                    if key in data_orig:
                        if key == 'gt_preds' or key=='has_preds':
                            if j > 0:
                                new_data[key] = ref_copy(data_orig[key][selected_agents])
                            else:
                                new_data[key] = ref_copy(data_orig[key])
                        else:
                            new_data[key] = ref_copy(data_orig[key])
                
                # aug1: rotation          
                angle_idx = j
                dt = angles[angle_idx] * np.pi / 180
                theta = data_orig['theta'] + dt
                new_data['theta'] = theta
                new_data['rot'] = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)
                rot = np.asarray([
                    [np.cos(-dt), -np.sin(-dt)],
                    [np.sin(-dt), np.cos(-dt)]], np.float32)
                new_data['feats'] = data_orig['feats'].copy()
                new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
                new_data['ctrs'] = np.matmul(data_orig['ctrs'], rot)
                
                if j > 0:
                    # aug2: dropout agents with prob 0.1
                    new_data['feats'] = new_data['feats'][selected_agents].copy()
                    new_data['ctrs'] = new_data['ctrs'][selected_agents].copy()
                
                    if random.randint(0, 1) == 1:
                        # aug3: add random noise
                        new_data['feats'][:, 1:, :2] += np.random.normal(0, 0.2, [new_data['feats'].shape[0], 19, 2]).copy()

                graph = dict()
                for key in ['num_nodes', 'turn', 'control', 'intersect', 'pre', 'suc', 'lane_idcs', 'left_pairs',
                            'right_pairs', 'left', 'right']:
                    graph[key] = ref_copy(data_orig['graph'][key])
                graph['ctrs'] = np.matmul(data_orig['graph']['ctrs'], rot)
                graph['feats'] = np.matmul(data_orig['graph']['feats'], rot)
                new_data['graph'] = graph
                data = new_data
                pickle_list.append(data)
        pickle.dump(pickle_list, f)
        op = 'n'
    f.close()


print("rot augmenting trajs in the pickle file")
write_file()
