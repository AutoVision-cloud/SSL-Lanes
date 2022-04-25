import pickle
import random
import numpy as np
import collections
from argoverse.map_representation.map_api import ArgoverseMap

x = np.load('train_crs_dist6_angle90.p', allow_pickle=True)


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
    f = open("train_dataset_0.25.p", "wb")
    op = 'y'

    while op == 'y':
        pickle_list = []
        for idx in range(len(x)):
            if idx % 1000 == 0:
                print(idx)
            data_orig = x[idx]
            # Choose data with 25% probability
            rand = random.randrange(0, 3)
            if rand == 0:
                new_data = dict()
                for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
                    if key in data_orig:
                        new_data[key] = ref_copy(data_orig[key])
                data = new_data
                pickle_list.append(data)
        pickle.dump(pickle_list, f)
        op = 'n'
    f.close()


print("choose quarter the data in the pickle file")
write_file()
