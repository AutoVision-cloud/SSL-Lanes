import pickle
import random
import numpy as np
import collections
from argoverse.map_representation.map_api import ArgoverseMap

x = np.load('train_intention.p', allow_pickle=True)


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


ctr_count = np.zeros([6])


def write_file():
    f = open("train_maneuver_imbalance.p", "wb")
    op = 'y'

    while op == 'y':
        pickle_list = []
        for idx in range(len(x)):
            if idx % 1000 == 0:
                print(idx)
            data_orig = x[idx]

            if data_orig['intention_cls'] == 1:
                if ctr_count[1] < 20000:
                    new_data = dict()
                    for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
                        if key in data_orig:
                            new_data[key] = ref_copy(data_orig[key])
                    data = new_data
                    pickle_list.append(data)
                    ctr_count[1] += 1
            else:
                intent_cls = data_orig['intention_cls']
                if ctr_count[intent_cls] < 10000:
                    new_data = dict()
                    for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
                        if key in data_orig:
                            new_data[key] = ref_copy(data_orig[key])
                    data = new_data
                    pickle_list.append(data)
                    ctr_count[intent_cls] += 1

        pickle.dump(pickle_list, f)
        op = 'n'
    f.close()


print("choose maneuver imbalanced data in the pickle file")
write_file()
