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
    f = open("train_city_imbalance.p", "wb")
    op = 'y'

    while op == 'y':
        pickle_list = []
        for idx in range(len(x)):
            if idx % 1000 == 0:
                print(idx)
            data_orig = x[idx]

            # If city is PIT, choose 100% of the data
            # If city is 'MIA', choose 20% of the data
            if data_orig['city'] == 'PIT':
                new_data = dict()
                for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
                    if key in data_orig:
                        new_data[key] = ref_copy(data_orig[key])
                data = new_data
                pickle_list.append(data)
            else:
                rand = random.randrange(0, 9)
                if rand == 8 or rand == 9:
                    new_data = dict()
                    for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
                        if key in data_orig:
                            new_data[key] = ref_copy(data_orig[key])
                    data = new_data
                    pickle_list.append(data)

        pickle.dump(pickle_list, f)
        op = 'n'
    f.close()


print("choose 100% from PIT, 20% from MIA")
write_file()
