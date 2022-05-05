import pickle
import numpy as np
import collections
from argoverse.map_representation.map_api import ArgoverseMap
import scipy


cluster_centers = np.load('cluster_centers.npy')
x = np.load('train_crs_dist6_angle90.p', allow_pickle=True)
avm = ArgoverseMap()


def get_confidence_flag(city, orig, final):
    flag_confidence = 1
    lane_dir1 = avm.get_lane_direction(query_xy_city_coords=orig, city_name=city, visualize=False)
    lane_dir2 = avm.get_lane_direction(query_xy_city_coords=final, city_name=city, visualize=False)
    if lane_dir1[-1] < 0.8 or lane_dir2[-1] < 0.8:
        flag_confidence = -1
    return flag_confidence


def get_lane_turn_flag(city, orig, final):
    lane_turn_flag = 1

    lane_turn_orig = avm.get_lane_turn_direction(avm.get_nearest_centerline(orig, city)[0].id, city)
    lane_turn_final = avm.get_lane_turn_direction(avm.get_nearest_centerline(final, city)[0].id, city)

    if lane_turn_final != 'NONE' or lane_turn_orig != 'NONE':
        lane_turn_flag = -1
    return lane_turn_flag


def get_intention(traj_norm, city, orig, final):
    tmp_idx = -1
    tmp_dist = 1000

    for i in range(5):
        dist = scipy.spatial.distance.cosine(cluster_centers[i], traj_norm[-1])
        if dist < tmp_dist:
            tmp_idx = i
            tmp_dist = dist

    idx_sel = -1
    dist_sel = 1000

    if tmp_idx < 3:
        if get_confidence_flag(city, orig, final) == -1 and get_lane_turn_flag(city, orig, final) == 1:
            idx_sel = 5
        else:
            for i in range(3):
                dist1 = np.linalg.norm(cluster_centers[i] - traj_norm[-1])
                if dist1 < dist_sel:
                    idx_sel = i
                    dist_sel = dist1
    else:
        for i in range(3, 5):
            dist1 = np.linalg.norm(cluster_centers[i] - traj_norm[-1])
            if dist1 < dist_sel:
                idx_sel = i
                dist_sel = dist1
    return idx_sel


def write_file():
    f = open("train_intention.p", "wb")
    op = 'y'

    while op == 'y':
        pickle_list = []
        for idx in range(len(x)):
            if idx % 1000 == 0:
                print(idx)
            data = x[idx]
            # ----------------new, get intention---------------------
            traj = data['gt_preds'][0][:np.where(data['has_preds'][0])[0][-1]]
            traj_norm = np.matmul(data['rot'], (traj - data['orig'].reshape(-1, 2)).T).T
            city = data['city']
            orig = data['orig']
            final = data['gt_preds'][0][np.where(data['has_preds'][0])[0][-1]]
            deg = np.rad2deg(np.arctan2(traj_norm[-1][-1], traj_norm[-1][0]))
            if abs(deg) > 90:
                traj_norm[:, 0] = - traj_norm[:, 0]
                traj_norm[:, 1] = - traj_norm[:, 1]
            intention_cls = get_intention(traj_norm, city, orig, final)
            data['intention_cls'] = intention_cls
            # ----------------new, get intention---------------------
            pickle_list.append(data)
        pickle.dump(pickle_list, f)
        op = 'n'
    f.close()


print("appending intention to trajs in the pickle file")
write_file()
