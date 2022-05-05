import pickle
import numpy as np
import collections
from argoverse.map_representation.map_api import ArgoverseMap


x = np.load('val_crs_dist6_angle90.p', allow_pickle=True)
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


def get_lane_curve_flag(city, orig, final):
    lane_curv_flag = 1

    a = avm.get_nearest_centerline(orig, city)[0].centerline
    dir_lane = a[-1] - a[0]
    lane_curv_orig = np.arctan2(dir_lane[-1], dir_lane[0])

    b = avm.get_nearest_centerline(final, city)[0].centerline
    dir_lane = b[-1] - b[0]
    lane_curv_final = np.arctan2(dir_lane[-1], dir_lane[0])

    if lane_curv_final * lane_curv_orig < 0:
        lane_curv_flag = -1
    else:
        if abs(lane_curv_final - lane_curv_orig) > 0.1:
            lane_curv_flag = -1
    return lane_curv_flag


def write_file():
    f = open("val_difficult.p", "wb")
    op = 'y'

    while op == 'y':
        pickle_list = []
        for idx in range(len(x)):
            if idx % 1000 == 0:
                print(idx)
            data_sample = x[idx]
            city = data_sample['city']
            orig = data_sample['orig']
            final = data_sample['gt_preds'][0][-1]

            sc1 = get_confidence_flag(city, orig, final)
            sc2 = get_lane_turn_flag(city, orig, final)
            sc3 = get_lane_curve_flag(city, orig, final)

            if sc1 == -1 or sc2 == -1 or sc3 == -1:
                pickle_list.append(data_sample)
        pickle.dump(pickle_list, f)
        op = 'n'
    f.close()


print("resampling difficult trajs in the pickle file")
write_file()
