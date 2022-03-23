import pickle
import numpy as np
import collections


x = np.load('train_crs_dist6_angle90.p', allow_pickle=True)


def get_intersection_task(data_sample):
    selected_node = []
    dis_matrix = []
    
    lane_idcs = data_sample['graph']['lane_idcs']
    graph_intersection = data_sample['graph']['intersect']

    masked_node_arr = np.array([])
    repeat_count = []

    intersection_nodes = []
    non_intersection_nodes = []
    unique_lanes = np.unique(lane_idcs).shape[0]
    lane_ids = lane_idcs

    for j in range(unique_lanes):
        lane_node_corresp = np.where(lane_ids == j)[0]
        start_lane = lane_node_corresp[0]
        end_lane = lane_node_corresp[-1]
        mask_len_lane = int(len(lane_node_corresp) * 0.4)

        if graph_intersection[start_lane] == 1:
            intersection_nodes.append(j)
        else:
            non_intersection_nodes.append(j)
            if mask_len_lane >= 2:
                masked_node_arr = np.concatenate([masked_node_arr, np.random.randint(start_lane, end_lane, mask_len_lane)])
                repeat_count.append(mask_len_lane)
            else:
                masked_node_arr = np.concatenate([masked_node_arr, [start_lane]])
                repeat_count.append(1)

    a = data_sample['graph']['suc_pairs']
    b = data_sample['graph']['left_pairs']
    c = data_sample['graph']['right_pairs']

    if c.ndim == 2:
        adj_pairs = np.vstack((a, b, c))
    else:
        adj_pairs = np.vstack((a, b))

    if len(intersection_nodes) > 0:
        selected_node = masked_node_arr
        dis_matrix = shortest_path_to_nodes(intersection_nodes, non_intersection_nodes, adj_pairs)
        dis_matrix = np.repeat(dis_matrix, repeat_count)

        perm = np.random.permutation(selected_node.shape[0])
        selected_node = np.array(selected_node)[perm]
        dis_matrix = dis_matrix[perm]
    return selected_node, dis_matrix
    
        
def transform_depth(depth):
        return depth

    
def shortest_path_to_nodes(graph_intersect, graph_unlabel, adj_pairs):
    """
    Do BFS from each labeled node, then we can get the shortest path length
    from other nodes to each labeled node
    """
    dis_matrix_ind = 1000 * np.ones([len(graph_intersect) + len(graph_unlabel), len(graph_intersect)])
    for ii, node_st in enumerate(graph_intersect):
        # do bfs
        visited = set([node_st])
        neighbors = adj_pairs[:, 1][np.where(adj_pairs[:, 0] == node_st)]
        q = collections.deque([(x, 1) for x in neighbors])
        while q:
            node_cur, depth = q.popleft()
            if node_cur in visited:
                continue
            visited.add(node_cur)
            dis_matrix_ind[node_cur][ii] = transform_depth(depth)
            neighbors = adj_pairs[:, 1][np.where(adj_pairs[:, 0] == node_cur)]
            for node_next in neighbors:
                q.append((node_next, depth + 1))
    dis_matrix_ind = np.min(dis_matrix_ind, 1)
    dis_matrix_ind[graph_intersect] = 0
    return dis_matrix_ind[graph_unlabel]
    
def write_file():
 
    f = open("train.p", "wb")
    op = 'y'
 
    while op == 'y':
        for i in range(len(x)):
            if i % 1000 == 0:
                print(i)
            data_sample = x[i]
            selected_nodes, dis_matrix = get_intersection_task(data_sample)
            x[i]['selected_nodes'] = selected_nodes
            x[i]['dis_matrix'] = dis_matrix
                
        pickle.dump(x, f)
        op = 'n'
 
    f.close()
 
print("entering the details of trajs in the pickle file")
write_file()
