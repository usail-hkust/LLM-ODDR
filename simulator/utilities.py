import numpy as np
from copy import deepcopy
import random
import pandas as pd

from math import radians, degrees, cos, sin, asin, sqrt, atan2
import time
import pickle


def get_sum_for_nodes(time, node_list, df, type):
    df.columns = [col.replace('Time', 'time').replace('Location_', '') for col in df.columns]
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    # Define the start time window
    if type == 'before':
        end_time = pd.to_datetime(time)
        start_time = end_time - pd.Timedelta(minutes=15)
    elif type == 'after':
        start_time = pd.to_datetime(time)
        end_time = start_time + pd.Timedelta(minutes=15)

    # Filter the dataframe within the 15-minute window
    df_window = df.loc[(df.index >= pd.to_datetime(start_time)) & (df.index <= pd.to_datetime(end_time))]

    # Calculate sum for each node in the list
    node_list_str = []
    for node in node_list:
        node_list_str.append(str(node))
    node_sums = df_window[node_list_str].sum()

    return node_sums


def refine_args(args):
    if args.dataset == 'small-100':
        args.request_file_name = 'all_requests_0.1'
        args.driver_file_name = 'df_driver_info_100'
        args.order_number_origin_name = 'all_requests_0.1_origin_order_num'
    elif args.dataset == 'small-200':
        args.request_file_name = 'all_requests_0.1'
        args.driver_file_name = 'df_driver_info_200'
        args.order_number_origin_name = 'all_requests_0.1_origin_order_num'
    elif args.dataset == 'large-500':
        args.request_file_name = 'all_requests_0.1'
        args.driver_file_name = 'df_driver_info_500'
        args.order_number_origin_name = 'all_requests_0.1_origin_order_num'


    if "AWQ" in args.model_name:
        args.model_path = 'xxx'
    else:
        args.model_path = 'xxx'
 
    if args.description is None:
        random_number = random.randint(1, 10000)
        args.description = random_number

    args.save_name = args.model_name+'_'+str(args.description)

    # zone related parameters
    ZONE_ID_LIST = []
    for i in range(263):
        ZONE_ID_LIST.append(i)
    args.ZONE_ID_LIST = ZONE_ID_LIST

    return args


def distance(coord_1, coord_2):
    lon1, lat1 = coord_1
    lon2, lat2 = coord_2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)
    r = 6371

    alat = sin(dlat / 2) ** 2
    clat = 2 * atan2(alat ** 0.5, (1 - alat) ** 0.5)
    lat_dis = clat * r * 1000

    alon = sin(dlon / 2) ** 2
    clon = 2 * atan2(alon ** 0.5, (1 - alon) ** 0.5)
    lon_dis = clon * r * 1000

    manhattan_dis = abs(lat_dis) + abs(lon_dis)

    return manhattan_dis


def distance_array(coord_1, coord_2):
    coord_1 = coord_1.astype(float)
    coord_2 = coord_2.astype(float)
    coord_1_array = np.radians(coord_1)
    coord_2_array = np.radians(coord_2)
    dlon = np.abs(coord_2_array[:, 0] - coord_1_array[:, 0])
    dlat = np.abs(coord_2_array[:, 1] - coord_1_array[:, 1])
    r = 6371

    alat = np.sin(dlat / 2) ** 2
    clat = 2 * np.arctan2(alat ** 0.5, (1 - alat) ** 0.5)
    lat_dis = clat * r * 1000

    alon = np.sin(dlon / 2) ** 2
    clon = 2 * np.arctan2(alon ** 0.5, (1 - alon) ** 0.5)
    lon_dis = clon * r * 1000

    manhattan_dis = np.abs(lat_dis) + np.abs(lon_dis)

    return manhattan_dis


def distance_array_line(coord_1, coord_2):
    coord_1 = coord_1.astype(float)
    coord_2 = coord_2.astype(float)
    coord_1_array = np.radians(coord_1)
    coord_2_array = np.radians(coord_2)
    dlon = coord_2_array[:, 0] - coord_1_array[:, 0]
    dlat = coord_2_array[:, 1] - coord_1_array[:, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(coord_1_array[:, 1]) * np.cos(coord_2_array[:, 1]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(a ** 0.5)
    r = 6371
    distance = c * r * 1000
    return distance


class GridSystem:
    def __init__(self, **kwargs):
        # read parameters
        self.range_lng = kwargs.pop('range_lng', 0)
        self.range_lat = kwargs.pop('range_lat', 0)

    def load_data(self, ):
        # self.df_neighbor_centroid = pickle.load(open('data/' + 'zone_info.pickle', 'rb'))
        self.df_neighbor_centroid = pd.read_csv('data/hex_updated_r300_info.csv')
        self.num_grid = self.df_neighbor_centroid.shape[0]
        self.adj_mat = pickle.load(open('data/' + 'adj_matrix_zone.pickle', 'rb'))
        print('adj_matrix_zone loaded.')

    def get_basics(self):
        # output: basic information about the grid network
        return self.num_grid


def sample_orders_drivers(request_distribution, driver_info, request_databases, driver_number_dist):
    # Used to refine the data
    sampled_request_distribution = request_distribution
    sampled_driver_info = driver_info
    sampled_request_databases = request_databases
    sampled_driver_number_dist = driver_number_dist
    return sampled_request_distribution, sampled_driver_info, sampled_request_databases, sampled_driver_number_dist


def sample_all_drivers(driver_info):
    # generate all the drivers in the system
    # add states for vehicle_table based on vehicle columns
    # the driver info include: 'driver_id', 'start_time', 'end_time', 'lng', 'lat'
    # the new driver info include: 'driver_id', 'start_time', 'end_time', 'lng', 'lat', 'status',
    # 'target_lng', 'target_lat', 'remaining_time', 'matched_order_id', 'total_idle_time'
    # t_initial and t_end are used to filter available drivers in the selected time period
    new_driver_info = deepcopy(driver_info)
    sampled_driver_info = new_driver_info

    sampled_driver_info['status'] = 0
    sampled_driver_info['target_loc_lng'] = sampled_driver_info['lng']
    sampled_driver_info['target_loc_lat'] = sampled_driver_info['lat']
    sampled_driver_info['target_grid_id'] = sampled_driver_info['grid_id']
    sampled_driver_info['remaining_time'] = 0
    sampled_driver_info['matched_order_id'] = 'None'
    sampled_driver_info['total_idle_time'] = 0

    return sampled_driver_info


def KM_simulation(wait_requests, driver_table, model_name, pick_up_dis_threshold=950):
    # currently, we use the dispatch alg of peibo
    idle_driver_table = driver_table[driver_table['status'] == 0]
    num_wait_request = wait_requests.shape[0]
    num_idle_driver = idle_driver_table.shape[0]

    if num_wait_request > 0 and num_idle_driver > 0:
        request_array = wait_requests.loc[:, ['origin_lng', 'origin_lat', 'order_id', 'weight']].values
        request_array = np.repeat(request_array, num_idle_driver, axis=0)
        driver_loc_array = idle_driver_table.loc[:, ['lng', 'lat', 'driver_id']].values
        driver_loc_array = np.tile(driver_loc_array, (num_wait_request, 1))
        assert driver_loc_array.shape[0] == request_array.shape[0]
        dis_array = distance_array(request_array[:, :2], driver_loc_array[:, :2])
        # print('negative: ', np.where(dis_array)<0)

        flag = np.where(dis_array <= pick_up_dis_threshold)[0]
        if 'pickup_distance' in model_name:
            order_driver_pair = np.vstack(
                [request_array[flag, 2], driver_loc_array[flag, 2], pick_up_dis_threshold + 1 - dis_array[flag],
                 dis_array[flag]]).T
        else:
            order_driver_pair = np.vstack(
                [request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3], dis_array[flag]]).T
        order_driver_pair = order_driver_pair.tolist()

        if len(order_driver_pair) > 0:
            # matched_pair_actual_indexes = km.run_kuhn_munkres(order_driver_pair)
            matched_pair_actual_indexes = dispatch_alg_array(order_driver_pair)
        else:
            matched_pair_actual_indexes = []
    else:
        matched_pair_actual_indexes = []

    return matched_pair_actual_indexes
