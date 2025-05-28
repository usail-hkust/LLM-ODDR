import pandas as pd
import numpy as np


def one_hour_ago_sum(id, data_pd, time):
    data_pd.columns = [col.replace('Location_', '').replace('Time', 'time') for col in data_pd.columns]
    data_pd['time'] = pd.to_datetime(data_pd['time'])  # 将time列转换为datetime类型
    time = pd.to_datetime(time)
    one_hour_data = data_pd[f'{id}'][(data_pd['time'] >= time - pd.Timedelta(minutes=60)) & (data_pd['time'] <= time)]
    return one_hour_data.sum()


def get_scoring_prompt(dispatch_observ, current_experiment_time):
    # columns_name = ['driver_id', 'order_id', 'trip_time', 'origin_lng', 'origin_lat', 'dest_lng', 'dest_lat',
    #                 'immediate_reward', 'wait_time', 'maximum_wait_time', 'origin_grid_id', 'dest_grid_id',
    #                 'origin_num_1h_ago', 'dest_num_1h_ago']
    dispatch_observ['future_value_of_dest_area'] = dispatch_observ['dest_order_num_1h_ago'] - dispatch_observ['origin_order_num_1h_ago']
    order_data = ("order_id, origin_lng, origin_lat, dest_lng, dest_lat, immediate_reward, wait_time, origin_num_1h_ago, dest_num_1h_ago"
                  "future_value_of_dest_area \n")
    order_text = ''
    for index, order_row in dispatch_observ.iterrows():
        order_id = order_row['order_id']
        origin_lng = order_row['origin_lng']
        origin_lat = order_row['origin_lat']
        dest_lng = order_row['dest_lng']
        dest_lat = order_row['dest_lat']
        immediate_reward = order_row['immediate_reward']
        wait_time = order_row['wait_time']
        origin_num_1h_ago = order_row['origin_order_num_1h_ago']
        dest_num_1h_ago = order_row['dest_order_num_1h_ago']
        future_value_of_dest_area = order_row['future_value_of_dest_area']
        order_text += f'{order_id}, {origin_lng}, {origin_lat}, {dest_lng}, {dest_lat}, {immediate_reward}, {wait_time}, {origin_num_1h_ago}, {dest_num_1h_ago}, {future_value_of_dest_area} \n'
    order_data += order_text + '\n'

    prompt = [
        {"role": "system",
         "content": "You are an expert in ride-hailing system. You can use your transportation knowledge "
                    "and common sense to comprehensively assess the value of each order."},
        {"role": "user",
         "content": "You are an expert in ride-hailing management with extensive knowledge of "
                    "transportation logistics and operational efficiency. Your task is to estimate the Value-Adding "
                    "Factor (ranging between 0 and 1) for incoming orders in a ride-hailing system, balancing multiple "
                    "objectives to ensure customer satisfaction and maximize overall value.\n"
                    "The current time: " + current_experiment_time + "\n"
                    "You will be given data on available orders in the following format.\n"
                    + order_data +
                    
                    "\nPlease analyze each order using the given data, considering the following hints:\n"
                    "- Higher-priced orders with longer trip time and waiting times should have a "
                    "higher value-adding factor.\n"
                    "- future_value_of_dest_area = dest_num_1h_ago - origin_num_1h_ago, "
                    "A higher future_value_of_dest_area indicates that the destination has a potentially higher future "
                    "value, increasing the order’s worth; otherwise, its value may be lower.\n"
                    "- Analyze the starting and destination coordinates along with the given timestamp to assess any "
                    "additional factors that might influence the order’s value "
                    "(e.g., peak hours, location popularity).\n\n"
                    
                    "Please answer:\n"
                    "For each unique order, estimate the Value-Adding Factor (between 0 and 1) that will "
                    "most effectively enhance overall efficiency in subsequent order dispatching.\n\n"
                    
                    "Requirements:\n"
                    "- Let's think step by step.\n"
                    "- Try to understand the meaning of each variable.\n"
                    "- Your estimation can only be given after finishing the analysis.\n"
                    "- Ensure that each order must estimate one Value-Adding Factor and it is between 0 and 1.\n"
                    "- For each order, your estimated Value-Adding Factor must be identified by the tag: "
                    "<increase_factor>ORDER_ID:ESTIMATED_INCREASE_FACTOR</increase_factor>,"
                    "e.g. <increase_factor>8898691:0.273</increase_factor>.\n"
         }
    ]

    return prompt


def get_scoring_prompt_refine(last_scoring_prompt_response, last_review_prompt_response):
    prompt = []
    for a in last_scoring_prompt_response['prompt']:
        prompt.append(a)
    prompt.append(last_scoring_prompt_response['response']['choices'][0]['message'])
    prompt.append(last_review_prompt_response['response']['choices'][0]['message'])

    prompt.append (

    )

    return prompt


def get_review_prompt(last_prompt):
    prompt = []
    prompt.append(last_prompt['prompt'][0])
    prompt.append(last_prompt['response'][0])

    prompt.append(
        {"role": "user",
         "content": "You are an expert in ride-hailing management with extensive knowledge of transportation logistics "
                    "and operational efficiency. Your task is to evaluate the estimated Value-Adding Factors "
                    "(ranging between 0 and 1) provided for incoming orders in a ride-hailing system.\n\n"
                    
                    # "The given information of each order:\n"
                    # + order_data + "\n" +
                    # "The estimated Value-Adding Factors for each order:\n"
                    # + order_data + "\n" +

                    "Your Task:\n"
                    "Evaluate whether the estimated Value-Adding Factors for each order are reasonable based on the "
                    "last round results.\n\n"
                    
                    "Requirements:\n"
                    "- Let's think step by step.\n"
                    "- Your evaluation can only be given after finishing the analysis.\n\n"
                    
                    "If any orders are unreasonable:\n"
                    "- Identify and explain the reasons why the estimations are not reasonable.\n"
                    "- Provide suggestions on how to improve the estimations.\n"
                    "- Provide the identifier: <evaluation_again>\n\n"
                    
                    "If all the orders are reasonable and require no further processing:\n"
                    "- Provide the identifier: <evaluation_complete>\n"
                    "- Only one identifier, either <evaluation_again> or <evaluation_complete>, can be provided.\n"

         }
    )

    return prompt


def get_dispatching_prompt(dispatch_observ_factor, driver_reward_table, current_time):
    driver_reward_table['driver_id'] = driver_reward_table['driver_id'].astype(str)
    # columns_name = ['order_id', 'driver_id', 'order_driver_distance', 'immediate_reward', 'wait_time',
    #                 'maximum_wait_time']
    # dispatch_observ = pd.DataFrame(dic_dispatch_observ, columns=columns_name)

    order_data = ""
    order_list = dispatch_observ_factor['order_id'].unique()
    for order_id in order_list:
        order_text = ''
        immediate_reward_factor = dispatch_observ_factor[dispatch_observ_factor['order_id'] == order_id]['immediate_reward_factor'].values[0]
        wait_time = dispatch_observ_factor[dispatch_observ_factor['order_id'] == order_id]['wait_time'].values[0]
        maximum_wait_time = dispatch_observ_factor[dispatch_observ_factor['order_id'] == order_id]['maximum_wait_time'].values[0]
        order_text += (f'order id: {order_id}, overall immediate reward: {immediate_reward_factor}, the time already waited:{wait_time}, '
                       f'the maximum waiting time: {maximum_wait_time}. \n')
        order_text += f'Available drivers list for order {order_id}:\n'
        order_text += f'driver id, distance from driver to order, num of finished order, current overall reward\n'
        for index, order_row in dispatch_observ_factor[dispatch_observ_factor['order_id'] == order_id].iterrows():
            driver_id = order_row['driver_id']
            order_driver_distance = order_row['order_driver_distance']
            num_finished_order = driver_reward_table[driver_reward_table['driver_id'] == driver_id]['num_finished_order'].values[0]
            current_overall_reward = driver_reward_table[driver_reward_table['driver_id'] == driver_id]['current_overall_reward'].values[0]
            order_text += f'{driver_id}, {order_driver_distance:.4f}, {num_finished_order}, {current_overall_reward} \n'
        order_data += order_text + '\n'

    prompt = [
        {"role": "system",
         "content": "You are an expert in ride-hailing order dispatching management. You can use your knowledge of "
                    "transportation commonsense to solve this ride-hailing order dispatching tasks."},
        {"role": "user",
         "content": "You are an expert in ride-hailing order dispatching management with extensive knowledge of "
                    "transportation logistics and operational efficiency. Your task is to optimize driver assignments "
                    "for incoming orders in a ride-hailing system, balancing multiple objectives to ensure "
                    "customer satisfaction, operational efficiency, and driver fairness.\n"
                    "Use your expertise to analyze the given data and make optimal dispatch decisions."
                    "When making dispatch decisions, consider the following factors in order of priority:\n"
                    "- Maximizing immediate rewards.\n"
                    "- Minimizing customer wait times.\n"
                    "- Maximizing driver utilization.\n"
                    "- Ensuring fairness among drivers.\n"
                    "- Optimizing for long-term system efficiency.\n\n"
                    "The current time: " + current_time + "\n"
                    "You will be given data on available orders and potential driver assignments "
                    "in the following format.\n\n"
                    + order_data +

                    "Please answer:\n"
                    "For each unique order, which is the optimal driver that will most significantly improve "
                    "the order dispatching tasks in the ride-hailing system?\n\n"
                    
                    "Requirements:\n"
                    "- Let's think step by step. Your analysis should be thorough but concise.\n"
                    "- Provide your analysis for understanding the distribution and relationship "
                    "between the orders and drivers.\n"
                    "- You must ensure that each driver can only be assigned to at most one order.\n"
                    "- If you determine that an order should not be assigned to any available driver, "
                    "explain your reasoning and disregard it.\n"
                    "- Answer your chosen driver for each order. Your choice can only be given after finishing "
                    "the above analysis.\n"
                    "- For each assignment, your choice must be identified by the tag: "
                    "<dispatch>ORDER_ID:CHOSEN_DRIVER_ID</dispatch>, e.g. <dispatch>8898691:66</dispatch>."
         }
    ]
    # print(prompt[0]['content'])
    # print(prompt[1]['content'])

    return prompt


def get_repositioning_prompt(dest_grid_info, driver_grid_dict, grid_info_csv, driver_dict, current_time):
    dest_grid_info['GridID'] = dest_grid_info['GridID'].astype(str)
    dest_grid_info['future_value'] = dest_grid_info['order_num_dest_before15m'] - dest_grid_info['order_num_dest_matched_before15m'] - dest_grid_info['driver_num_dest_after15m']
    df_sorted = dest_grid_info.sort_values(by='future_value', ascending=False)

    df_sorted['order_num_dest_before15m'] = df_sorted['order_num_dest_before15m'].replace(0, np.nan)
    df_sorted['order_num_dest_matched_before15m'] = df_sorted['order_num_dest_matched_before15m'].replace(0, np.nan)
    df_sorted['driver_num_dest_after15m'] = df_sorted['driver_num_dest_after15m'].replace(0, np.nan)

    grid_info_csv.rename(columns={grid_info_csv.columns[0]: 'hex_id'}, inplace=True)

    order_data = ""
    driver_num = 0
    for driver_id, available_grids in driver_grid_dict.items():
        driver_num += 1
        if driver_num > 100:
            break
        order_text = ''
        origin_center_lon = grid_info_csv.loc[grid_info_csv["hex_id"] == driver_dict[driver_id], "center_lon"].values[0]
        origin_center_lat = grid_info_csv.loc[grid_info_csv["hex_id"] == driver_dict[driver_id], "center_lat"].values[0]
        order_text += f'Driver: {driver_id}, current longitude: {origin_center_lon}, current latitude: {origin_center_lat}\n'
        order_text += f'Available grid for this driver:\n'
        order_text += (f'grid id, longitude, latitude, number of order requests for this grid in the past 15 minutes, '
                       f'number of matched order for this grid in the past 15 minutes, '
                       f'number of incoming driver to this grid in the next 15 minutes\n')
        num_grids = 0
        for index, row in df_sorted.iterrows():
            grid_id = int(row['GridID'])
            if grid_id in available_grids:
                num_grids += 1
                dest_center_lon = grid_info_csv.loc[grid_info_csv["hex_id"] == grid_id, "center_lon"].values[0]
                dest_center_lat = grid_info_csv.loc[grid_info_csv["hex_id"] == grid_id, "center_lat"].values[0]
                requests_num_before5m = row['order_num_dest_before15m']
                matched_requests_num_before5m = row['order_num_dest_matched_before15m']
                driver_num_dest_after15m = row['driver_num_dest_after15m']
                order_text += (f'{grid_id}, {dest_center_lon}, {dest_center_lat}, {requests_num_before5m}, '
                               f'{matched_requests_num_before5m}, {driver_num_dest_after15m} \n')
            if num_grids == 5:
                break
        order_data += order_text + '\n'


    prompt = [
        {"role": "system",
         "content": "You are an expert in ride-hailing vehicle repositioning management. You can use your knowledge of "
                    "transportation commonsense to solve this ride-hailing vehicle repositioning tasks."},
        {"role": "user",
         "content": "You are an expert in ride-hailing order dispatching management with extensive knowledge of "
                    "transportation logistics and operational efficiency. Your task is to optimize vehicle "
                    "repositioning strategies in a ride-hailing system, considering spatio-temporal dependencies "
                    "to ensure customer satisfaction, operational efficiency, and driver fairness.\n"
                    "The current time: " + current_time + "\n"
                    "You will be given data on current vehicle locations, potential repositioning areas, "
                    "and predicted future order in the areas in the following format:\n\n"
                    + order_data +

                    "Please answer:\n"
                    "For each driver in the ride-hailing system, which is the optimal grid it should be "
                    "repositioned to?\n\n"
                    "considering spatio-temporal dependencies, that will most significantly improve "
                    "the driver repositioning tasks in the ride-hailing system.\n\n"
                    "Requirements:\n"
                    "- Let's think step by step. Your analysis should be thorough but concise.\n"
                    "- You must follow the following steps to provide your analysis:\n"
                    "Step 1: Analyze the spatio-temporal distribution and relationship between the vehicles, "
                    "given grids, and traffic or regional factors over time.\n"
                    "Step 2: Specify your chosen repositioning area for each driver.\n"
                    "- Your choice can only be given after finishing the analysis.\n"
                    "- You must ensure that each driver is assigned to only one repositioning grid.\n"
                    "- If you determine that a vehicle should not be repositioned, "
                    "explain your reasoning and disregard it.\n"
                    "- For each assignment, your choice must be identified by the tag: "
                    "<reposition>DRIVER_ID:CHOSEN_GRID_ID</reposition>."
         }
    ]
    # print(prompt[0]['content'])
    # print(prompt[1]['content'])

    return prompt


def get_three_hop_neighbors(adj_matrix, nodes, driver_grid_id_dict):
    # 确保输入列表唯一
    nodes = list(set(nodes))
    data = {}

    for node in nodes:
        hop_neighbors = set()

        # 第一步：找到一跳邻居
        first_neighbors = np.nonzero(adj_matrix[node])[0]
        hop_neighbors.update(first_neighbors)

        # 第二步：找到二跳邻居
        second_hop_neighbors = set()
        for first_neighbor in first_neighbors:
            second_neighbors = np.nonzero(adj_matrix[first_neighbor])[0]
            second_hop_neighbors.update(second_neighbors)
        hop_neighbors.update(second_hop_neighbors)

        # 第三步：找到三跳邻居
        third_hop_neighbors = set()
        for second_neighbor in second_hop_neighbors:
            third_neighbors = np.nonzero(adj_matrix[second_neighbor])[0]
            third_hop_neighbors.update(third_neighbors)
        hop_neighbors.update(third_hop_neighbors)

        data[node] = hop_neighbors
    new_data = {}
    for key, value in data.items():
        new_keys = driver_grid_id_dict[key]
        for new_key in new_keys:
            new_data[new_key] = value

    return new_data
