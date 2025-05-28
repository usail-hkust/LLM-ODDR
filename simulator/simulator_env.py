import os
import json
import pandas as pd
import math
import re
from simulator.utilities import *
import warnings
from simulator.agent import *
from simulator.prompts import *
from datetime import timedelta, datetime


warnings.filterwarnings("ignore")


class Simulator:
    def __init__(self, args):
        # basic parameters: time & sample
        self.start_date = args.start_date
        self.end_date = args.end_date
        self.t_initial = args.start_time
        self.t_end = args.end_time
        self.delta_t = args.delta_t
        self.vehicle_speed = args.vehicle_speed
        # self.pre_run_time = kwargs['pre_run_time']
        # self.finish_run_time = kwargs['finish_run_time']
        self.experiment_date = None

        self.model_name = args.model_name
        self.pickup_dis_threshold = args.pickup_dis_threshold

        # wait cancel
        self.maximum_wait_time_mean = args.maximum_wait_time_mean
        self.maximum_wait_time_std = args.maximum_wait_time_std

        # pattern
        # experiment mode: train, test
        # experiment date: the date for experiment
        self.request_interval = args.request_interval
        self.request_file_name = args.request_file_name
        self.driver_file_name = args.driver_file_name

        self.request_all = pickle.load(open(f'data/{args.dataset}/{self.request_file_name}.pickle', 'rb'))
        self.driver_info = pickle.load(open(f'data/{args.dataset}/{self.driver_file_name}.pickle', 'rb'))
        self.order_num_origin = pd.read_csv(f'data/{args.dataset}/{args.order_number_origin_name}.csv')
        # self.order_num_dest = pd.read_csv(f'data/{args.dataset}/{args.order_number_dest_name}.csv')
        self.grid_info_csv = pd.read_csv(f'data/{args.dataset}/hex_updated_r300_info.csv')
        self.grid_info_csv.rename(columns={self.grid_info_csv.columns[0]: 'hex_id'}, inplace=True)

        self.hex_adj_csv = pd.read_csv(f'data/{args.dataset}/hexo_updated_r300_adj.csv')
        self.hex_adj = self.hex_adj_csv.to_numpy()
        print('Request and Driver information loaded.')

        self.max_idle_time = args.max_idle_time

        # get steps
        # self.one_ep_steps = int((self.t_end - self.t_initial) // self.delta_t) + 1
        # self.pre_run_step = int((self.pre_run_time - self.t_initial) // self.delta_t)
        # self.finish_run_step = int((self.finish_run_time - self.t_initial) // self.delta_t)
        # self.terminate_run_step = int((self.t_end - self.t_initial) // self.delta_t)

        self.finish_run_step = int((self.t_end) // self.delta_t)

        self.request_databases = None
        self.driver_reward_table = None
        # time/day
        self.time = None
        self.current_step = None
        # vehicle/driver table update status
        self.finished_driver_id = None

        # request tables
        # 暂时不考虑cancel prob，均设置为0
        self.request_columns = ['order_id', 'trip_time', 'origin_lng', 'origin_lat', 'dest_lng', 'dest_lat',
                                'immediate_reward', 'designed_reward', 'origin_grid_id', 'dest_grid_id',
                                't_start', 't_matched', 'pickup_time', 'wait_time', 't_end', 'status', 'driver_id',
                                'maximum_wait_time', 'cancel_prob', 'pickup_distance', 'weight', 'origin_order_num_1h_ago', 'dest_order_num_1h_ago']

        self.wait_requests = None
        self.matched_requests = None
        self.requests_final_records = None
        self.num_all_requests = None  # number of all requests in the experiments date

        # driver tables
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time']
        self.driver_table = None
        self.matched_requests_arrive_num = None
        self.matched_requests_origin = None

        # order and driver databases
        self.sampled_driver_info = self.driver_info
        self.sampled_driver_info['grid_id'] = self.sampled_driver_info['grid_id'].values.astype(int)

        # setup agent
        # 定义模型类型到代理类的映射
        AGENT_MAPPING = {
            'gpt4-api': GPT4Api,
            'together-api': TogetherApi,
            'llama3': LlamaAgent,  # 假设 AgentLlama 为统一代理类名称
            'qwen2.5': QwenAgent,  # 假设 AgentLlama 为统一代理类名称
        }

        # 根据模型类型动态初始化代理
        agent_class = AGENT_MAPPING.get(args.model_type)

        if agent_class:
            self.Agent = agent_class(args.model_path, args.model_name)
            # self.scoring_agent = agent_class(args.model_name)
            # self.review_agent = agent_class(args.model_name)
            # self.dispatching_agent = agent_class(args.model_name)
            # self.repositioning_agent = agent_class(args.model_name)
        else:
            raise ValueError(f"Unsupported model type or name: {args.model_type or args.model_name}")

    def initial_base_tables(self):
        # demand $ supply patterns
        self.request_databases = deepcopy(self.request_all[self.experiment_date])
        self.driver_table = sample_all_drivers(self.sampled_driver_info)
        self.driver_table['target_grid_id'] = self.driver_table['target_grid_id'].values.astype(int)

        self.num_all_requests = 0

        # driver reward table
        self.driver_reward_table = pd.DataFrame(columns=['driver_id', 'num_finished_order', 'current_overall_reward'])
        self.driver_reward_table['driver_id'] = self.driver_table['driver_id']
        self.driver_reward_table['num_finished_order'] = 0
        self.driver_reward_table['current_overall_reward'] = 0

        # order list update status
        self.wait_requests = pd.DataFrame(columns=self.request_columns)  # state: (wait 0, matched 1, finished 2)
        self.matched_requests = pd.DataFrame(columns=self.request_columns)  # state: (wait 0, matched 1, finished 2)

        time_index = pd.date_range(start=self.start_date, end=self.end_date, freq='T')
        # 创建一个包含263列的数据框，列名为 "Location_1" 到 "Location_263"
        columns = ['time'] + [f'{i}' for i in range(263)]
        # 初始化数据为0，并创建DataFrame
        zero_data = np.zeros((len(time_index), 263))
        self.matched_requests_arrive_num = pd.DataFrame(zero_data, columns=columns[1:])
        self.matched_requests_arrive_num.insert(0, 'time', time_index)

        self.matched_requests_origin = deepcopy(self.matched_requests_arrive_num)
        # self.requests_final_records = []

        # time/day
        self.time = deepcopy(self.t_initial)
        self.current_step = int((self.time) // self.delta_t)

        # vehicle/driver table update status
        self.finished_driver_id = []

    def reset(self):
        """
        reset env
        :return:
        """
        # state_repo is a list of states of repositioned drivers
        self.initial_base_tables()

    def step(self, args):
        # Step 1: bipartite matching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)
        if len(wait_requests) > 0:
            wait_requests['origin_order_num_1h_ago'] = wait_requests.apply(
                lambda row: one_hour_ago_sum(row['origin_grid_id'], self.order_num_origin, self.curent_experiment_time), axis=1)
            wait_requests['dest_order_num_1h_ago'] = wait_requests.apply(
                lambda row: one_hour_ago_sum(row['dest_grid_id'], self.order_num_origin, self.curent_experiment_time), axis=1)

        order_driver_pair = self.get_order_driver_pair(wait_requests, driver_table, self.pickup_dis_threshold)
        if len(order_driver_pair) > 0:
            # matched_pair_actual_indexes = []
            matched_pair_actual_indexes = self.openllm_inference(args, order_driver_pair, self.driver_reward_table)
        else:
            matched_pair_actual_indexes = []

        # step 2
        # removed matched requests from wait list, and put them into matched list
        # delete orders with too much waiting time
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes)
        self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
        self.matched_requests = self.matched_requests.reset_index(drop=True)
        self.wait_requests = df_update_wait_requests.reset_index(drop=True)

        # update driver reward table and matched requests arrive num
        if len(df_new_matched_requests) > 0:
            for index, matched_pair in df_new_matched_requests.iterrows():
                driver_id = matched_pair['driver_id']
                immediate_reward = float(matched_pair['immediate_reward'])
                self.driver_reward_table.loc[
                    self.driver_reward_table['driver_id'] == driver_id, 'current_overall_reward'] += immediate_reward
                self.driver_reward_table.loc[
                    self.driver_reward_table['driver_id'] == driver_id, 'num_finished_order'] += 1

                # matched requested arrive num
                arrive_time = matched_pair['trip_time'] + matched_pair['pickup_time']
                future_date = datetime.strptime(self.curent_experiment_time, "%Y-%m-%d %H:%M:%S") + timedelta(
                    minutes=int(arrive_time / 60))
                future_date_formatted = future_date.strftime("%Y-%m-%d %H:%M:%S")
                self.matched_requests_arrive_num.loc[self.matched_requests_arrive_num['time'] == pd.Timestamp(future_date_formatted), f'{matched_pair["dest_grid_id"]}'] += 1

                # matched requested arrive num
                self.matched_requests_origin.loc[self.matched_requests_origin['time'] == self.curent_experiment_time, f'{matched_pair["origin_grid_id"]}'] += 1


        # Step 3: generate new orders
        self.step_bootstrap_new_orders()

        # Step 4: update next state
        self.update_state()
        self.update_time()

    def get_order_driver_pair(self, wait_requests, driver_table, pick_up_dis_threshold):
        idle_driver_table = driver_table[driver_table['status'] == 0]
        num_wait_request = wait_requests.shape[0]
        num_idle_driver = idle_driver_table.shape[0]

        if num_wait_request > 0 and num_idle_driver > 0:
            request_array = wait_requests.loc[:, ['order_id', 'trip_time', 'origin_lng', 'origin_lat', 'dest_lng', 'dest_lat',
                                                  'immediate_reward', 'wait_time', 'maximum_wait_time',
                                                  'origin_grid_id', 'dest_grid_id', 'origin_order_num_1h_ago', 'dest_order_num_1h_ago']].values
            request_array = np.repeat(request_array, num_idle_driver, axis=0)

            driver_loc_array = idle_driver_table.loc[:, ['lng', 'lat', 'driver_id']].values
            driver_loc_array = np.tile(driver_loc_array, (num_wait_request, 1))
            assert driver_loc_array.shape[0] == request_array.shape[0]
            dis_array = distance_array(request_array[:, 2:4], driver_loc_array[:, :2])
            # print('negative: ', np.where(dis_array)<0)
            request_array = np.column_stack((request_array, dis_array))

            flag = np.where(dis_array <= pick_up_dis_threshold)[0]
            order_driver_pair = np.hstack((driver_loc_array[flag, 2].reshape(-1, 1), request_array[flag, 0:14]))
            order_driver_pair = order_driver_pair.tolist()
        else:
            order_driver_pair = []

        return order_driver_pair

    def openllm_inference(self, args, order_driver_pair, driver_reward_table):
        # input
        # order_driver_pair selection
        # [['1301922', '9', 1.0, 934.8200679453382], ['1301922', '10', 1.0, 934.8200679453382],
        #  ['1301922', '11', 1.0, 934.8200679453382], ['1301922', '12', 1.0, 934.8200679453382],
        #  ['1301911', '39', 1.0, 334.9648392559138], ['1301911', '40', 1.0, 334.9648392559138],
        #  ['1301911', '72', 1.0, 741.9409594090995]]

        columns_name = ['driver_id', 'order_id', 'trip_time', 'origin_lng', 'origin_lat', 'dest_lng', 'dest_lat',
                        'immediate_reward', 'wait_time', 'maximum_wait_time', 'origin_grid_id', 'dest_grid_id',
                        'origin_order_num_1h_ago', 'dest_order_num_1h_ago', 'order_driver_distance']
        dispatch_observ = pd.DataFrame(order_driver_pair, columns=columns_name)
        dispatch_observ['order_id'] = dispatch_observ['order_id'].astype(int).astype(str)
        dispatch_observ['driver_id'] = dispatch_observ['driver_id'].astype(int).astype(str)
        # dispatch_observ['reward_units'] = 0. + dispatch_observ['reward_units'].values
        # dic_dispatch_observ = dispatch_observ.to_dict(orient='records')

        dispatch_observ_factor = self.agent_scoring_review(dispatch_observ)

        if args.dataset in ['medium', 'medium-1000', 'large']:
            dispatch_observ_factor = dispatch_observ_factor.groupby('order_id', group_keys=False).apply(lambda x: x.nsmallest(10, 'order_driver_distance'))

        dispatch_action = []
        dispatching_prompt = get_dispatching_prompt(dispatch_observ_factor, driver_reward_table, self.curent_experiment_time)

        dispatching_response = self.Agent.get_completions(dispatching_prompt)

        if self.curent_experiment_time not in self.Agent.dispatch_memory:
            self.Agent.dispatch_memory[self.curent_experiment_time] = []
        dispatching_prompt_response = {}
        dispatching_prompt_response['prompt'] = dispatching_prompt
        dispatching_prompt_response['response'] = [{'content': dispatching_response, 'role': 'system'}]
        self.Agent.dispatch_memory[self.curent_experiment_time].append(dispatching_prompt_response)

        dispatches = re.findall(r'<dispatch>(.*?)</dispatch>', dispatching_response)

        random.shuffle(dispatches)  # 首先随机排序 data
        seen_driver_ids = set()  # 用于存储已经出现过的 driver_id
        unique_dispatches = []  # 用于存储处理后的数据
        # 遍历原始数据
        for item in dispatches:
            order_id, driver_id = item.split(":")  # 分割每一项以获取 order_id 和 driver_id
            if driver_id not in seen_driver_ids:  # 检查 driver_id 是否已在集合中
                unique_dispatches.append(item)  # 如果不在集合中，则添加到结果列表中
                seen_driver_ids.add(driver_id)  # 并将 driver_id 添加到集合中
        # print(unique_dispatches)

        for dispatch in unique_dispatches:
            order_id, driver_id = dispatch.split(':')
            try:
                distance_value = \
                    dispatch_observ[
                        (dispatch_observ['order_id'] == order_id) & (dispatch_observ['driver_id'] == driver_id)][
                        'order_driver_distance'].iloc[0]
                dispatch_action.append([order_id, driver_id, 1.0, distance_value])
            except:
                print(f'value error: {order_id}, {driver_id}')

        # output final selection
        # [['1301922', '9', 1.0, 934.8200679453382], ['1301911', '37', 1.0, 334.9648392559138]]
        return dispatch_action

    def agent_scoring_review(self, dispatch_observ, max_time=2):
        dispatch_observ_selected = deepcopy(dispatch_observ)
        dispatch_observ_selected.drop_duplicates(subset=['order_id'], keep='first', inplace=True)

        if self.curent_experiment_time not in self.Agent.score_memory:
            self.Agent.score_memory[self.curent_experiment_time] = []
        if self.curent_experiment_time not in self.Agent.review_memory:
            self.Agent.review_memory[self.curent_experiment_time] = []

        n = 1
        score_refine = False
        while n <= max_time:
            scoring_prompt_response = {}

            if score_refine:
                scoring_prompt = scoring_prompt_refine
            else:
                scoring_prompt = get_scoring_prompt(dispatch_observ_selected, self.curent_experiment_time)
            # print(scoring_prompt)
            scoring_agent_output = self.Agent.get_completions(scoring_prompt)

            scoring_prompt_response['prompt'] = scoring_prompt
            scoring_prompt_response['response'] = [{'content': scoring_agent_output, 'role': 'system'}]
            self.Agent.score_memory[self.curent_experiment_time].append(scoring_prompt_response)
            n += 1
            if not score_refine:
                review_prompt_response = {}
                review_prompt = get_review_prompt(self.Agent.score_memory[self.curent_experiment_time][-1])
                # print(review_prompt)
                review_agent_output = self.Agent.get_completions(review_prompt)

                review_prompt_response['prompt'] = review_prompt
                review_prompt_response['response'] = [{'content': review_agent_output, 'role': 'system'}]
                self.Agent.review_memory[self.curent_experiment_time].append(review_prompt_response)

                if '<evaluation_again>' in review_agent_output:
                    scoring_prompt_refine = scoring_prompt
                    scoring_prompt_refine.append({'content': scoring_agent_output, 'role': 'system'})
                    scoring_prompt_refine.append(review_prompt_response['response'][0])
                    review_text = {"role": "user",
                                   "content": "The previous estimations of the Value-Adding Factors seem unreasonable. "
                                              "Please re-estimate the Value-Adding Factors for each unique order, "
                                              "considering the above requirements and feedback."
                                   }
                    scoring_prompt_refine.append(review_text)
                    score_refine = True

        order_factor_pair = re.findall(r'<increase_factor>(.*?)</increase_factor>', scoring_agent_output)
        dispatch_observ['value_adding_factor'] = 0
        dispatch_observ['immediate_reward_factor'] = 0
        factor_dict = {}
        for item in order_factor_pair:
            try:
                order_id, factor = item.split(":")  # 分割每一项以获取 order_id 和 driver_id
                factor_dict[order_id] = float(factor)
            except:
                pass

        for index, row in dispatch_observ.iterrows():
            order_id = row['order_id']
            if order_id in factor_dict:
                dispatch_observ.loc[index, 'value_adding_factor'] = factor_dict[order_id]

        dispatch_observ['immediate_reward_factor'] = (dispatch_observ['value_adding_factor'] + 1) * dispatch_observ['immediate_reward']

        return dispatch_observ

    def update_info_after_matching_multi_process(self, matched_pair_actual_indexes):
        new_matched_requests = pd.DataFrame([], columns=self.request_columns)
        update_wait_requests = pd.DataFrame([], columns=self.request_columns)

        matched_pair_index_df = pd.DataFrame(matched_pair_actual_indexes,
                                             columns=['order_id', 'driver_id', 'weight', 'pickup_distance'])
        matched_order_id_list = matched_pair_index_df['order_id'].values.tolist()
        con_matched = self.wait_requests['order_id'].isin(matched_order_id_list)
        con_keep_wait = self.wait_requests['wait_time'] <= self.wait_requests['maximum_wait_time']

        # when the order is matched
        df_matched = self.wait_requests[con_matched].reset_index(drop=True)
        if df_matched.shape[0] > 0:
            idle_driver_table = self.driver_table[self.driver_table['status'] == 0]
            order_array = df_matched['order_id'].values

            # multi process if necessary
            cor_order = []
            cor_driver = []
            for i in range(len(matched_pair_index_df)):
                cor_order.append(np.argwhere(order_array == matched_pair_index_df['order_id'][i])[0][0])
                cor_driver.append(
                    idle_driver_table[idle_driver_table['driver_id'] == matched_pair_index_df['driver_id'][i]].index[0])
            cor_driver = np.array(cor_driver)
            df_matched = df_matched.iloc[cor_order, :]

            # decide whether cancelled
            # currently no cancellation
            cancel_prob = np.zeros(len(matched_pair_index_df))
            prob_array = np.random.rand(len(cancel_prob))
            con_remain = prob_array >= cancel_prob

            # order after cancelled
            update_wait_requests = df_matched[~con_remain]

            # driver after cancelled
            self.driver_table.loc[cor_driver[~con_remain], ['status', 'remaining_time', 'total_idle_time']] = 0

            # order not cancelled
            new_matched_requests = df_matched[con_remain]
            new_matched_requests['t_matched'] = self.time
            new_matched_requests['pickup_distance'] = matched_pair_index_df[con_remain]['pickup_distance'].values
            new_matched_requests['pickup_time'] = new_matched_requests['pickup_distance'].values / self.vehicle_speed
            new_matched_requests['t_end'] = self.time + new_matched_requests['pickup_time'].values + \
                                            new_matched_requests['trip_time'].values
            new_matched_requests['status'] = 1
            new_matched_requests['driver_id'] = matched_pair_index_df[con_remain]['driver_id'].values

            # driver not cancelled
            self.driver_table.loc[cor_driver[con_remain], 'status'] = 1
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lng'] = new_matched_requests['dest_lng'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lat'] = new_matched_requests['dest_lat'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_grid_id'] = new_matched_requests[
                'dest_grid_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time'] = new_matched_requests['t_end'].values \
                                                                              - new_matched_requests['t_matched'].values
            self.driver_table.loc[cor_driver[con_remain], 'matched_order_id'] = new_matched_requests['order_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'total_idle_time'] = 0

            # self.requests_final_records += new_matched_requests.values.tolist()

        # when the order is not matched
        update_wait_requests = pd.concat([update_wait_requests, self.wait_requests[~con_matched & con_keep_wait]],
                                         axis=0)
        # self.requests_final_records += self.wait_requests[~con_matched & ~con_keep_wait].values.tolist()

        return new_matched_requests, update_wait_requests

    def step_bootstrap_new_orders(self):
        # generate new orders

        count_interval = int(math.floor(self.time / self.request_interval))
        self.request_database = self.request_databases[str(count_interval * self.request_interval)]
        weight_array = np.ones(len(self.request_database))

        self.num_all_requests += len(self.request_database)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        if len(self.request_database) > 0:
            order_id = [request[0] for request in self.request_database]
            requests = np.array([request[3:] for request in self.request_database])
            new_requests = np.delete(requests, -3, axis=1)
            column_name = ['origin_lng', 'origin_lat', 'dest_lng', 'dest_lat', 'immediate_reward', 'trip_distance',
                           'trip_time', 'designed_reward', 'origin_grid_id', 'dest_grid_id']
        else:
            requests = []

        if len(requests) > 0:
            wait_info = pd.DataFrame(new_requests, columns=column_name)
            wait_info['origin_grid_id'] = wait_info['origin_grid_id'].values.astype(int)
            wait_info['dest_grid_id'] = wait_info['dest_grid_id'].values.astype(int)
            wait_info['order_id'] = order_id
            wait_info['t_start'] = self.time
            wait_info['wait_time'] = 0
            wait_info['status'] = 0
            wait_info['maximum_wait_time'] = self.maximum_wait_time_mean
            wait_info['cancel_prob'] = 0
            wait_info['weight'] = weight_array
            wait_info = wait_info.drop(columns=['trip_distance'])
            self.wait_requests = pd.concat([self.wait_requests, wait_info], ignore_index=True)

        return

    def update_state(self):
        # update next state
        # update driver information
        self.driver_table['remaining_time'] = self.driver_table['remaining_time'].values - self.delta_t
        loc_negative_time = self.driver_table['remaining_time'] <= 0
        loc_idle = self.driver_table['status'] == 0
        loc_on_trip = self.driver_table['status'] == 1

        self.driver_table.loc[loc_negative_time, 'remaining_time'] = 0
        self.driver_table.loc[loc_negative_time, 'lng'] = self.driver_table.loc[
            loc_negative_time, 'target_loc_lng'].values
        self.driver_table.loc[loc_negative_time, 'lat'] = self.driver_table.loc[
            loc_negative_time, 'target_loc_lat'].values
        self.driver_table.loc[loc_negative_time, 'grid_id'] = self.driver_table.loc[
            loc_negative_time, 'target_grid_id'].values.astype(int)
        self.driver_table.loc[loc_idle, 'total_idle_time'] += self.delta_t
        self.driver_table.loc[loc_negative_time & loc_on_trip, 'status'] = 0

        # cruising decision
        loc_long_idle = self.driver_table['total_idle_time'] >= self.max_idle_time
        grid_id_array = self.driver_table.loc[loc_long_idle, 'grid_id'].values
        if grid_id_array.shape[0] > 0:
            driver_dict = self.driver_table.loc[loc_long_idle, ['driver_id', 'grid_id']].set_index('driver_id')['grid_id'].to_dict()
            driver_grid_id_dict = self.driver_table.loc[loc_long_idle, ['driver_id', 'grid_id']].groupby('grid_id')[
                'driver_id'].apply(list).to_dict()
            three_hop_neighbors = get_three_hop_neighbors(self.hex_adj, list(grid_id_array), driver_grid_id_dict)
            # three_hop_neighbors = {128: {128, 129, 130, 256, 133, 134, 135, 136, 137, 138, 141, 142, 143, 144, 145, 149, 150, 151, 152, 102, 103, 104, 109, 110, 111, 112, 113, 117, 118, 119, 120, 121, 122, 125, 126, 127}, 166: {142, 143, 144, 145, 149, 150, 151, 152, 153, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 169, 171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 184, 185, 186, 187}, 41: {21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 59, 60, 61}, 74: {54, 55, 56, 59, 60, 61, 62, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 92, 93, 94, 95, 252}, 42: {22, 23, 24, 27, 28, 29, 30, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 53, 54, 55, 56, 59, 60, 61, 62, 251, 252}, 12: {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 19, 20, 21, 25, 26, 31, 32, 236, 237, 238, 239, 249}, 13: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 31, 32, 33, 236, 237, 238, 239, 249, 250}, 143: {128, 129, 130, 133, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 151, 152, 153, 156, 157, 158, 159, 160, 164, 165, 166, 167, 118, 119, 120, 121, 126, 127}, 243: {64, 65, 70, 71, 72, 77, 78, 79, 84, 85, 91, 241, 242, 243, 244, 245, 246, 57, 58, 63}, 116: {131, 132, 133, 134, 139, 140, 141, 92, 93, 94, 95, 98, 99, 100, 101, 102, 105, 106, 107, 108, 109, 110, 114, 115, 116, 117, 118, 119, 123, 124, 125, 126, 127}, 87: {66, 67, 68, 69, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 106, 107, 108, 109}, 88: {67, 68, 69, 73, 74, 75, 76, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 107, 108, 109, 110, 253, 254}, 120: {128, 129, 256, 130, 255, 134, 135, 136, 137, 138, 142, 143, 144, 145, 96, 97, 102, 103, 104, 109, 110, 111, 112, 113, 117, 118, 119, 120, 121, 122, 126, 127}, 152: {128, 129, 130, 135, 136, 137, 138, 142, 143, 144, 145, 146, 149, 150, 151, 152, 153, 154, 157, 158, 159, 160, 161, 162, 165, 166, 167, 168, 169, 173, 174, 175, 176}, 127: {128, 129, 130, 132, 133, 134, 135, 136, 137, 140, 141, 142, 143, 144, 148, 149, 150, 151, 101, 102, 103, 104, 108, 109, 110, 111, 112, 116, 117, 118, 119, 120, 121, 124, 125, 126, 127}}
            merged_set = set()
            for value in three_hop_neighbors.values():
                merged_set.update(value)
            dest_grid_info = pd.DataFrame()
            dest_grid_info['GridID'] = list(merged_set)
            dest_grid_info['order_num_dest_before15m'] = 0
            dest_grid_info['order_num_dest_matched_before15m'] = 0
            dest_grid_info['driver_num_dest_after15m'] = 0

            order_num_origin_before15m_series = get_sum_for_nodes(self.curent_experiment_time, merged_set, self.order_num_origin, 'before')
            for key, value in order_num_origin_before15m_series.items():
                dest_grid_info.loc[dest_grid_info['GridID'] == int(key), 'order_num_dest_before15m'] = value

            order_num_origin_matched_before15m_series = get_sum_for_nodes(self.curent_experiment_time, merged_set, self.matched_requests_origin, 'before')
            for key, value in order_num_origin_matched_before15m_series.items():
                dest_grid_info.loc[dest_grid_info['GridID'] == int(key), 'order_num_dest_matched_before15m'] = value

            order_num_dest_after15m_series = get_sum_for_nodes(self.curent_experiment_time, merged_set, self.matched_requests_arrive_num, 'after')
            for key, value in order_num_dest_after15m_series.items():
                dest_grid_info.loc[dest_grid_info['GridID'] == int(key), 'driver_num_dest_after15m'] = value

            repositioning_prompt_response = {}
            repositioning_prompt = get_repositioning_prompt(dest_grid_info, three_hop_neighbors, self.grid_info_csv, driver_dict, self.curent_experiment_time)
            # print(review_prompt)
            repositioning_agent_output = self.Agent.get_completions(repositioning_prompt)

            repositioning_prompt_response['prompt'] = repositioning_prompt
            repositioning_prompt_response['response'] = [{'content': repositioning_agent_output, 'role': 'system'}]
            if self.curent_experiment_time not in self.Agent.reposition_memory:
                self.Agent.reposition_memory[self.curent_experiment_time] = []
            self.Agent.reposition_memory[self.curent_experiment_time].append(repositioning_prompt_response)

            # reposition_results = re.findall(r'<reposition>(.*?)</reposition>', repositioning_agent_output.json()['choices'][0]['message']['content'])
            reposition_results = re.findall(r'<reposition>(.*?)</reposition>', repositioning_agent_output)

            for item in reposition_results:
                driver_id, grid_id = item.split(":")  # 分割每一项以获取 order_id 和 driver_id
                try:
                    grid_id = int(grid_id)
                    self.driver_table.loc[self.driver_table['driver_id'] == driver_id, 'lng'] = self.grid_info_csv.loc[
                        self.grid_info_csv['hex_id'] == grid_id, 'center_lon'].values

                    self.driver_table.loc[self.driver_table['driver_id'] == driver_id, 'lat'] = self.grid_info_csv.loc[
                        self.grid_info_csv['hex_id'] == grid_id, 'center_lat'].values

                    self.driver_table.loc[self.driver_table['driver_id'] == driver_id, 'grid_id'] = grid_id

                    self.driver_table.loc[self.driver_table['driver_id'] == driver_id, 'target_loc_lng'] = \
                    self.grid_info_csv.loc[
                        self.grid_info_csv['hex_id'] == grid_id, 'center_lon'].values

                    self.driver_table.loc[self.driver_table['driver_id'] == driver_id, 'target_loc_lat'] = \
                    self.grid_info_csv.loc[
                        self.grid_info_csv['hex_id'] == grid_id, 'center_lat'].values

                    self.driver_table.loc[self.driver_table['driver_id'] == driver_id, 'target_grid_id'] = grid_id
                    self.driver_table.loc[self.driver_table['driver_id'] == driver_id, 'total_idle_time'] = 0
                except:
                    print(item)

        finished_driver_id = self.driver_table.loc[loc_negative_time & loc_on_trip, 'driver_id'].values.tolist()
        self.driver_table.loc[loc_negative_time & loc_on_trip, 'matched_order_id'] = 'None'

        # finished order deleted
        # self.finished_driver_id.extend(finished_driver_id)
        # if len(self.finished_driver_id) > 0:
        #     self.matched_requests = self.matched_requests[~self.matched_requests['driver_id'].isin(finished_driver_id)]
        #     self.matched_requests = self.matched_requests.reset_index(drop=True)
        #     self.finished_driver_id = []

        # wait list update
        self.wait_requests['wait_time'] += self.delta_t

        return

    def update_time(self):
        # time counter
        self.time += self.delta_t
        self.current_step += 1

        result_date = datetime.strptime(self.experiment_date, "%Y-%m-%d") + timedelta(minutes=self.current_step)
        self.curent_experiment_time = result_date.strftime("%Y-%m-%d %H:%M:%S")
        return

    def evaluation(self, args):
        save_path = f'save/{args.model_name}/{args.save_name}'

        driver_reward_table = pd.DataFrame()
        for filename in os.listdir(save_path+'/driver_reward_table'):
            if filename.endswith(".csv"):
                file_path = os.path.join(save_path+'/driver_reward_table', filename)
                # 读取CSV文件并将其追加到driver_reward_table中
                data = pd.read_csv(file_path)
                driver_reward_table = pd.concat([driver_reward_table, data])
                # 按 driver_id 分组并求和
                driver_reward_table = driver_reward_table.groupby('driver_id').sum().reset_index()

        driver_table = pd.DataFrame()
        for filename in os.listdir(save_path+'/driver_table'):
            if filename.endswith(".csv"):
                file_path = os.path.join(save_path+'/driver_table', filename)
                # 读取CSV文件并将其追加到driver_table中
                data = pd.read_csv(file_path)
                driver_table = pd.concat([driver_table, data])
                # 按 driver_id 分组并求和
                # driver_table = driver_table.groupby('driver_id').sum().reset_index()

        matched_requests = pd.DataFrame()
        for filename in os.listdir(save_path+'/matched_requests'):
            if filename.endswith(".csv"):
                file_path = os.path.join(save_path+'/matched_requests', filename)
                # 读取CSV文件并将其追加到matched_requests中
                data = pd.read_csv(file_path)
                matched_requests = pd.concat([matched_requests, data])

        num_all_requests, num_matched_requests = 0, 0
        for filename in os.listdir(save_path+'/others'):
            if filename.endswith(".txt"):
                file_path = os.path.join(save_path+'/others', filename)
                with open(file_path, 'r') as f:
                    data = f.readlines()
                    num_matched_requests += int(data[0].split(',')[0].split(':')[1])
                    num_all_requests += int(data[0].split(',')[1].split(':')[1])

        # GMV: the total value of orders served in the simulator
        # OCR: order completion rate
        gmv = driver_reward_table['current_overall_reward'].sum()
        ocr = driver_reward_table['num_finished_order'].sum() / num_all_requests
        print(f'matched requests num: {len(matched_requests)}, all requests num: {num_all_requests}')
        print(f'GMV: {gmv:.2f}, OCR: {ocr * 100:.2f}%')

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        matched_requests.to_csv(save_path + '/matched_requests.csv', index=False)
        driver_table.to_csv(save_path + '/driver_table.csv', index=False)
        driver_reward_table.to_csv(save_path + '/driver_reward_table.csv', index=False)

        with open(save_path+'/result.txt', 'w') as file:
            file.write(
                f'matched requests num: {len(matched_requests)}, all requests num: {num_all_requests}\n')
            file.write(f'GMV: {gmv:.2f}, OCR: {ocr * 100:.2f}%')

        # 将args转换为字典
        args_dict = vars(args)
        # 保存到文件
        with open(save_path+'/args.json', 'w') as f:
            json.dump(args_dict, f, indent=4)

    def save_daily_results(self, args):
        driver_reward_table_save_path = f'save/{args.dataset}/{args.model_type}/{args.save_name}/driver_reward_table'
        if not os.path.exists(driver_reward_table_save_path):
            os.makedirs(driver_reward_table_save_path)
        self.driver_reward_table.to_csv(
            driver_reward_table_save_path + f'/driver_reward_table_{self.experiment_date}.csv', index=False)

        driver_table_save_path = f'save/{args.dataset}/{args.model_type}/{args.save_name}/driver_table'
        if not os.path.exists(driver_table_save_path):
            os.makedirs(driver_table_save_path)
        self.driver_table.to_csv(
            driver_table_save_path + f'/driver_table_{self.experiment_date}.csv', index=False)

        matched_request_save_path = f'save/{args.dataset}/{args.model_type}/{args.save_name}/matched_requests'
        if not os.path.exists(matched_request_save_path):
            os.makedirs(matched_request_save_path)
        self.matched_requests.to_csv(
            matched_request_save_path + f'/matched_requests_{self.experiment_date}.csv', index=False)

        others_save_path = f'save/{args.dataset}/{args.model_type}/{args.save_name}/others'
        if not os.path.exists(others_save_path):
            os.makedirs(others_save_path)
        with open(others_save_path + f'/others_{self.experiment_date}.txt', 'w') as file:
            file.write(
                f'matched requests num: {len(self.matched_requests)}, all requests num: {self.num_all_requests}\n')

        agent_memory_save_path = f'save/{args.dataset}/{args.model_type}/{args.save_name}/agent_memory'
        if not os.path.exists(agent_memory_save_path):
            os.makedirs(agent_memory_save_path)
        scoring_agent_file = f'/scoring_agent_memory_{self.experiment_date}.json'
        with open(agent_memory_save_path+scoring_agent_file, 'w', encoding='utf-8') as f:
            json.dump(self.Agent.score_memory, f, indent=4)
        review_agent_file = f'/review_agent_memory_{self.experiment_date}.json'
        with open(agent_memory_save_path + review_agent_file, 'w', encoding='utf-8') as f:
            json.dump(self.Agent.review_memory, f, indent=4)
        dispatching_agent_file = f'/dispatching_agent_memory_{self.experiment_date}.json'
        with open(agent_memory_save_path + dispatching_agent_file, 'w', encoding='utf-8') as f:
            json.dump(self.Agent.dispatch_memory, f, indent=4)
        repositioning_agent_file = f'/repositioning_agent_memory_{self.experiment_date}.json'
        with open(agent_memory_save_path + repositioning_agent_file, 'w', encoding='utf-8') as f:
            json.dump(self.Agent.reposition_memory, f, indent=4)

    def save_cache(self, args):
        cache_path = f'save/{args.dataset}/{args.model_type}/{args.save_name}/cache_{self.experiment_date}_{args.description}'
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.driver_table.to_csv(cache_path + f'/driver_table.csv', index=False)
        self.driver_reward_table.to_csv(cache_path + f'/driver_reward_table.csv', index=False)

        self.matched_requests_arrive_num.to_csv(cache_path + f'/matched_requests_arrive_num.csv', index=False)
        self.matched_requests_origin.to_csv(cache_path + f'/matched_requests_origin.csv', index=False)

        self.wait_requests.to_csv(cache_path + f'/wait_requests.csv', index=False)
        self.matched_requests.to_csv(cache_path + f'/matched_requests.csv', index=False)

        other_dict = {'time': self.time, 'current_step': self.current_step, 'num_all_requests': self.num_all_requests,
                      'experiment_date': self.experiment_date, 'finished_driver_id': str(self.finished_driver_id)}
        with open(cache_path + '/other_dict.txt', 'w') as file2:
            file2.write(str(other_dict))

        scoring_agent_file = f'/scoring_agent_memory.json'
        with open(cache_path + scoring_agent_file, 'w', encoding='utf-8') as f:
            json.dump(self.Agent.score_memory, f, indent=4)
        review_agent_file = f'/review_agent_memory.json'
        with open(cache_path + review_agent_file, 'w', encoding='utf-8') as f:
            json.dump(self.Agent.review_memory, f, indent=4)
        dispatching_agent_file = f'/dispatching_agent_memory.json'
        with open(cache_path + dispatching_agent_file, 'w', encoding='utf-8') as f:
            json.dump(self.Agent.dispatch_memory, f, indent=4)
        repositioning_agent_file = f'/repositioning_agent_memory.json'
        with open(cache_path + repositioning_agent_file, 'w', encoding='utf-8') as f:
            json.dump(self.Agent.reposition_memory, f, indent=4)

    def check_cache(self, args):
        cache_path = f'save/{args.dataset}/{args.model_type}/{args.save_name}/cache_{self.experiment_date}_{args.description}'
        self.driver_table = pd.read_csv(cache_path + '/driver_table.csv')
        self.driver_table['driver_id'] = self.driver_table['driver_id'].astype(str)

        self.driver_reward_table = pd.read_csv(cache_path + '/driver_reward_table.csv')

        self.matched_requests_arrive_num = pd.read_csv(cache_path + '/matched_requests_arrive_num.csv')
        self.matched_requests_origin = pd.read_csv(cache_path + '/matched_requests_origin.csv')

        self.wait_requests = pd.read_csv(cache_path + '/wait_requests.csv')
        self.wait_requests['order_id'] = self.wait_requests['order_id'].astype(str)

        self.matched_requests = pd.read_csv(cache_path + '/matched_requests.csv')
        self.matched_requests['order_id'] = self.matched_requests['order_id'].astype(str)
        self.matched_requests['driver_id'] = self.matched_requests['driver_id'].astype(str)

        with open(cache_path+"/other_dict.txt", "r") as file:
            data = file.read()
            other_dict = eval(data)
        self.time = other_dict['time']
        self.current_step = other_dict['current_step']+1
        self.experiment_date = other_dict['experiment_date']
        self.num_all_requests = other_dict['num_all_requests']
        # self.requests_final_records = list(other_dict['requests_final_records'])
        self.finished_driver_id = list(other_dict['finished_driver_id'])

        self.request_databases = deepcopy(self.request_all[other_dict['experiment_date']])

        result_date = datetime.strptime(self.experiment_date, "%Y-%m-%d") + timedelta(minutes=self.current_step)
        self.curent_experiment_time = result_date.strftime("%Y-%m-%d %H:%M:%S")

        scoring_agent_file = f'/scoring_agent_memory.json'
        with open(cache_path + scoring_agent_file, 'r', encoding='utf-8') as f:
            self.Agent.score_memory = json.load(f)
        review_agent_file = f'/review_agent_memory.json'
        with open(cache_path + review_agent_file, 'r', encoding='utf-8') as f:
            self.Agent.review_memory = json.load(f)
        dispatching_agent_file = f'/dispatching_agent_memory.json'
        with open(cache_path + dispatching_agent_file, 'r', encoding='utf-8') as f:
            self.Agent.dispatch_memory = json.load(f)
        repositioning_agent_file = f'/repositioning_agent_memory.json'
        with open(cache_path + repositioning_agent_file, 'r', encoding='utf-8') as f:
            self.Agent.reposition_memory = json.load(f)



