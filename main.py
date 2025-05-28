import os
import argparse
from simulator.simulator_env import Simulator
from datetime import datetime, timedelta
from simulator.utilities import *

# os.environ['WANDB_API_KEY'] = ''
os.environ['WANDB_MODE'] = 'offline'
os.environ['TOGETHER_API_KEY'] = 'XXX'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default='2015-07-27')
    parser.add_argument("--start_time", type=float, default=0, help='start time: 3am')
    parser.add_argument("--end_date", type=str, default='2015-07-31')
    parser.add_argument("--end_time", type=float, default=86400, help='end time: 9am')
    parser.add_argument("--delta_t", type=float, default=60, help='time interval for matching: 1 minute')
    parser.add_argument("--vehicle_speed", type=float, default=6.33, help='the speed of vehicle: 6.33m/s')
    parser.add_argument("--max_scoring_review_round", type=int, default=2, help='the number of scoring review round')

    parser.add_argument("--maximum_wait_time_mean", type=float, default=300, help='')
    parser.add_argument("--maximum_wait_time_std", type=float, default=0, help='')
    parser.add_argument("--request_interval", type=float, default=60, help='')
    parser.add_argument("--max_idle_time", type=float, default=300, help='')
    parser.add_argument("--pickup_dis_threshold", type=float, default=950, help='')

    parser.add_argument("--dataset", type=str, default='large',
                        help='small, small-500, medium, medium-1000, large')
    parser.add_argument("--model_type", type=str, default='together-api',
                        help='models type: gpt4-api, together-api, llama3, qwen2.5')
    parser.add_argument("--model_name", type=str, default='Meta-Llama-3.1-70B-Instruct-Turbo',
                        help='models: gpt-4, Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-8B-Instruct-Turbo')

    parser.add_argument("--description", type=str, default='test', help='None, or random number')
    parser.add_argument("--comments", type=str, default='', help='')

    return parser.parse_args()


def main(args):
    args = refine_args(args)

    # initialize simulator
    simulator = Simulator(args)

    # experiments date
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    start_time = timedelta(seconds=args.start_time)
    start_datetime = start_date + start_time

    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    end_time = timedelta(seconds=args.end_time)
    end_datetime = end_date + end_time

    current_date = start_datetime
    while current_date <= end_datetime:
        print(current_date.strftime('%Y-%m-%d'))
        simulator.experiment_date = current_date.strftime('%Y-%m-%d')

        cache_path = f'save/{args.dataset}/{args.model_type}/{args.save_name}/cache_{simulator.experiment_date}_{args.description}'
        if os.path.exists(cache_path) and os.listdir(cache_path):  # 判断文件夹是否存在
            simulator.check_cache(args)

            print(f'Cache Loaded:')
            print(f'- experiment date: {simulator.experiment_date}')
            print(f'- done at steps: {simulator.current_step-1}')

        else:
            # reset drive table, requested table etc.
            simulator.reset()

        for i in range(simulator.current_step, simulator.finish_run_step):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} {simulator.current_step}/{simulator.finish_run_step}")
            simulator.step(args)
            simulator.save_cache(args)
        print(current_date.strftime('%Y-%m-%d') + ' done')
        current_date += timedelta(days=1)
        simulator.save_daily_results(args)

    simulator.evaluation(args)


if __name__ == "__main__":
    para = parse_args()
    main(para)
