import argparse
import json
import math
import os
from typing import List

from datetime import datetime
from litellm import provider_list
from env.retail import TaskExecutionResult, generate_trajectory_and_evaluate_reward
from tau_bench.envs.retail.tasks_dev import TASKS_DEV
from tau_bench.envs.retail.tasks_test import TASKS_TEST
from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
from tau_bench.envs.user import UserStrategy, load_user
from agents.retail_customer_support.llm_engines import GeminiEngine, OpenAIEngine

def save_trajectory(result: TaskExecutionResult, file_str: str) :
    with open(file_str, 'w', encoding='utf-8') as f:
        data = {
            'instruction': result.task.instruction,
            'trajectory': result.trajectory,
            'ground_truth_tool_calls': [{'tool_name': action.name, 'tool_arguments': action.kwargs} for action in result.task.actions],
            'reward_info': {
                'correct_actions_reward': result.rewardActionInfo.r_actions,
                'final_reward': result.rewardResult.reward
            } 
        }
        json.dump(data, f, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        default="openai",
        # for now only few models are supported, but will adding models here for Agents. 
        # reason: agents are build on top of smolagent framework (which does support all models) 
        # but to generalize it for this use-case in this project, few changes are required.
        choices=['openai', 'google'],
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        default="openai",
        # this uses default set of models from litellm which tau-bench uses. 
        # as of now, user agents are not build on top of smolagent framework, 
        # since for first iteration, only simulated responses are required w/o memory.
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=10, help="Run first 10 tasks only by default.")
    parser.add_argument(
        "--task-split",
        type=str,
        default="train",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--log-dir", type=str, default="results")
    args = parser.parse_args()
    print(args)

    time_str = datetime.now().strftime("%m%d%H%M%S")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    user = load_user(
            user_strategy=args.user_strategy, model=args.user_model, provider=args.user_model_provider
        )
    
    if args.model_provider == 'openai':
        agent_engine = OpenAIEngine(model_name=args.model)
    elif args.model_provider == 'google':
        agent_engine = GeminiEngine(model_name=args.model)
    else:
        raise NotImplementedError('provider is not supported yet, please free feel to implement it.')
    
    if args.task_split == 'train':
        tasks = TASKS_TRAIN[args.start_index:args.end_index]
    elif args.task_split == 'dev':
        tasks = TASKS_DEV[args.start_index:args.end_index]
    else:
        tasks = TASKS_TEST[args.start_index:args.end_index]
    rewards = []
    
    for task in tasks:
        task_reward = generate_trajectory_and_evaluate_reward(
            model=args.model, 
            task=task, 
            user=user, 
            llm_engine=agent_engine
        )
        rewards.append(task_reward)
        # print(f"Task Instruction:\n{task.instruction}\n")
        # print(f"Task rewards: {task_reward.rewardActionInfo.r_actions}")
        file_str = f"{args.log_dir}/{task_reward.computedHash}_{time_str}.json"
        save_trajectory(task_reward, file_str)
    
    print("\n\n>>>>>>>>>> REWARD SUMMARY >>>>>>>>>>>>>")
    r_actions = 0.0
    r_total = 0.0
    for r in rewards:
        r_actions += r.rewardActionInfo.r_actions
        r_total += r.rewardResult.reward
    print('total_executions: ' + str(len(rewards)))
    print('total_actions rewards: ' + str(r_actions))
    print('total rewards: ' + str(r_total))
    print(">>>>>>>>>> REWARD SUMMARY >>>>>>>>>>>>>")