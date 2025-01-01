import json
from env.retail import generate_trajectory_and_evaluate_reward
from tau_bench.envs.retail.tasks_test import TASKS_TEST
from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
from tau_bench.envs.user import load_user


if __name__ == '__main__':
    user = load_user(
            user_strategy='llm', model='gpt-4o', provider='openai'
        )
    tasks_test = TASKS_TEST[:5]
    rewards = []
    for task in tasks_test:
        task_reward = generate_trajectory_and_evaluate_reward(task, user)
        rewards.append(task_reward)
        print(f"Task Instruction:\n{task.instruction}\n")
        print(f"Task rewards: {task_reward.rewardActionInfo.r_actions}")
    
    print(">>>>>>>>>> REWARD SUMMARY >>>>>>>>>>>>>")
    r_actions = 0.0
    r_total = 0.0
    for r in rewards:
        r_actions += r.rewardActionInfo.r_actions
        r_total += r.rewardResult.reward
    print('total_executions: ' + str(len(rewards)))
    print('total_actions rewards: ' + str(r_actions))
    print('total rewards: ' + str(r_total))
    print(">>>>>>>>>> REWARD SUMMARY >>>>>>>>>>>>>")