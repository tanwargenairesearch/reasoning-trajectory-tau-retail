import json
from env.retail import generate_trajectory_and_evaluate_reward
from tau_bench.envs.retail.tasks_test import TASKS_TEST
from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
from tau_bench.envs.user import load_user


if __name__ == '__main__':
    user = load_user(
            #user_strategy='verify', model='claude-3-5-sonnet-20240620', provider='anthropic'
            user_strategy='llm', model='gpt-4o', provider='openai'
        )
    tasks_test = TASKS_TEST[:5]
    rewards = []
    running_action_rewards = 0.0
    for task in tasks_test:
        print(f"Task Instruction:\n{task.instruction}\n")
        task_reward = generate_trajectory_and_evaluate_reward(task, user)
        #print(">>>>>>>>>>> Hash Info")
        #print(f"agent hash: {reward_data['agent_data_hash']} ---- gt hash: {reward_data['gt_data_hash']}")
        agent_call_logs = []
        #print(">>>>>> tool call logs")
        for i, step_log in enumerate(task_reward.agentLogs):
            if "tool_call" in step_log:
                if step_log["tool_call"]['tool_name'] in ['respond_customer', 'final_answer']:
                    continue
                if "error" in step_log:
                    continue
                print(json.dumps(step_log["tool_call"]))
        #print(">>>>>> task action calls")
        # for action in task.actions:
        #     print(f"name: {action.name} args: {action.kwargs}")
        rewards.append(task_reward)
        running_action_rewards += task_reward.rewardActionInfo.r_actions
        # print(">>>>>>>>>> Reward Info")
        # print(f"running rewards: {running_action_rewards}")
    
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