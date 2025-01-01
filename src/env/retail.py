from dataclasses import dataclass
import json
from typing import Any, Dict, Type
from agents.retail_customer_support.llm_engines import GeminiEngine, OpenAIEngine
from tau_bench.envs.base import consistent_hash, to_hashable
from tau_bench.envs.retail.data import load_data
from tau_bench.envs.retail.tools import ALL_TOOLS
from tau_bench.envs.retail.wiki import WIKI
from tau_bench.envs.tool import Tool
from tau_bench.envs.user import BaseUserSimulationEnv
from tau_bench.types import RewardActionInfo, RewardOutputInfo, RewardResult, Task
from agents.retail_customer_support.tool_wrapper import (
    convert_tool,
    RespondToCustomer
)
from smolagents.tools import (
    Toolbox
)

from agents.retail_customer_support.agents import (
    RetailSupportMultiStepAgent
)

@dataclass
class TaskExecutionResult:
    task: Task
    computedHash: str
    groundTruthHash: str
    agentLogs: list[Any]
    rewardActionInfo: RewardActionInfo
    rewardOutputInfo: RewardOutputInfo
    rewardResult: RewardResult

def generate_trajectory_and_evaluate_reward(task: Task, user: BaseUserSimulationEnv) -> TaskExecutionResult : 
    dataset = load_data()
    converted_tools = [convert_tool(tool, dataset) for tool in ALL_TOOLS]
    converted_tools.append(RespondToCustomer(user))
    agent = RetailSupportMultiStepAgent(
        tool_box=Toolbox(
            tools=converted_tools
        ),
        #llm_engine=GeminiEngine(), 
        llm_engine=OpenAIEngine(model_name="gpt-4o"),
        policy_wiki=WIKI, 
        max_iterations=50, 
        planning_interval=4, 
        belief_computation_interval=2
    )
    response = user.reset(instruction=task.instruction)
    agent.run(response)

    data_hash = consistent_hash(to_hashable(dataset))
    trajectory = agent.extract_trajectory()
    reward = 1.0
    new_data = load_data()
    tools_map: Dict[str, Type[Tool]] = {
        tool.get_info()["function"]["name"]: tool for tool in ALL_TOOLS
    }
    for action in task.actions:
       tools_map[action.name].invoke(data=new_data, **action.kwargs)
    gt_data_hash = consistent_hash(to_hashable(new_data))
    reward_action_info = RewardActionInfo(
        r_actions=data_hash == gt_data_hash, gt_data_hash=gt_data_hash
    )
    reward_output_info = None

    if len(task.outputs) > 0:
        # check outputs
        r_outputs = 1.0
        outputs = {}
        for output in task.outputs:
            found = False
            for i, step_log in enumerate(agent.logs):
                if "tool_call" in step_log:
                    tool_call = step_log["tool_call"]
                    if tool_call['tool_name'] == 'respond_customer':
                        responsed_content = tool_call['tool_arguments']['query']
                        if output.lower() in responsed_content.lower().replace(",", ""):
                            found = True
                            break
            outputs[output] = found
            if not found:
                r_outputs = 0.0
                reward = 0.0
        reward_output_info = RewardOutputInfo(r_outputs=r_outputs, outputs=outputs)
    reward_result = RewardResult(reward=reward, info= reward_output_info if reward_output_info else reward_action_info, actions=[])
    return TaskExecutionResult(
        task=task,
        computedHash=data_hash,
        groundTruthHash=gt_data_hash,
        agentLogs=agent.logs,
        rewardActionInfo=reward_action_info,
        rewardOutputInfo=reward_output_info,
        rewardResult=reward_result
    )