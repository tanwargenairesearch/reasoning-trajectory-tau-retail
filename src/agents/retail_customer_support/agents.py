from dataclasses import dataclass
import json
import logging
import time
from typing import (
    Any, 
    Callable, 
    Dict, 
    List, 
    Optional
)

from transformers.utils import logging as transformers_logging

from smolagents.tools import (
    Toolbox,
    DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
    get_tool_description_with_args
)
from smolagents.utils import (
    parse_json_tool_call,
    parse_json_blob
)
from smolagents.models import MessageRole
from smolagents.agents import (
    format_prompt_with_tools,
    AgentParsingError,
    AgentExecutionError,
    AgentError,
    AgentGenerationError,
    AgentMaxIterationsError
)

from .prompts import (
    RETAIL_SUPPORT_AGENT_SYSTEM_PROMPT, 
    SYSTEM_PROMPT_GENERATE_BELIEF,
    SYSTEM_PROMPT_PLAN, 
    USER_PROMPT_GENERATE_BELIEF,
    USER_PROMPT_PLAN
)


logger = transformers_logging.get_logger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)

@dataclass
class BeliefFacts:
    step: int
    llmOutput: Any
    beliefState: str
    beliefStateExplaination: str
    knownFacts: List[str]
    unknownFacts: List[str]

@dataclass
class ExecutionPlan:
    step: int
    beliefFactsUsed: BeliefFacts
    planDetails: str


class RetailSupportMultiStepAgent:
    def __init__(
            self, 
            tool_box: Toolbox, 
            llm_engine: Callable,
            policy_wiki : str,
            system_prompt_template: str = RETAIL_SUPPORT_AGENT_SYSTEM_PROMPT,
            max_iterations:  int = 10,
            belief_computation_interval: Optional[int] = None,
            planning_interval: Optional[int] = None,
    ):
        self.toolbox = tool_box
        self.tool_parser = parse_json_tool_call
        self.llm_engine = llm_engine
        self.policy_wiki = policy_wiki
        self.system_prompt_template = system_prompt_template
        self.logs = []
        self.task = None
        self.max_iterations = max_iterations
        self.belief_computation_interval = belief_computation_interval
        self.planning_interval = planning_interval
        self.belief_facts = []
        self.computed_plans = []
        self.logger = logger

        # TODO: Add Logger Level

    def initialize_for_run(self):
        self.system_prompt = format_prompt_with_tools(
            self.toolbox, self.system_prompt_template, DEFAULT_TOOL_DESCRIPTION_TEMPLATE
        )
         
        self.logger.log(33, "======== New task ========")
        self.logger.log(34, self.task)
        
    def create_inner_memory_from_logs(self) -> List[Dict[str, str]]:
        task_message = {
            "role": MessageRole.USER,
            "content": "Task: " + self.task,
        }
        memory = [task_message]

        for step_log in self.logs:
            if "llm_output" in step_log:
                thought_message = {"role": MessageRole.ASSISTANT, "content": step_log["llm_output"].strip()}
                memory.append(thought_message)
            if "error" in step_log or "observation" in step_log:
                if "error" in step_log:
                    message_content = (
                        f"Error:\n"
                        + str(step_log["error"])
                        + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                    )
                elif "observation" in step_log:
                    message_content = f"Observation:\n{step_log['observation']}"
                tool_response_message = {"role": MessageRole.USER, "content": message_content}
                memory.append(tool_response_message)

        return memory
    
    def extract_trajectory(self) -> List[Any]:
        trajectory = []
        for step_log in self.logs:
            traj_node = {}
            if 'tool_call' in step_log:
                traj_node['action'] = step_log['tool_call']
            if 'rationale' in step_log:
                traj_node['thought'] = step_log['rationale'].replace("Thought: ", "")
            if 'observation' in step_log:
                traj_node['observation'] = step_log['observation']
            if 'error' in step_log:
                traj_node['observation'] = 'resulted in error'
            trajectory.append(traj_node)
        return trajectory
    
    def extract_action(self, llm_output: str, split_token: str) -> str:
        """
        Parse action from the LLM output

        Args:
            llm_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = llm_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception as e:
            self.logger.error(e, exc_info=1)
            raise AgentParsingError(
                f"Error: No '{split_token}' token provided in your output.\nYour output:\n{llm_output}\n. Be sure to include an action, prefaced with '{split_token}'!"
            )
        return rationale.strip(), action.strip()
    
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.toolbox).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_tools = self.toolbox.tools
        if tool_name not in available_tools:
            error_msg = f"Error: unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            self.logger.error(error_msg, exc_info=1)
            raise AgentExecutionError(error_msg)

        try:
            if isinstance(arguments, str):
                observation = available_tools[tool_name](arguments)
            elif isinstance(arguments, dict):
                observation = available_tools[tool_name](**arguments)
            else:
                raise AgentExecutionError(
                    f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                )
            return observation
        except Exception as e:
            if tool_name in self.toolbox.tools:
                raise AgentExecutionError(
                    f"Error in tool call execution: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following:\n{get_tool_description_with_args(available_tools[tool_name])}"
                )
    
    def compute_beliefs(self, step: int):
        #print(f"=========== Compute Facts @ {step} ===================")
        agent_memory = self.create_inner_memory_from_logs()
        message_system_prompt_belief_facts = {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT_GENERATE_BELIEF.replace("<<domain_knowledge>>",self.policy_wiki)}
        message_user_prompt_belief_facts = {
            "role": MessageRole.USER,
            "content": USER_PROMPT_GENERATE_BELIEF,
        }

        beliefFacts = None
        trials = 0
        while beliefFacts is None and trials < 3:
            try:
                llm_output = self.llm_engine(
                    [message_system_prompt_belief_facts] + agent_memory + [message_user_prompt_belief_facts], 
                    stop_sequences=["<belief_state_with_facts>"]
                )
                parsed_belief_and_facts = parse_json_blob(llm_output)
                #print(json.dumps(parsed_belief_and_facts, indent=2))
                beliefFacts = BeliefFacts(
                    step=step,
                    llmOutput=llm_output,
                    beliefState=parsed_belief_and_facts['belief_state'], 
                    beliefStateExplaination=parsed_belief_and_facts['belief_explanation'], 
                    knownFacts=parsed_belief_and_facts['known_facts'], 
                    unknownFacts=parsed_belief_and_facts['unknown_facts']
                )
            except Exception as e:
                if trials == 2:
                    raise AgentExecutionError(f"facts computation failed {e}")
                trials+=1
                # TODO Add logging for Failures

        self.belief_facts.append(beliefFacts)
    
    def compute_plan(self, step: int):
        #print(f"=========== Compute Plan @ {step} ===================")
        agent_memory = self.create_inner_memory_from_logs()

        # Get Latest Computed Facts (or Information Gathered)
        if len(self.belief_facts) == 0:
            raise AgentExecutionError(f"Facts not computed before computing plan @step: {step}")
        
        beliefFacts = self.belief_facts[-1]
        plan_update_message = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_PLAN.format(task=self.task),
            }
        
        previous_plan = f"Your previous prepared plan:\n{self.computed_plans[-1]}" if len(self.computed_plans) > 0 else ""

        plan_update_message_user = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_PLAN.format(
                    task=self.task,
                    tool_descriptions=self.toolbox.show_tool_descriptions(DEFAULT_TOOL_DESCRIPTION_TEMPLATE),
                    domain_knowledge=self.policy_wiki,
                    known_facts=beliefFacts.knownFacts,
                    unknown_facts=beliefFacts.unknownFacts,
                    previous_plan=previous_plan
                ),
            }
        llm_output = self.llm_engine(
            [plan_update_message] + agent_memory + [plan_update_message_user], 
            stop_sequences=["<end_plan>"]
        )
        #print(llm_output)
        computed_plan = ExecutionPlan(
            step=step,
            beliefFactsUsed=beliefFacts,
            planDetails=llm_output
        )
        self.computed_plans.append(computed_plan)
    
    def execute_step(self, step_index: int, running_log_entry: Dict[str, Any]):
        #print(f"\n\n ===================================== Step {step_index} =====================================")

        # get latest plan
        latest_computed_plan = self.computed_plans[-1]

        # get latest computed belief
        latest_computed_facts = self.belief_facts[-1]

        # create messages
        system_prompt = self.system_prompt.replace(
            "<<known_facts>>", 
            '\n'.join(latest_computed_facts.knownFacts)
        ).replace(
            "<<unknown_facts>>",
            '\n'.join(latest_computed_facts.unknownFacts)
        ).replace(
            "<<execution_plan>>", 
            latest_computed_plan.planDetails
        )
        message_system_prompt_step_execution = {"role": MessageRole.SYSTEM, "content": system_prompt}

        agent_memory = self.create_inner_memory_from_logs()
        try:
            llm_output = self.llm_engine([message_system_prompt_step_execution] + agent_memory, stop_sequences=["<end_action>", "Observation:"])
            #print(llm_output)
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")
        
        running_log_entry['llm_output'] = llm_output
        rationale, action = self.extract_action(llm_output=llm_output, split_token="Action:")

        try:
            tool_name, arguments = self.tool_parser(action)
        except Exception as e:
            raise AgentParsingError(f"Could not parse the given action: {e}.")

        running_log_entry["rationale"] = rationale
        running_log_entry["tool_call"] = {"tool_name": tool_name, "tool_arguments": arguments}

        if tool_name == "final_answer":
            if isinstance(arguments, dict):
                if "answer" in arguments:
                    answer = arguments["answer"]
                else:
                    answer = arguments
            else:
                answer = arguments
            running_log_entry["final_answer"] = answer
            return answer
        else:
            if arguments is None:
                arguments = {}
            observation = self.execute_tool_call(tool_name, arguments)
            updated_information = str(observation).strip()
            #print(f"Observation: {updated_information}")
            self.logger.info(updated_information)
            running_log_entry["observation"] = updated_information
            return running_log_entry
        
    def provide_final_answer(self, task) -> str:
        """
        This method provides a final answer to the task, based on the logs of the agent's interactions.
        """
        self.prompt = [
            {
                "role": MessageRole.SYSTEM,
                "content": "An agent tried to answer an user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
            }
        ]
        self.prompt += self.create_inner_memory_from_logs()
        self.prompt += [
            {
                "role": MessageRole.USER,
                "content": f"Based on the above, please provide an answer to the following user request:\n{task}",
            }
        ]
        try:
            return self.llm_engine(self.prompt)
        except Exception as e:
            return f"Error in generating final llm output: {e}."
    
    def run(self, task: str):
        self.task = task
        self.initialize_for_run()

        iteration = 0
        final_answer = None
        
        while final_answer is None and iteration < self.max_iterations:
            step_start_time = time.time()
            step_log_entry = {"iteration": iteration, "start_time": step_start_time}
            try:
                # Compute Facts
                if self.belief_computation_interval is None:
                    self.compute_beliefs(iteration)
                else:
                    if iteration % self.belief_computation_interval == 0:
                        self.compute_beliefs(iteration)
                
                # Compute Plan
                if self.planning_interval is None:
                    self.compute_plan(iteration)
                else:
                    if iteration % self.planning_interval == 0:
                        self.compute_plan(iteration)

                # Execute Step
                self.execute_step(step_index=iteration, running_log_entry=step_log_entry)


                if "final_answer" in step_log_entry:
                    final_answer = step_log_entry["final_answer"]
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                step_log_entry["error"] = e
            finally:
                step_end_time = time.time()
                step_log_entry["step_end_time"] = step_end_time
                step_log_entry["step_duration"] = step_end_time - step_start_time
                self.logs.append(step_log_entry)
                iteration += 1
        
        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = {"error": AgentMaxIterationsError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer
            final_step_log["step_duration"] = 0

        return final_answer