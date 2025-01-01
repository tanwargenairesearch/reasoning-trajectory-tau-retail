import inspect
from typing import Any, Dict

from tau_bench.envs.tool import Tool as TauBenchTool
from smolagents.tools import Tool

from tau_bench.envs.user import BaseUserSimulationEnv


def convert_tool(tau_retail_tool : TauBenchTool, data: Dict[str, Any]) -> Tool:

    class TauRetailToolWrapper(Tool):
        def __init__(self, tau_retail_tool : TauBenchTool, data: Dict[str, Any]):
            tool_info = tau_retail_tool.get_info()['function']
            self.name = tool_info['name']
            self.description = tool_info['description']
            self.output_type = "string"
            self.inputs = tool_info['parameters']['properties']
            self.tau_retail_tool = tau_retail_tool
            self.data = data
            super().__init__()

        def forward(self, *args, **kwargs):
            return self.tau_retail_tool.invoke(data=self.data, **kwargs)
    
    original_signature = inspect.signature(tau_retail_tool.invoke)
    new_parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(
        original_signature.parameters.values()
    )
    new_parameters = list(filter(lambda p : p.name != 'data', new_parameters))
    new_signature = original_signature.replace(parameters=new_parameters)
    TauRetailToolWrapper.forward.__signature__ = new_signature
    return TauRetailToolWrapper(tau_retail_tool, data)


class RespondToCustomer(Tool):
    name = 'respond_customer'
    description = "Use this function to respond to customer with defined query."
    inputs = {"query": {
                            "type": "string",
                            "description": "query or question or clarification to ask customer.",
                        }}
    output_type = "string"

    def __init__(self, user : BaseUserSimulationEnv):
        self.user = user
        super().__init__()

    def forward(self, query: str) -> str:
        response = self.user.step(content=query)
        return response