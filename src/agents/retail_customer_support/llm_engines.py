from copy import deepcopy
import os
import random
import time
from typing import Dict, List
from smolagents.models import MessageRole
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from openai import OpenAI

class GeminiEngine:
    def __init__(self, model_name="gemini-2.0-flash-exp"):
        self.model_name = model_name
        self.role_conversions = {
            MessageRole.ASSISTANT : 'model',
            MessageRole.TOOL_CALL : 'model',
            MessageRole.TOOL_RESPONSE : 'user',
            MessageRole.USER : 'user',
        }
    
    def get_clean_message_list(self, message_lists: List[Dict[str, str]]):
        """
        Subsequent messages with the same role will be concatenated to a single message.

        Args:
            message_list (`List[Dict[str, str]]`): List of chat messages.
        """
        final_message_list = []
        message_list_cpy = deepcopy(message_lists)  # Avoid modifying the original list
        for message in message_list_cpy:
            if not set(message.keys()) == {"role", "content"}:
                raise ValueError("Message should contain only 'role' and 'content' keys!")

            role = message["role"]
            if role not in MessageRole.roles():
                raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

            gemini_role = self.role_conversions[role]
            final_message_list.append({
                'role' : gemini_role,
                'parts' : message["content"],
            })
        return final_message_list

    def __call__(self, messages, stop_sequences=[], grammar=None):
        system_instruction = ""
        system_messages = list(filter(lambda m : m['role'] == MessageRole.SYSTEM, messages))
       
        system_instruction = system_messages[0]['content']
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=4096,
            temperature=0.3, 
            top_p=0.9,
            stop_sequences=stop_sequences
        )
        model = genai.GenerativeModel(self.model_name, system_instruction=system_instruction, generation_config=generation_config)
        messages_cpy = deepcopy(messages)
        messages_cpy = list(filter(lambda m : m['role'] != MessageRole.SYSTEM, messages_cpy))
        messages_cleaned = self.get_clean_message_list(messages_cpy[:-1])
        chat_session = model.start_chat(history=messages_cleaned)
        sleep_attempts = 0
        sleep_time = 2
        while True:
            try:
                response = chat_session.send_message(
                    messages_cpy[-1]['content']
                )
                return response.text
            except ResourceExhausted as re:
                sleep_attempts += 1
                if sleep_attempts > 5:
                    print(f"ResourceExhausted exception occurred 5 times in a row. Exiting.")
                    break
                time.sleep(sleep_time)
                sleep_time *= (2 + random.uniform(0, 1))

class OpenAIEngine:
    role_conversions = {
        MessageRole.TOOL_RESPONSE: MessageRole.USER,
    }

    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def get_clean_message_list(self, message_list: List[Dict[str, str]]):
        """
        Subsequent messages with the same role will be concatenated to a single message.

        Args:
            message_list (`List[Dict[str, str]]`): List of chat messages.
        """
        final_message_list = []
        message_list = deepcopy(message_list)  # Avoid modifying the original list
        for message in message_list:
            if not set(message.keys()) == {"role", "content"}:
                raise ValueError("Message should contain only 'role' and 'content' keys!")

            role = message["role"]
            if role not in MessageRole.roles():
                raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

            if role in self.role_conversions:
                message["role"] = self.role_conversions[role]

            if len(final_message_list) > 0 and message["role"] == final_message_list[-1]["role"]:
                final_message_list[-1]["content"] += "\n=======\n" + message["content"]
            else:
                final_message_list.append(message)
        return final_message_list

    def __call__(self, messages, stop_sequences=[], grammar=None):
        messages = self.get_clean_message_list(messages)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
            response_format=grammar
        )
        return response.choices[0].message.content