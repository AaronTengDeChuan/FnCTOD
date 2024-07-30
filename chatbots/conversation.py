#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import dataclasses
from copy import deepcopy
from enum import auto, IntEnum
from typing import List, Any, Dict, Union
import json
import re

import openai
import anthropic


class Conversation(object):
    def __init__(
        self,
        template_name: str = "",
        system_message: str = "",
        system_template: str = "{system_message}",
        roles: List[str] = ["User", "Assistant"],
        offset: int = 20,
        colon: str = ": ",
        separators: List[str] = ["\n", "\n", "\n"],
        function_type: str = "json",
        function_call_prefix: str = "<function_call>",
        function_call_suffix: str = "</function_call>",
        tokenizer=None,
        verbose: bool = False,
    ):
        self.template_name = template_name
        self._verbose = verbose
        self.offset = offset  # context window
        self.tokenizer = tokenizer

        # function call template
        self.function_type = function_type
        self.function_call_prefix = function_call_prefix
        self.function_call_suffix = function_call_suffix

        # conversation template
        # assert self.template_name is not None
        if self.template_name == "chatgpt":
            self.system_template = "{system_message}"
            self.separators = ("\n\n", "\n", "\n")
            self.roles = ("user", "assistant", "function")
            self.colon = ": "
        elif self.template_name == "claude":
            self.system_template = "{system_message}"
            self.separators = ("\n\n", "\n\n", "\n\n")
            self.roles = (anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT)
            self.colon = ""
        elif self.template_name == "alpaca":
            self.system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            self.system_template = "{system_message}"
            self.separators = ("\n\n", "\n\n", "</s>")
            self.roles = ("### Instruction", "### Response")
            self.colon = ": "
        elif self.template_name == "vicuna":
            self.system_message = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
            )
            self.system_template = "{system_message}"
            self.separators = (" ", " ", "</s>")
            self.roles = ("USER", "ASSISTANT")
            self.colon = ": "
        elif self.template_name == "baize":
            self.system_message = "The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format."
            self.system_template = "{system_message}"
            self.separators = ("\n", "\n", "\n")
            self.roles = ("[|Human|]", "[|AI|]")
            self.colon = ""
        elif self.template_name == "llama2":
            self.system_template = "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
            self.separators = ("", " ", " </s><s>")
            self.roles = ("[INST]", "[/INST]")
            self.colon = " "
        elif self.template_name == "baichuan2":
            self.system_template = "{system_message}"
            self.separators = ("", "", "")
            self.roles = ("<reserved_106>", "<reserved_107>")
            self.colon = ""
        elif self.template_name == "openassistant":  # llama-based
            self.system_template = "{system_message}"
            self.separators = ("", "", "</s>")
            self.roles = ("<|prompter|>", "<|assistant|>")
            self.colon = ""
        elif self.template_name == "zephyr":
            self.system_template = "<|system|>\n{system_message}"
            self.separators = ("</s>\n", "</s>\n", "</s>\n")
            self.roles = ("<|user|>", "<|assistant|>")
            self.colon = "\n"
        else:  # customized
            self.system_message = system_message
            self.system_template = system_template
            self.roles = roles
            self.colon = colon
            self.separators = separators

    def separate_func_call_and_output(self, messages):
        new_messages = []
        for idx, message in enumerate(messages):
            if message["role"] in ["system", "user"]:
                new_messages.append(message)
            elif message["role"] == "assistant" and "function_call" not in message:
                new_messages.append(message)
            elif message["role"] == "assistant" and "function_call" in message:
                function_call = message["function_call"]
                if "results" in function_call:
                    assert function_call["results"], f"Empty results: {message} ({idx + 1} / {len(messages)})"
                    assert idx == len(messages) - 1 or message["content"], f"Empty content: {message} ({idx + 1} / {len(messages)})"
                    new_messages.append({"role": "assistant", "content": f"FunctionCall(arguments={function_call.get('arguments')}, name='{function_call.get('function')}')"})
                    new_messages.append({"role": "function", "name": function_call.get('function'), "content": function_call["results"]})
                    if message["content"]:
                        new_messages.append({"role": "assistant", "content": message["content"]})
                else:
                    new_messages.append(message)
            else:
                raise ValueError(f"Invalid role: {message['role']}")
        return new_messages


    def get_prompt(
        self,
        system_message=None,
        messages: List[Dict] = [],
        functions: List[Dict] = [],
        function_call: Dict = {},
        examples: List[List] = [],
    ) -> Union[List, str]:
        system_messages = []

        # part 1: system instruction
        if system_message:
            system_messages.append(system_message)
        else:
            system_messages.append(self.system_message)

        # part 2: functions
        if functions:
            function_prompt = self.get_functions(functions=functions)
            system_messages.append(function_prompt)

        # part 3: examples
        if examples:
            example_prompt = self.get_examples(examples=examples)
            system_messages.append(example_prompt)

        # combine them, fill in the template
        system_prompt = "\n\n".join(system_messages)

        if self.tokenizer is None:
            system_prompt = self.system_template.format(system_message=system_prompt)

            # part 4: the current conversation, consisting of the current turn, with the function_call prefix
            conversation = self.get_conversation(
                messages=messages, function_call=function_call
            )
            ret = system_prompt + self.separators[0] + conversation
        else:
            # construct messages and then apply chat template
            ret = self.apply_chat_template(system_prompt, messages=deepcopy(messages), function_call=function_call)

        return ret

    def get_examples(self, examples):
        example_prompts = []
        for example in examples:
            if self.template_name != "llama2":
                example_prompt = self.get_conversation(example)
            else:  # not use instruction and system tokens in the examples, only for llama2
                example_prompt = self.get_conversation(
                    messages=example,
                    function_call={},
                    roles=["User", "Assistant"],
                    colon=": ",
                    separators=["", "\n", "\n"],
                )
            example_prompt = "<EXAMPLE>\n" + example_prompt + "\n</EXAMPLE>"
            example_prompts.append(example_prompt)

        example_prompts = "\n\n".join(example_prompts)
        example_prompts += "\n\n"

        return example_prompts

    def apply_chat_template(
            self,
            system_prompt: str = "",
            messages: List[Dict] = [],
            function_call: Dict = {},
    ) -> str:
        if not messages:
            return ""

        new_messages = []
        start_idx = 0
        if messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
            new_messages.append(messages[0])
            start_idx = 1

        messages = messages[start_idx:][-(self.offset + 1):]
        last_assistant_turn = None
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                new_messages.append(message)
            elif message["role"] == "assistant":
                if idx == len(messages) - 1:
                    last_assistant_turn = message
                    break
                assistant_content = ""
                # add function call
                if "function_call" in message:
                    function_call_json = json.dumps(message["function_call"])
                    assistant_content += (
                        self.function_call_prefix
                        + function_call_json
                        + self.function_call_suffix
                    )
                # add content
                assistant_content += message["content"]
                new_messages.append({"role": "assistant", "content": assistant_content})
            else:
                raise ValueError(f"Invalid role: {message['role']}")

        ret = self.tokenizer.apply_chat_template(
            new_messages, tokenize=False, add_generation_prompt=True)

        if last_assistant_turn:
            assert not (isinstance(function_call, dict) and function_call), \
                f"The last message is an assistant turn, but the function call is not empty: {function_call}"
            ret += last_assistant_turn["content"]

        # add the prefix to trigger the arguments output
        if isinstance(function_call, dict) and function_call:
            function_name = function_call["name"]
            assistant_content = f'{{"function": "{function_name}", "arguments":'
            ret += self.function_call_prefix + assistant_content

        return ret

    def get_conversation(
        self,
        messages: List[Dict] = (),
        function_call: Dict = {},
        roles: List = [],
        colon: str = "",
        separators: List = [],
        predict: bool = True,
    ) -> str:
        if not messages:
            return ""

        # if not override
        if not roles:
            roles = self.roles
        if not colon:
            colon = self.colon
        if not separators:
            separators = self.separators

        if len(roles) == 2:
            user_role, assistant_role = roles
        elif len(roles) == 3:
            user_role, assistant_role, function_role = roles
        else:
            raise ValueError(f"Invalid roles: {roles}")

        # content window
        messages = messages[-(self.offset + 1) :]

        # exclude the system message
        contain_system = False
        if messages[0]["role"] == "system":
            messages = messages[1:]
            contain_system = True

        # construct the message prompt
        ret = ""
        for midx, message in enumerate(messages):
            if message["role"] == "user":
                user_content = message["content"]
                if (
                    midx == 0 and contain_system and user_role == "[INST]"
                ):  # do not add user role at the first turn, only for llama2
                    ret += colon + user_content + separators[1]
                else:
                    ret += user_role + colon + user_content + separators[1]
            elif message["role"] == "assistant":
                assistant_content = ""
                # add function call
                if "function_call" in message:
                    function_call_json = json.dumps(message["function_call"])
                    assistant_content += (
                        self.function_call_prefix
                        + function_call_json
                        + self.function_call_suffix
                    )
                # add content
                assistant_content += message["content"]
                ret += assistant_role + colon + assistant_content

                # do not append the separators in the current turn
                if midx + 1 < len(messages) or not predict:
                    ret += separators[2]
                if midx + 1 == len(messages) and predict:
                    ret = ret.strip()
            elif message["role"] == "function":
                if midx + 1 == len(messages):
                    ret += separators[2]
                ret += function_role + colon + message["content"] + separators[2]

        # add the prefix to trigger the arguments output
        if isinstance(function_call, dict) and function_call:
            function_name = function_call["name"]
            assistant_content = f'{{"function": "{function_name}", "arguments":'
            ret += (
                assistant_role
                + self.colon
                + self.function_call_prefix
                + assistant_content
            )

        return ret

    def function2description(self, function):
        # convert the function json dictionary to natural language description
        text = []
        function_name = function["name"]
        function_description = function["description"]
        function_parameters = function["parameters"]["properties"]
        text.append(
            f"Function name: {function_name}",
        )
        text.append(f"Function description: {function_description}")
        text.append("Function arguments:")
        for param_name, param_info in function_parameters.items():
            param_type = param_info["type"]
            param_description = param_info["description"]
            param_desc = f" - {param_name} ({param_type}): {param_description}"
            if "enum" in param_info:
                param_enum = param_info["enum"]
                param_enum = ", ".join(param_enum)
                param_desc += f" (must be one of {param_enum})"
            text.append(param_desc)
        text = "\n".join(text)
        return text

    def get_functions(self, functions):
        """
        param: functions (dict)
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            }
        }
        return the prompt text
        """
        if self.function_type == "json":
            if self.template_name == "claude":
                prompts = []
                for function in functions:
                    function_prompt = (
                        f'<function_name>{function["name"]}</function_name>\n'
                    )
                    function_prompt += f'<function_description>{function["description"]}</function_description>'
                    parameters = function["parameters"]["properties"]
                    for parameter_name, parameter_info in parameters.items():
                        parameter_type = parameter_info["type"]
                        parameter_description = parameter_info["description"]
                        parameter_prompt = (
                            f"<parameter_name>{parameter_name}</parameter_name>\n"
                        )
                        parameter_prompt += f"<parameter_description>{parameter_description}</parameter_description>\n"
                        parameter_prompt += (
                            f"<parameter_type>{parameter_type}</parameter_type>\n"
                        )
                        if "enum" in parameter_info:
                            parameter_enum = parameter_info["enum"]
                            parameter_prompt += (
                                f"<parameter_enum>{parameter_enum}</parameter_enum>\n"
                            )
                        parameter_prompt = (
                            f"<parameter>{parameter_prompt}</parameter>\n"
                        )
                    function_prompt += parameter_prompt
                    if "required" in function["parameters"]:
                        function_prompt += f'<required_parameters>{function["parameters"]["required"]}</required_parameters>'
                    function_prompt = f"<function>{function_prompt}</function>\n"
                    prompts.append(function_prompt)
                prompt = "\n\n".join(prompts)
                prompt = f"<functions>{prompt}</functions>\n"

            else:
                prompts = []
                for function in functions:
                    function_dict = {
                        "name": function["name"],
                        "description": function["description"],
                    }
                    parameter_list = []
                    parameters = function["parameters"]["properties"]
                    for parameter_name, parameter_info in parameters.items():
                        parameter_type = parameter_info["type"]
                        parameter_description = parameter_info["description"]
                        parameter_dict = {
                            "name": parameter_name,
                            "type": parameter_type,
                            "description": parameter_description,
                        }
                        if "enum" in parameter_info:
                            parameter_dict["possible_values"] = parameter_info["enum"]
                        parameter_list.append(parameter_dict)
                    function_dict["arguments"] = parameter_list
                    if "required" in function["parameters"]["properties"]:
                        function_dict["required"] = function["parameters"][
                            "properties"
                        ]["required"]

                    # json obj -> str
                    function_prompt = json.dumps(function_dict, indent=4)
                    function_prompt = "<FUNCTION>\n" + function_prompt + "\n</FUNCTION>"
                    prompts.append(function_prompt)

                prompt = "\n".join(prompts)
                prompt = "<FUNCTIONS>\n" + prompt + "\n</FUNCTIONS>\n\n"

        elif self.function_type == "text":
            prompts = []
            for function in functions:
                function_prompt = self.function2description(function)
                prompts.append(function_prompt)
            prompt = "\n\n".join(prompts)
            prompt = "<FUNCTIONS>\n" + prompt + "\n</FUNCTIONS>\n\n"

        else:
            raise NotImplementedError

        # example
        prompt += 'To call a function with a JSON object of the following format: {"function": "function_name", "arguments": {"argument1": "argument_value", "argument2": "argument_value"}}'

        return prompt

    def get_response(
        self,
        text,
        function_call={},
        required=["function_call", "content"],
        stop_strs=[("</s>", 0), ("\n\n", 0)],
    ):
        def extract_first_dict(s):
            first_left_brace = s.find("{")
            start = s.find("{")
            while start >= 0 and start <= len(s):
                index = s.find("}", start)
                if index == -1:  # No more occurrences
                    break
                try:
                    json_obj = s[first_left_brace : index + 1]
                    json_dict = json.loads(json_obj)
                    json_str = json.dumps(json_dict)
                    return json_obj
                except:
                    start = index + 1
            return ""

        original_text = text
        text = text.lower().strip()
        # remove \_
        text = text.replace("\_", "_")

        function_call_suffix = self.function_call_suffix.strip()
        function_call_prefix = self.function_call_prefix.strip()

        # the beginning of next turn
        if isinstance(function_call, dict) and "name" in function_call and "arguments" not in function_call:
            function_name = function_call["name"]
            response_prefix = f'{{"function": "{function_name}", "arguments": '
            text = response_prefix + text

        parsed_function_call = {}
        parsed_response = text

        # get function call
        if "function_call" in required:
            try:
                if function_call_suffix in text:
                    parsed_response = text.split(function_call_suffix)[1].strip()
                    parsed_function_call = text.split(function_call_suffix)[0].strip()
                    if function_call_prefix in parsed_function_call:
                        parsed_function_call = parsed_function_call.split(
                            function_call_prefix
                        )[1].strip()
                    parsed_function_call = json.loads(parsed_function_call)
                else:
                    parsed_function_call = extract_first_dict(text)
                    parsed_response = text.replace(parsed_function_call, "").strip()
                    parsed_function_call = json.loads(parsed_function_call)
            except Exception as error:
                # print(error)
                parsed_function_call = original_text

        # get response
        if "content" in required:
            parsed_response = text
            if function_call_prefix in parsed_response:
                parsed_response = parsed_response.split(function_call_prefix)[0].strip()

        # get response
        for s in stop_strs:
            if s[0] in parsed_response:
                parsed_response = parsed_response.split(s[0])[s[1]]

        ret = {
            "role": "assistant",
            "content": parsed_response,
            "function_call": parsed_function_call,
        }
        return ret


import datetime


def current_time():
    """Get the current local time as a string."""
    return str(datetime.now())


def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b


tools = [current_time, multiply]


if __name__ == '__main__':
    from chatbots.configs import llm_configs
    from transformers import AutoTokenizer

    model_name = ["llama-2-13b-chat", "llama-3-8b", "llama-3-8b-instruct",
                  "llama-3.1-8b", "llama-3.1-8b-instruct",
                  "zephyr-7b-beta", "baichuan-13b-chat", "vicuna-13b-v1.5"]
    name2tokenizer = {name: AutoTokenizer.from_pretrained(
        llm_configs[name]["model_name"], trust_remote_code=True) for name in model_name}

    messages = [
        {"role": "system", "content": "This is a conversation between a user and an assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
        {"role": "user", "content": "What is the weather today?"},
        {"role": "assistant", "content": "It is sunny."},
        {"role": "user", "content": "Can you help me with something?"},
    ]

    fc_prefix = "<function_call> "
    assistant_content = f'{{"function": "find_book_hotel", "arguments":'
    assistant_prefix = fc_prefix + assistant_content

    for name, tokenizer in name2tokenizer.items():
        chat_prompt = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True)
        print(f"########### Model: {name} ###########")
        print(f"CHAT TEMPLATE: {tokenizer.chat_template}\n")
        print(f"###{chat_prompt}###")
        print(tokenizer.convert_ids_to_tokens(tokenizer.encode(chat_prompt)))
        print(f"###{chat_prompt + assistant_prefix}###")
        print(tokenizer.convert_ids_to_tokens(tokenizer.encode(chat_prompt + assistant_prefix)))

        chat_prompt = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": assistant_prefix}], tokenize=False, add_generation_prompt=False)
        print(f"###{chat_prompt}###")
        print(tokenizer.convert_ids_to_tokens(tokenizer.encode(chat_prompt)))

        print("\n\n")

