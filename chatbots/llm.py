#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import asyncio
import aiohttp
import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time
import json
import torch
from copy import deepcopy
from typing import Any, Dict, List, Union
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoConfig


import openai
import anthropic
import tiktoken

from chatbots.utils import *
from chatbots.configs import llm_configs, get_model_name
from chatbots.conversation import Conversation

"""
The main class for all LLMs: api-accessible gpt and claude, huggingface's llama-series and others
"""


class LLM:
    def __init__(
        self, model_name, no_load_model=False, interval=0.5, timeout=10.0, exp=2, patience=10, max_interval=4
    ):
        self.model_name = model_name
        self.openai_api = (
            True if any([x in self.model_name for x in ["gpt-3.5", "gpt-4"]]) else False
        )
        self.anthropic_api = True if "claude" in self.model_name else False

        # load model, either API-accessible or local models
        if self.openai_api:  # OPENAI API
            self.model = OpenAI(
                model=model_name,
                interval=interval,
                timeout=timeout,
                exp=exp,
                patience=patience,
                max_interval=max_interval,
            )
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif self.anthropic_api:  # CLAUDE API
            self.model = Claude(
                model=model_name,
                interval=interval,
                timeout=timeout,
                exp=exp,
                patience=patience,
                max_interval=max_interval,
            )
            self.tokenizer = None
        else:  # HUGGINGFACE MODELS
            if no_load_model:
                self.model = None
                self.tokenizer = None
            else:
                print(model_name)
                self.model, self.tokenizer = load_hf_model(model_name)
                gen_pti = self.model.generation_config.pad_token_id
                self.gen_config_pad_token_id = gen_pti
                if gen_pti is not None:
                    gen_pt = self.tokenizer.convert_ids_to_tokens(gen_pti)
                else:
                    gen_pt = ""
                print(f"'model.generation_config.pad_token_id': {gen_pti}\t{gen_pt}")
                eos_id = self.tokenizer.eos_token_id
                print(f"'tokenizer.eos_token_id': {eos_id}\t{self.tokenizer.convert_ids_to_tokens(eos_id)}")


    def generate(
        self,
        prompt,
        functions,  # only useful for openai or claude, otherwise have already included in prompt
        function_call,  # only useful for openai or claude, otherwise have already included in prompt
        temperature=0.5,
        top_p=0.5,
        max_tokens=128,
        n_seqs=1,
        stop=["\n\n", "User", "Example"],
    ):
        # api-accessible models (call api)
        if self.openai_api:  # the openai official api
            generations = self.model.generate(
                messages=prompt,
                functions=functions,
                function_call=function_call,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n_seqs,
                stop=stop,
            )

        elif self.anthropic_api:  # the openai official api (function calls in prompt)
            generations = self.model.generate(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n_seqs,
                stop=stop,
            )
        else:  # huggingface's models (# huggingface's models (local inference))
            inputs = self.tokenizer(
                [prompt],
                truncation=True,
                max_length=4096,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(self.model.device)
            stop = [] if stop is None else stop
            stop = list(
                set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])
            )  # In Llama \n is <0x0A>; In OPT \n is Ċ
            try:
                stop_token_ids = list(
                    set(
                        [
                            self.tokenizer._convert_token_to_id(stop_token)
                            for stop_token in stop
                        ]
                        + [self.model.config.eos_token_id]
                    )
                )
            except:  # some tokenizers don't have _convert_token_to_id function
                stop_token_ids = list(
                    set(
                        [
                            self.tokenizer.vocab.get(
                                stop_token, self.tokenizer.unk_token_id
                            )
                            for stop_token in stop
                        ]
                        + [self.model.config.eos_token_id]
                    )
                )

            if not self.tokenizer.unk_token_id:
                stop_token_ids.remove(self.tokenizer.unk_token_id)

            extra_params = {}
            if self.gen_config_pad_token_id is None:
                extra_params["pad_token_id"] = self.tokenizer.eos_token_id
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                num_return_sequences=n_seqs,
                eos_token_id=stop_token_ids,
                **extra_params
            )
            generations = [
                (self.tokenizer.decode(
                    output[inputs["input_ids"].size(1) :], skip_special_tokens=True
                ), len(output[inputs["input_ids"].size(1) :]))
                for output in outputs
            ]

        return generations


async def async_http_request(url, json_data, counter, semaphore):
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    # timeout = aiohttp.ClientTimeout(total=100)

    # reconnect when the request fails
    retry_interval_exp = 1
    interval = 0.5
    exp = 2
    patience = 3
    max_interval = 4
    async with semaphore:
        while True and retry_interval_exp <= 3:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=json_data) as response:
                        res = await response.json()
                break
            except Exception as e:
                wait_time = max(max_interval, interval * (exp**retry_interval_exp))
                logging.warning(f"Error: {e}")
                logging.warning(f"Retry in {wait_time} seconds")
                await asyncio.sleep(wait_time)
                retry_interval_exp += 1
    if counter is not None:
        counter.completed_requests += 1
    return res


"""
The wrapper for ChatCompletions
"""


class chat_completion(object):
    def __init__(
        self,
        model,
        api=False,
        host=None,
        port=None,
        system_message: str = "",
        system_template: str = "{system_message}",
        roles: List[str] = ["User", "Assistant"],
        offset: int = 20,
        colon: str = ": ",
        separators: List[str] = ["\n", "\n", "\n"],
        function_type: str = "json",
        function_call_prefix: str = "<function_call>",
        function_call_suffix: str = "</function_call>\n",
        verbose: bool = False,
        no_load_model: bool = False,
        counter=None,
        semaphore=None,
    ):
        self.verbose = verbose
        self.api = api

        model_name, model_port = get_model_name(model, no_load_model=no_load_model)
        print(f"\nUsing '{model_name}' for finding the model '{model}'.")
        self.base_url = f"http://{host}:{port if isinstance(port, int) else model_port}"
        print(f"\nInference Server: {self.base_url}\n")

        self.model_name = model_name
        self.counter = counter
        self.semaphore = semaphore

        if self.api:  # docker api calling
            assert isinstance(self.base_url, str) and self.base_url, f"Invalid base_url: {self.base_url}"
            self.url = f"{self.base_url}/v1/completions"
            # port = llm_configs[model]["port"]
            # self.url = f"http://127.0.0.1:{port}/generate"
        else:  # local inference
            self.model = LLM(model_name=model_name, no_load_model=no_load_model)

        # NOTE: if apply_chat_template is set True, make sure the tokenizer has chat template that contains system template
        self.apply_chat_template = False
        self.tokenizer = None

        # default templates
        if "gpt-3.5" in model or "gpt-4" in model:
            template_name = "chatgpt"
        elif "claude" in model:
            template_name = "claude"
        elif "llama-2" in model and "-chat" in model:
            template_name = "llama2"
        elif "llama-3" in model or model == "minicpm-2b-128k":
            template_name = "llama2"
            if not no_load_model:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_fast=False, trust_remote_code=True)
                if tokenizer.chat_template:
                    pass
                    self.apply_chat_template = True
                    self.tokenizer = tokenizer
                else:
                    print(f"Tokenizer of '{model_name}' does not have chat template, using template '{template_name}'.")
        elif "fnctod-llama2" in model:
            template_name = "llama2"
        elif "baichuan" in model and "-chat" in model:
            template_name = "baichuan2"
        elif "claude" in model:
            template_name = "claude"
        elif "vicuna" in model:
            template_name = "vicuna"
        elif "alpaca" in model:
            template_name = "alpaca"
        elif "baize" in model:
            template_name = "baize"
        elif "zephyr" in model:
            template_name = "zephyr"
        elif "openassistant" in model:
            template_name = "openassistant"
        else:
            raise ValueError(f"Invalid template for model: {model}")
        self.template = template_name

        # the conversation template
        self.conversation = Conversation(
            template_name=template_name,
            system_template=system_template,
            system_message=system_message,
            roles=roles,
            offset=offset,
            colon=colon,
            function_type=function_type,
            function_call_prefix=function_call_prefix,
            function_call_suffix=function_call_suffix,
            separators=separators,
            tokenizer=self.tokenizer,
        )

    async def complete(
        self,
        messages: List[Dict] = [],
        functions: List[Dict] = [],
        function_call: Dict = {},
        examples: List[Dict] = [],  # examples in the system instruction
        required: List[str] = ["function_call", "content"],
        temperature: float = 0.5,
        top_p: float = 0.5,
        max_tokens: int = 64,
        n_seqs: int = 1,
        stop: List[str] = ["\n\n"],  # ["\n\n", "###", "User", "Assistant", "Example"]
        regex: str = None,
    ) -> List[str]:
        in_out = {}

        if self.template == "chatgpt":
            # system messages
            system_message = ""
            for message in messages:
                if message["role"] == "system":
                    system_message = message["content"]
                    break

            # separate function calls and function outputs
            messages = self.conversation.separate_func_call_and_output(messages)
            examples = [self.conversation.separate_func_call_and_output(example) for example in examples]

            # construct system prompt with only examples
            system_prompt = self.conversation.get_prompt(
                system_message=system_message, examples=examples
            )

            # replace the original system message with the one with examples
            for message in messages:
                if message["role"] == "system":
                    message["content"] = system_prompt
                    break

            in_out["prompt"] = {
                "messages": deepcopy(messages),
                "functions": functions,
                "function_call": function_call,
            }

            t1 = time.time()
            outputs = self.model.generate(
                prompt=messages,
                functions=functions,
                function_call=function_call,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n_seqs=n_seqs,
                stop=stop,
            )
            t2 = time.time()
            duration = t2 - t1

            # input tokens
            input_tokens = 0
            for message in messages:
                input_tokens += len(self.model.tokenizer.encode(message["content"]))

            # output tokens
            for idx, output in enumerate(outputs):
                output["duration"] = duration
                output["input_tokens"] = input_tokens
                outputs[idx] = output

        else:  # local inference
            # system messages
            system_message = ""
            for message in messages:
                if message["role"] == "system":
                    system_message = message["content"]
                    break

            # construct prompt from the user utterances
            prompt = self.conversation.get_prompt(
                system_message=system_message,
                messages=messages,
                functions=functions,
                function_call=function_call,
                examples=examples,
            ).strip()

            if self.verbose:
                print(prompt)

            in_out["prompt"] = prompt

            t1 = time.time()
            if self.api:  # docker run
                # data = {
                #     "input": prompt,
                #     "temperature": temperature,
                #     "top_p": top_p,
                #     "max_tokens": max_tokens,
                #     "n_seqs": n_seqs,
                #     "stop": stop,
                # }
                # response = requests.post(self.url, json=data)
                # outputs = response.json()["generated_text"]

                if regex and prompt.strip().endswith('"arguments":'):
                    regex += r"\}"
                    # regex += r"\}" + self.conversation.function_call_suffix

                if regex:
                    in_out["regex"] = regex

                retry = True
                max_retry = max(len(examples), 1)
                while retry and max_retry > 0:
                    retry = False
                    data = {
                        "model": "",
                        "prompt": prompt,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "stop": stop if regex is None else None,
                        "regex": regex,
                        # "logprobs": 3,
                    }
                    response = await async_http_request(
                        self.url, data, self.counter, self.semaphore)
                    if "choices" in response:
                        input_tokens = response["usage"]["prompt_tokens"]
                        output_tokens = response["usage"]["completion_tokens"]
                        outputs = [(choice["text"], output_tokens) for choice in response["choices"]]
                    elif "meta_info" in response:
                        input_tokens = response["meta_info"]["prompt_tokens"]
                        output_tokens = response["meta_info"]["completion_tokens"]
                        outputs = [(response["text"], output_tokens)]
                    else:
                        outputs = [(None, None)]
                    if outputs[0][0] is None or (regex and re.fullmatch(regex, outputs[0][0]) is None):
                        print(f"Invalid response: {response}")
                        retry = True
                        max_retry -= 1
                        prompt = self.conversation.get_prompt(
                            system_message=system_message,
                            messages=messages,
                            functions=functions,
                            function_call=function_call,
                            examples=examples[:max_retry],
                        )
                        in_out["prompt"] = prompt
                        print(f"Retry with {min(len(examples), max_retry)} examples")
            else:
                input_tokens = len(self.model.tokenizer.encode(prompt))
                outputs = self.model.generate(
                    prompt=prompt,
                    functions=functions,
                    function_call=function_call,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n_seqs=n_seqs,
                    stop=stop,
                )
            t2 = time.time()
            duration = t2 - t1

            # parse the output
            parsed_outputs = []
            for output, output_tokens in outputs:
                if self.verbose:
                    print("Before parsing:", output)
                parsed_output = self.conversation.get_response(
                    output, function_call, required=required
                )
                if self.verbose:
                    print("After parsing:", parsed_output)
                parsed_outputs.append((parsed_output, output_tokens))
            outputs = parsed_outputs

            # cost summary
            for idx, (output, out_tokens) in enumerate(outputs):
                output["duration"] = duration
                output["input_tokens"] = input_tokens
                output["output_tokens"] = out_tokens
                outputs[idx] = output

        in_out["output"] = outputs

        # chatml format: {"content": "xxx", "function": {}}
        return outputs, in_out


"""
The wrapper for TextCompletions
"""


class text_completion(object):
    def __init__(self, model, api=False, verbose=False):
        self.verbose = verbose
        self.api = api
        model_name = llm_configs[model]["model_name"]
        self.model_name = model_name

        if self.api:  # api calling
            port = llm_configs[model]["port"]
            self.url = f"http://127.0.0.1:{port}/generate"
        else:  # local inference
            self.model = LLM(model_name=model_name)

    def to_openai_chat_completion(self, input) -> list[dict[str, str]]:
        messages = [
            {
                "role": "user",
                "content": input,
            }
        ]
        return messages

    def to_claude_completion(self, input) -> list[dict[str, str]]:
        messages = [f"{anthropic.HUMAN_PROMPT} {input}", f"{anthropic.AI_PROMPT}"]
        return "\n\n".join(messages)

    def get_prompt(self, input):
        # construct prompt from the user utterances
        if any([x in self.model_name for x in ["gpt-3.5", "gpt-4"]]):  # ChatML
            prompt = self.to_openai_chat_completion(input=input)
        elif any([x in self.model_name for x in ["claude"]]):  # ChatML
            prompt = self.to_claude_completion(input=input)
        else:  # str
            prompt = input
        return prompt

    def complete(
        self,
        input: str,
        temperature: float = 0.5,
        top_p: float = 1.0,
        max_tokens: int = 64,
        n_seqs: int = 1,
        stop: List[str] = ["\n", "\n\n", "User", "Example"],
    ):
        prompt = self.get_prompt(input)

        if self.verbose:
            print(prompt)

        if self.api:  # docker api call
            data = {
                "input": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n_seqs": n_seqs,
                "stop": stop,
            }
            response = requests.post(self.url, json=data)
            outputs = response.json()["generated_text"]
        else:
            outputs = self.model.generate(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n_seqs=n_seqs,
                stop=stop,
            )

        if self.verbose:
            print(outputs)

        return outputs
