#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
HOME_PATH = os.environ.get("HOME")

llm_configs = {
    "gpt-4-0409": {"model_name": "gpt-4-turbo-2024-04-09", "port": 8000},
    "gpt-3.5-0125": {"model_name": "gpt-3.5-turbo-0125", "port": 8001},
    "gpt-4-1106": {"model_name": "gpt-4-1106-preview", "port": 8002},
    "gpt-3.5-1106": {"model_name": "gpt-3.5-turbo-1106", "port": 8003},
    "claude-2.1": {"model_name": "claude-2.1", "port": 8004},
    "vicuna-7b-v1.3": {"model_name": "lmsys/vicuna-7b-v1.3", "port": 8005},
    "vicuna-13b-v1.3": {"model_name": "lmsys/vicuna-13b-v1.3", "port": 8006},
    "vicuna-7b-v1.5": {"model_name": "lmsys/vicuna-7b-v1.5", "port": 8007},
    "vicuna-13b-v1.5": {"model_name": "lmsys/vicuna-13b-v1.5", "port": 8008},
    "chatglm-6b": {"model_name": "THUDM/chatglm-6b", "port": 8009},
    "alpaca-7b": {"model_name": "chavinlo/alpaca-native", "port": 8010},
    # "llama-2-7b-chat": {"model_name": "meta-llama/Llama-2-7b-chat-hf", "port": 8012},
    "llama-2-7b-chat": {
        "model_name": "/home/share/models/Llama-2-7b-chat-hf",
        "model_path": f"{HOME_PATH}/models/Llama-2-7b-chat-hf", "port": 30000},
    # "llama-2-13b-chat": {"model_name": "meta-llama/Llama-2-13b-chat-hf", "port": 8011},
    "llama-2-13b-chat": {
        "model_name": "/home/share/models/Llama-2-13b-chat-hf",
        "model_path": f"{HOME_PATH}/models/Llama-2-13b-chat-hf", "port": 30030},
    # "llama-2-70b-chat": {"model_name": "meta-llama/Llama-2-70b-chat-hf", "port": 8013},
    "llama-2-70b-chat": {
        "model_name": "/home/share/models/Llama-2-70b-chat-hf",
        "model_path": f"{HOME_PATH}/models/Llama-2-70b-chat-hf", "port": 30060},
    "llama-3-8b": {
        "model_name": "/home/share/models/Meta-Llama-3-8B",
        "model_path": f"{HOME_PATH}/models/Meta-Llama-3-8B", "port": 30100},
    "llama-3-8b-instruct": {
        "model_name": "/home/share/models/Meta-Llama-3-8B-Instruct",
        "model_path": f"{HOME_PATH}/models/Meta-Llama-3-8B-Instruct", "port": 30130},
    "llama-3-70b-instruct": {
        "model_name": "/home/share/models/Meta-Llama-3-70B-Instruct",
        "model_path": f"{HOME_PATH}/models/Meta-Llama-3-70B-Instruct", "port": 30160},
    "llama-3.1-8b": {
        "model_name": "/home/share/models/Meta-Llama-3.1-8B",
        "model_path": f"{HOME_PATH}/models/Meta-Llama-3.1-8B", "port": 30200},
    "llama-3.1-8b-instruct": {
        "model_name": "/home/share/models/Meta-Llama-3.1-8B-Instruct",
        "model_path": f"{HOME_PATH}/models/Meta-Llama-3.1-8B-Instruct", "port": 30230},
    "llama-3.1-70b-instruct": {
        "model_name": "/home/share/models/Meta-Llama-3.1-70B-Instruct",
        "model_path": f"{HOME_PATH}/models/Meta-Llama-3.1-70B-Instruct", "port": 30260},
    "baize-7b": {"model_name": "project-baize/baize-lora-7B", "port": 8014},
    "baize-13b": {"model_name": "project-baize/baize-lora-13B", "port": 8015},
    "baichuan-7b-chat": {"model_name": "baichuan-inc/Baichuan2-7B-Chat", "port": 8016},
    "baichuan-13b-chat": {"model_name": "baichuan-inc/Baichuan2-13B-Chat", "port": 8017},
    "zephyr-7b-beta": {"model_name": f"{HOME_PATH}/models/zephyr-7b-beta", "port": 30900},
    "fnctod-llama2-13b-100": {
        "model_name": "./ckpt/hf_ckpt/llama-2-13b-chat-sft-llama2-100",
        "port": 8019,
    },
    "fnctod-llama2-13b-200": {
        "model_name": "./ckpt/hf_ckpt/llama-2-13b-chat-sft-llama2-200",
        "port": 8020,
    },
    "fnctod-llama2-13b-300": {
        "model_name": "./ckpt/hf_ckpt/llama-2-13b-chat-sft-llama2-300",
        "port": 8021,
    },
    "fnctod-llama2-13b-400": {
        "model_name": "./ckpt/hf_ckpt/llama-2-13b-chat-sft-llama2-400",
        "port": 8022,
    },
}


def get_model_name(model, no_load_model=False):
    model_name = llm_configs[model]["model_name"]
    port = llm_configs[model]["port"]
    model_path = llm_configs[model].get("model_path", None)
    if not os.path.exists(model_name) and not no_load_model and model_path:
        if os.path.exists(model_path):
            model_name = model_path
        else:
            raise ValueError(f"Invalid model name and path: '{model_name}' and '{model_path}'")
    return model_name, port


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    model_name, port = get_model_name(args.model)
    # give the result to shell
    print(f"--model-path {model_name} --port {port}")
