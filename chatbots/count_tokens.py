# coding: utf-8
import os
import json
import argparse
import tiktoken
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from openai_token_counter import openai_token_counter

from chatbots.configs import llm_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument("--eval_message_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-3.5-0125")
    parser.add_argument("--inspect_dst_empty", action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    print(args)

    model_name = llm_configs[args.model]["model_name"]
    tokenizer = tiktoken.encoding_for_model(model_name)

    with open(args.eval_message_file, "r") as f:
        eval_messages = json.load(f)

    eval_metrics_file = args.eval_message_file.replace("messages", "metrics")

    tasks = ["domain", "dst", "nlg"]
    valid_turns, valid_dials = 0, 0
    usage = defaultdict(lambda : defaultdict(int))
    details = defaultdict(list)

    for dial_id, eval_message in tqdm(eval_messages.items(), desc="Counting tokens"):
        task_flag = defaultdict(lambda : False)
        for idx, turn in enumerate(eval_message):
            valid_turn = False
            for task in tasks:
                if task in turn:
                    valid_turn = True
                    task_flag[task] = True
                    if isinstance(turn[task], dict):
                        logs = [turn[task]]
                    elif isinstance(turn[task], list):
                        logs = turn[task]
                    else:
                        raise ValueError(f"Invalid type for turn {idx} of dial {dial_id}:\n{turn}\n")
                    for log in logs:
                        prompt = log["prompt"]
                        output = log["output"][0]

                        counter_params = {
                            "messages": prompt["messages"],
                            "model": model_name,
                        }
                        if "functions" in prompt:
                            counter_params["functions"] = prompt["functions"]
                        if "function_call" in prompt:
                            counter_params["function_call"] = prompt["function_call"]
                        prompt_tokens = openai_token_counter(**counter_params)
                        usage[task]["prompt_tokens"] += prompt_tokens
                        details[f"{task}: prompt_tokens"].append(prompt_tokens)
                        try:
                            completion_text = output.get("content", None)
                            if not completion_text:
                                completion_text = output["function_call"]["arguments"]
                        except KeyError:
                            print(f"Error in turn {idx} of dial {dial_id}:\n{turn}\n")
                            exit()
                        completion_tokens = len(tokenizer.encode(completion_text))
                        usage[task]["completion_tokens"] += completion_tokens
                        details[f"{task}: completion_tokens"].append(completion_tokens)
                    usage[task]["num_turns"] += 1
                elif task == "dst" and args.inspect_dst_empty:
                    print(f"Empty DST in turn {idx} of dial {dial_id}:\n{turn}\n")
            valid_turns += int(valid_turn)
        valid_dials += int(any(task_flag.values()))
        for task in tasks:
            if task_flag[task]:
                usage[task]["num_dials"] += 1

    usage_df = pd.DataFrame.from_dict(usage, orient="index")
    usage_df.loc["all"] = usage_df.sum(axis=0)
    usage_df.loc["all", "num_turns"] = valid_turns
    usage_df.loc["all", "num_dials"] = valid_dials
    usage_df["p_t/turn"] = usage_df["prompt_tokens"] / usage_df["num_turns"]
    usage_df["p_t/dial"] = usage_df["prompt_tokens"] / usage_df["num_dials"]
    usage_df["c_t/turn"] = usage_df["completion_tokens"] / usage_df["num_turns"]
    usage_df["c_t/dial"] = usage_df["completion_tokens"] / usage_df["num_dials"]


    details_df = pd.concat({k: pd.Series(v).describe() for k, v in details.items()}, axis=1)

    print(usage_df)
    print(details_df.round(2))
    print(usage_df.to_json(indent=4))

    if os.path.isfile(eval_metrics_file):
        with open(eval_metrics_file, "r") as f:
            eval_metrics = json.load(f)
        eval_metrics["token_usage"] = usage_df.to_dict()
    else:
        raise FileNotFoundError(f"File not found: {eval_metrics_file}")
    with open(eval_metrics_file, "w") as f:
        json.dump(eval_metrics, f, indent=4)


