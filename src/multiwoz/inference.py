#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import time
import random
import argparse
import copy
import logging
from tqdm import tqdm
from functools import partial

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.multiwoz.utils import *
from src.multiwoz.utils.config import *
from src.multiwoz.utils.reader import *
from src.multiwoz.utils.utils import (
    paser_dict_to_bs,
    paser_bs_to_dict,
)
from src.multiwoz.postprocess import (
    unzip_session_data,
    zip_session_data,
    get_data_split,
    load_schema,
    load_examples,
)
from src.utils import *
from chatbots.utils import *
from chatbots.llm import *
from src.multiwoz.schema2function import schema2function
from src.multiwoz.independent_inference import generate_dst

EXPERIMENT_DOMAINS = ["[taxi]", "[train]", "[attraction]", "[hotel]", "[restaurant]"]
domain2function_mapping = {
    "hotel": "find_book_hotel",
    "train": "find_book_train",
    "attraction": "find_attraction",
    "restaurant": "find_book_restaurant",
    "taxi": "book_taxi",
    "hospital": "find_hospital",
    "police": "police",
}

domain2user_inform_mapping = {
    "hotel": "track_hotel_inform_by_user",

}


def parse_dst_response(assistant_message, domain2mapping, schema, user_goal):
    turn_domain = None
    current_function = None
    turn_bs_dict = {}
    if "function_call" in assistant_message:
        function_call = assistant_message["function_call"]
        try:
            # step 1: get the domain
            if "name" in function_call:
                pred_function = function_call["name"].strip()
            elif "function" in function_call:
                pred_function = function_call["function"].strip()

            for d, f in domain2mapping.items():
                if pred_function == f:
                    turn_domain = "[" + d + "]"
                    break
            assert turn_domain is not None

            # step 2: get the current function
            for service in schema:
                if service["service_name"] == turn_domain[1:-1]:
                    current_function = schema2function(
                        service,
                        template=ChatCompletion.template,
                        rename_mapping=domain2function_mapping,
                    )
            assert current_function is not None

            # step 3: get the arguments
            turn_bs_dict_gen = function_call["arguments"]
            if isinstance(turn_bs_dict_gen, str):
                turn_bs_dict_gen = json.loads(turn_bs_dict_gen)
            assert isinstance(turn_bs_dict_gen, dict)
        except:
            print("Can not parse:", function_call)
            turn_bs_dict_gen = {}

        # update user goal
        if turn_domain in EXPERIMENT_DOMAINS:
            if turn_domain not in user_goal:
                user_goal[turn_domain] = {}
            if turn_domain not in turn_bs_dict:
                turn_bs_dict[turn_domain] = {}

            # clean the generation and update user goal
            for slot, value in turn_bs_dict_gen.items():
                slot = slot.strip().lower()
                value = str(value).strip().lower()
                # only update the valid generations
                if slot in current_function["parameters"]["properties"]:
                    if "enum" not in current_function["parameters"]["properties"][slot]:
                        user_goal[turn_domain][slot] = value
                        turn_bs_dict[turn_domain][slot] = value
                    elif value in current_function["parameters"]["properties"][slot]["enum"]:
                        user_goal[turn_domain][slot] = value
                        turn_bs_dict[turn_domain][slot] = value

    return turn_domain, turn_bs_dict


def prepare_evaluation(data):
    eval_data = {}
    in_out_data = {}
    for dial_id, turns in data.items():
        eval_turns = []
        in_out_turns = []
        for turn in turns:
            eval_turn = copy.deepcopy(turn)

            for key in [
                "dial_id",
                "turn_num",
                "user",
                "resp",
                "nodelx_resp",
                "resp_gen",
                "dspn",
                "dspn_gen",
                "bsdx",
                "bspn",
                "bspn_gen",
                "bspn_dict",
                "bspn_dict_gen",
                "turn_bspn",
                "turn_bspn_gen",
                "turn_bspn_dict",
                "turn_bspn_dict_gen",
                "db",
                "db_gen",
                "aspn",
                "aspn_gen",
                "aspn_dict",
                "aspn_dict_gen",
                "all_domains",
            ]:
                if key in turn:
                    eval_turn[key] = turn[key]
                elif "dict" in key:
                    eval_turn[key] = {}
                else:
                    eval_turn[key] = ""
            eval_turns.append(eval_turn)

            in_out_turn = {}
            for key in [
                "dial_id",
                "turn_num",
                "user",
                "resp",
                "nodelx_resp",
                "dspn",
                "bspn_dict",
                "turn_bspn_dict",
                "all_domains",
            ]:
                in_out_turn[key] = turn[key]
            in_out_turns.append(in_out_turn)
        eval_data[dial_id] = eval_turns
        in_out_data[dial_id] = in_out_turns
    return eval_data, in_out_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="2.1",
        choices=["2.0", "2.1", "2.2", "2.3"],
    )  #
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )  #
    parser.add_argument(
        "--n_eval", type=int, default=100, help="number of evaluated dialogues"
    )  #

    parser.add_argument("--ind_dst", type=str2bool, default=False,
                        help="whether to decode belief state slot by slot")
    parser.add_argument("--divide_inform_confirm", type=str2bool, default=False,
                        help="whether to divide inform values and confirm values when tracking belief states")
    parser.add_argument("--gen_state_channel", type=str, default="bspn_gen",
                        choices=["bspn_gen", "inform_bspn_gen", "confirm_bspn_gen"])
    parser.add_argument("--track_slot_status", type=str2bool, default=False,
                        help="whether to track slot status")
    parser.add_argument("--dst_result_path", type=str, default=None,
                        help="path to the dst result file, which is combined with slot status to generate the final result")

    # parser.add_argument('--delx', type=str2bool, default=False, help='whether to use multiple functions') #
    parser.add_argument(
        "--ref_domain",
        type=str2bool,
        default=False,
        help="whether to use oracle domain",
        choices=[False, True],
    )  #
    parser.add_argument(
        "--ref_bs",
        type=str2bool,
        default=False,
        help="whether to use oracle belief states",
        choices=[False, True],
    )  #

    parser.add_argument(
        "--multi_domain",
        type=str2bool,
        default=False,
        help="whether to use multiple functions or a single one",
    )  #
    parser.add_argument(
        "--add_prev",
        type=str2bool,
        default=True,
        help="whether to use intermediate steps in conversation context, i.e., previous turns",
    )  #
    parser.add_argument(
        "--function_type", type=str, default="json", choices=["json", "text"]
    )  #

    parser.add_argument(
        "--dst_nshot",
        type=int,
        default=0,
        help="number of demonstration examples for dst",
    )  #
    parser.add_argument(
        "--nlg_nshot",
        type=int,
        default=0,
        help="number of demonstration examples for nlg",
    )  #
    parser.add_argument(
        "--task", type=str, default="dst", choices=["dst", "nlg", "e2e"]
    )  #

    # for evaluation
    parser.add_argument("--model", type=str, default="gpt-3.5-0125")  #
    parser.add_argument("--temperature", type=float, default=0.3)  #
    parser.add_argument("--top_p", type=float, default=0.2)  #

    parser.add_argument(
        "--generate", action="store_true", help="whether to continue the reference"
    )  #
    parser.add_argument(
        "--debug", action="store_true", help="whether to print out message"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="whether to use llm api or local inference",
    )
    parser.add_argument(
        "--eval_valid",
        action="store_true",
        help="whether to evaluate only on valid turns",
    )  #

    args, unknown = parser.parse_known_args()
    print(args)

    # fixed setups
    if args.model in ["gpt-3.5", "gpt-4"]:
        assert args.dst_nshot == 0
        assert args.add_prev == False
        assert args.function_type == "json"
    if args.multi_domain:
        assert args.dst_nshot == 0
        assert not args.ref_domain
    if args.task == "nlg":
        assert args.ref_domain == True
        assert args.ref_bs == True
    elif args.task == "dst" or args.task == "e2e":
        assert args.ref_bs == False

    # load configuration file and reader (for database query)
    data_prefix = "./data/multiwoz/data/"
    if args.dataset_version == "2.0":
        cfg = Config20(data_prefix)
    elif args.dataset_version == "2.1":
        cfg = Config21(data_prefix)
    elif args.dataset_version == "2.2":
        cfg = Config22(data_prefix)
    elif args.dataset_version == "2.3":
        cfg = Config23(data_prefix)
    reader = MultiWozReader(tokenizer=None, cfg=cfg, data_mode=args.split)

    # load schema, examples, data
    train_data, val_data, test_data = get_data_split(
        dataset_version=args.dataset_version,
        reader=reader,
        n_train=10000,
        n_val=args.n_eval,
        n_test=args.n_eval,
        return_list=False,
    )
    schema = load_schema(args.dataset_version)
    # get domain descriptions
    domain2desc = {}
    for service in schema:
        dn = service["service_name"]
        if f"[{dn}]" in EXPERIMENT_DOMAINS:
            domain2desc[dn] = service["description"] + "."

    examples = load_examples(args.dataset_version, train_data)
    div_examples = json.load(open(f"./src/multiwoz/divide_examples.json", "r"))

    if args.split == "val":
        eval_data = val_data
    elif args.split == "test":
        eval_data = test_data
    eval_data, eval_in_out = prepare_evaluation(eval_data)


    # save data path
    data_prefix = f"./outputs/multiwoz{args.dataset_version}"
    if not os.path.exists(data_prefix):
        os.makedirs(data_prefix, exist_ok=True)

    if args.ind_dst:
        config_prefix = f"{data_prefix}/{args.split}{args.n_eval}-multi{args.multi_domain}-ref{args.ref_domain}-prev{args.add_prev}-{args.function_type}-ind_dst{args.dst_nshot}shot"
    elif args.divide_inform_confirm:
        config_prefix = f"{data_prefix}/{args.split}{args.n_eval}-multi{args.multi_domain}-ref{args.ref_domain}-prev{args.add_prev}-{args.function_type}-div_dst{args.dst_nshot}shot"
    elif args.track_slot_status:
        config_prefix = f"{data_prefix}/{args.split}{args.n_eval}-multi{args.multi_domain}-ref{args.ref_domain}-prev{args.add_prev}-{args.function_type}-status_dst{args.dst_nshot}shot"
    else:
        config_prefix = f"{data_prefix}/{args.split}{args.n_eval}-multi{args.multi_domain}-ref{args.ref_domain}-prev{args.add_prev}-{args.function_type}-dst{args.dst_nshot}shot"

    # save dst result path
    dst_eval_result_path = f"{config_prefix}-{args.model}.json"
    dst_eval_messages_path = f"{config_prefix}-nlg{args.nlg_nshot}shot-{args.model}-messages.json"

    # save error path and metrics path
    if not args.divide_inform_confirm or args.gen_state_channel == "bspn_gen":
        dst_eval_error_path = f"{config_prefix}-{args.model}-errors.json"
        eval_metrics_path = f"{config_prefix}-nlg{args.nlg_nshot}shot-{args.model}-metrics.json"
    else:
        dst_eval_error_path = f"{config_prefix}-{args.model}-{args.gen_state_channel}-errors.json"
        eval_metrics_path = f"{config_prefix}-nlg{args.nlg_nshot}shot-{args.model}-{args.gen_state_channel}-metrics.json"

    # save nlg result path
    if args.task == "e2e":
        nlg_eval_result_path = f"{config_prefix}-nlg{args.nlg_nshot}shot-{args.model}.json"
    elif args.task == "nlg":
        nlg_eval_result_path = f"{data_prefix}/{args.split}{args.n_eval}-multi{args.multi_domain}-prev{args.add_prev}-{args.function_type}-nlg{args.nlg_nshot}shot-{args.model}.json"
        eval_metrics_path = f"{data_prefix}/{args.split}{args.n_eval}-multi{args.multi_domain}-prev{args.add_prev}-{args.function_type}-nlg{args.nlg_nshot}shot-{args.model}-metrics.json"


    eval_messages_path = dst_eval_messages_path
    print(f"File '{dst_eval_messages_path}' exists: {os.path.exists(dst_eval_messages_path)}")
    if os.path.exists(eval_messages_path):
        with open(eval_messages_path, "r") as file:
            evaluated_in_out = json.load(file)
    else:
        evaluated_in_out = {}

    print(f"File '{dst_eval_result_path}' exists: {os.path.exists(dst_eval_result_path)}")

    # load existing data
    if args.task == "dst":
        eval_result_path = dst_eval_result_path
        if os.path.exists(dst_eval_result_path):
            with open(dst_eval_result_path, "r") as file:
                evaluated_data = json.load(file)
        elif not args.generate:
            raise NotImplementedError
        else:
            evaluated_data = {}
    elif args.task == "e2e":
        eval_result_path = nlg_eval_result_path
        if os.path.exists(nlg_eval_result_path):
            with open(nlg_eval_result_path, "r") as file:
                evaluated_data = json.load(file)
        elif os.path.exists(dst_eval_result_path):
            with open(dst_eval_result_path, "r") as file:
                evaluated_data = json.load(file)
        else:
            # raise NotImplementedError
            evaluated_data = {}
    elif args.task == "nlg":
        eval_result_path = nlg_eval_result_path
        if os.path.exists(nlg_eval_result_path):
            with open(nlg_eval_result_path, "r") as file:
                evaluated_data = json.load(file)
        else:
            evaluated_data = {}

    # add the data that has been evaluated
    for dial_id, dp in evaluated_data.items():
        eval_data[dial_id] = dp
    for dial_id, dp in evaluated_in_out.items():
        for idx, turn in enumerate(dp):
            assert eval_in_out[dial_id][idx]["turn_num"] == turn["turn_num"]
            eval_in_out[dial_id][idx].update(turn)

    # evaluation
    if args.generate:
        # load the model
        ChatCompletion = chat_completion(
            model=args.model,
            function_type=args.function_type,
            function_call_prefix=fc_prefix,
            function_call_suffix=fc_suffix,
            verbose=args.verbose,
        )

        save_interval = 1 if args.ind_dst else 10

        d_pbar = tqdm(total=len(eval_data), desc=f"Evaluation {args.split}")
        num_turns = sum([len(turns) for turns in eval_data.values()])
        num_completed_turns = 0

        for didx, (dial_id, eval_turns) in enumerate(eval_data.items()):
            user_goal = {}
            inform_user_goal = {}
            confirm_user_goal = {}

            for idx, eval_turn in enumerate(eval_turns):
                turn_st = time.time()
                """Step 1: Domain prediction"""
                if args.multi_domain:
                    turn_domain = None
                elif args.ref_domain:
                    turn_domain = eval_turn["dspn"]
                    eval_turn["dspn_gen"] = turn_domain
                elif eval_turn["dspn_gen"]:
                    turn_domain = eval_turn["dspn_gen"]
                else:  # inference
                    messages = []
                    dp_instruction = (
                        "You are a task-oriented assistant. "
                        "Your role is to determine which domain the user is seeking information about or attempting to make a booking in during each turn of the conversation. "
                        "Select the most relevant domain from the following options: [restaurant], [hotel], [taxi], [train], [hospital], [police], [attraction]. "
                        "If the user's inquiry does not align with a specific domain, use: [general]. "
                        "Note that the [attraction] domain encompasses various categories, including architecture, boat, cinema, college, concert hall, "
                        "entertainment, museum, sports activities, nightclub, park, swimming pool, and theatre."
                    )
                    messages.append({"role": "system", "content": dp_instruction})

                    # examples
                    dp_examples = [
                        [
                            (
                                "hi, could i find some museum in the center of the town ?",
                                domain_prefix
                                + "[attraction]"
                                + domain_suffix
                                + "The railroad museum would be nice for you .",
                            ),
                            (
                                "great , and i also want to book a taxi to leave the attraction by 08:00 . get contact number and car type .",
                                domain_prefix + "[taxi]" + domain_suffix,
                            ),
                        ],
                        [
                            (
                                "please find me a place to dine that serves vegetarian food .",
                                domain_prefix
                                + "[restaurant]"
                                + domain_suffix
                                + "i found a cheap one that serves korea food .",
                            ),
                        ],
                        [
                            (
                                "i am also looking for place -s to go in town . i would love for it to be sports related .",
                                domain_prefix
                                + "[attraction]"
                                + domain_suffix
                                + "we have 4 swimming pool location -s . what do you think about those ?",
                            ),
                            (
                                "okay, thank you . have a good day !",
                                domain_prefix
                                + "[general]"
                                + domain_suffix
                                + "you too, bye !",
                            ),
                        ],
                        [
                            (
                                "do you have any place -s to stay in the west that include free parking ?",
                                domain_prefix
                                + "[hotel]"
                                + domain_suffix
                                + "yes, what price range are you looking for ?",
                            )
                        ],
                    ]
                    example_messages = []
                    for example in dp_examples:
                        example_message = []
                        for turn in example:
                            user, resp = turn
                            example_message.extend(
                                [
                                    {"role": "user", "content": user},
                                    {"role": "assistant", "content": resp},
                                ]
                            )
                        example_messages.append(example_message)

                    # history message
                    for prev_turn in eval_turns[:idx]:
                        usr = prev_turn["user"]
                        resp = prev_turn["nodelx_resp"]
                        domain_gen = prev_turn["dspn_gen"]
                        resp = domain_prefix + domain_gen + domain_suffix + resp
                        messages.append({"role": "user", "content": usr})
                        messages.append({"role": "assistant", "content": resp})

                    # current turn
                    usr = eval_turn["user"]
                    messages.append({"role": "user", "content": usr})
                    resp_prefix = domain_prefix + "["
                    messages.append({"role": "assistant", "content": resp_prefix})

                    # predict domain
                    chat_response, in_out = ChatCompletion.complete(
                        messages=messages,
                        examples=example_messages,
                        required=["content"],
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=8,
                        n_seqs=1,
                    )
                    dspn_gen = chat_response[0]["content"]

                    turn_domain = ""
                    for d in [
                        "taxi",
                        "attraction",
                        "hotel",
                        "restaurant",
                        "train",
                        "hotel",
                        "police",
                        "general",
                    ]:
                        if d in dspn_gen:
                            turn_domain = "[" + d + "]"
                            eval_turn["dspn_gen"] = turn_domain
                            break
                    for d in [
                        "architecture",
                        "boat",
                        "cinema",
                        "college",
                        "concert hall",
                        "entertainment",
                        "museum",
                        "sports",
                        "nightclub",
                        "park",
                        "swimming pool",
                        "theatre",
                    ]:
                        if d in dspn_gen:
                            turn_domain = "[attraction]"
                            eval_turn["dspn_gen"] = turn_domain
                            break

                    eval_in_out[dial_id][idx]["domain"] = {
                        "prompt": in_out["prompt"],
                        "output": in_out["output"],
                        "dspn": eval_turn["dspn"],
                        "dspn_gen": turn_domain
                    }

                    if not turn_domain:
                        print("Can not parse:", dspn_gen)
                        continue

                    if args.debug:
                        print("Oracle dspn:", eval_turn["dspn"])
                        print("Generated dspn:", turn_domain)
                        _ = input()

                """
                Find the domain schema, examples for the prompt construction
                """
                functions = []
                current_function = None
                for domain in EXPERIMENT_DOMAINS:
                    for service in schema:
                        if service["service_name"] == domain[1:-1]:
                            function = schema2function(
                                service,
                                template=ChatCompletion.template,
                                rename_mapping=domain2function_mapping,
                            )
                            if args.multi_domain:
                                functions.append(function)
                            elif domain == turn_domain:  # only the current turn domain
                                current_function = function
                                functions.append(current_function)
                            break

                """
                Step 2: Dialogue State Tracking (DST)
                """
                if args.ref_bs:
                    user_goal = paser_bs_to_dict(eval_turn["bspn"])
                    eval_turn["bspn_gen"] = eval_turn["bspn"]
                elif eval_turn["bspn_gen"]:
                    user_goal = paser_bs_to_dict(eval_turn["bspn_gen"])
                else:  # inference
                    """
                    Construct prompt for inference
                    """
                    messages = []
                    # system instruction
                    system_messages = [random.choice(tod_instructions)]
                    system_messages.extend(tod_notes)
                    system_message = "\n".join(system_messages)
                    messages.append({"role": "system", "content": system_message})

                    # select examples for the current domain
                    if args.divide_inform_confirm:
                        dst_examples = div_examples
                    else:
                        # dst_examples = examples
                        dst_examples = div_examples
                    if not args.multi_domain and turn_domain in dst_examples:
                        domain_examples = dst_examples[turn_domain][: args.dst_nshot]
                    else:
                        domain_examples = []

                    # previous example conversations (NODELX)
                    example_messages = []
                    for bs_example in domain_examples:
                        example_message = []
                        for turn in bs_example:
                            domain = turn["dspn"]
                            user = turn["user"]
                            resp = turn["nodelx_resp"]
                            bs_dict = turn["bspn_dict"]
                            db_num = turn["db"]

                            # add user message
                            example_message.append({"role": "user", "content": user})
                            # add assistant message
                            if domain in EXPERIMENT_DOMAINS:
                                if domain in bs_dict and args.add_prev:
                                    function_call_dict = {
                                        "function": domain2function_mapping[
                                            domain[1:-1]
                                        ],
                                        "arguments": bs_dict[domain],
                                    }
                                    example_message.append(
                                        {
                                            "role": "assistant",
                                            "content": resp,
                                            "function_call": function_call_dict,
                                        }
                                    )
                                else:
                                    example_message.append(
                                        {"role": "assistant", "content": resp}
                                    )
                            else:
                                example_message.append(
                                    {"role": "assistant", "content": resp}
                                )
                        example_messages.append(example_message)

                    # history message in the current conversation
                    for prev_turn in eval_turns[:idx]:
                        usr = prev_turn["user"]
                        resp = prev_turn["nodelx_resp"]
                        prev_domain = prev_turn["dspn_gen"]
                        prev_bs_dict = prev_turn["bspn_dict_gen"]

                        # add user message
                        messages.append({"role": "user", "content": usr})
                        # add assistant message
                        assistant_message = {"role": "assistant", "content": resp}
                        if args.add_prev:
                            if args.multi_domain:
                                if prev_domain in prev_bs_dict:
                                    function_call_dict = {
                                        "function": domain2function_mapping[
                                            prev_domain[1:-1]
                                        ],
                                        "arguments": prev_bs_dict[prev_domain],
                                    }
                                    assistant_message[
                                        "function_call"
                                    ] = function_call_dict
                            else:
                                if turn_domain in prev_bs_dict:
                                    function_call_dict = {
                                        "function": domain2function_mapping[
                                            turn_domain[1:-1]
                                        ],
                                        "arguments": prev_bs_dict[turn_domain],
                                    }
                                    assistant_message[
                                        "function_call"
                                    ] = function_call_dict
                        messages.append(assistant_message)

                    # current turn
                    usr = eval_turn["user"]
                    messages.append({"role": "user", "content": usr})

                    # generate dst
                    if args.ind_dst:
                        completion_func = partial(
                            ChatCompletion.complete,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=10,
                            n_seqs=1,
                        )
                        state, in_out = generate_dst(
                            eval_turn,
                            messages,
                            schema,
                            EXPERIMENT_DOMAINS if args.multi_domain else [turn_domain],
                            completion_func,
                        )
                        if args.multi_domain:
                            user_goal = state
                        else:
                            user_goal.update(state)
                    elif args.track_slot_status:
                        status_messages, status_functions, domain2mapping = adapt_track_slot_status(
                            messages, functions, domain2function_mapping, domain2desc)
                        chat_response, in_out = ChatCompletion.complete(
                            messages=status_messages,
                            functions=status_functions,
                            function_call={"name": status_functions[0]["name"]}
                            if current_function
                            else {},
                            required=["function_call"],
                            examples=example_messages,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            # temperature=0,
                            # top_p=0,
                            max_tokens=128,
                            n_seqs=1,
                        )
                        turn_status_dict_gen = parse_status_response(
                            chat_response[0], domain2mapping, status_functions, user_goal, EXPERIMENT_DOMAINS
                        )
                        in_out["turn_status_dict_gen"] = turn_status_dict_gen
                    else:
                        if args.divide_inform_confirm:
                            in_out = []
                            # first track slot values informed by user
                            inform_sys_msg, inform_functions, domain2mapping = adapt_inform_or_confirm(
                                "inform", functions, domain2function_mapping)
                            messages[0]["content"] = inform_sys_msg
                            example_messages = build_divide_examples(
                                "inform", domain_examples, domain2mapping, EXPERIMENT_DOMAINS)
                            chat_response, in_out_ = ChatCompletion.complete(
                                messages=messages,
                                functions=inform_functions,
                                function_call={"name": inform_functions[0]["name"]}
                                if current_function
                                else {},
                                required=["function_call"],
                                examples=example_messages,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=128,
                                n_seqs=1,
                            )
                            td, turn_inform_bs_dict_gen = parse_dst_response(
                                chat_response[0], domain2mapping, schema, user_goal
                            )
                            in_out_["turn_inform_bs_dict_gen"] = turn_inform_bs_dict_gen
                            in_out.append(in_out_)
                            if td:
                                if turn_domain:
                                    assert turn_domain == td
                                turn_domain = td

                            # then track slot values confirmed by user
                            confirm_sys_msg, confirm_functions, domain2mapping = adapt_inform_or_confirm(
                                "confirm", functions, domain2function_mapping)
                            example_messages = build_divide_examples(
                                "confirm", domain_examples, domain2mapping, EXPERIMENT_DOMAINS)
                            messages[0]["content"] = confirm_sys_msg
                            chat_response, in_out_ = ChatCompletion.complete(
                                messages=messages,
                                functions=confirm_functions,
                                function_call={"name": confirm_functions[0]["name"]}
                                if current_function
                                else {},
                                required=["function_call"],
                                examples=example_messages,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=128,
                                n_seqs=1,
                            )
                            td, turn_confirm_bs_dict_gen = parse_dst_response(
                                chat_response[0], domain2mapping, schema, user_goal
                            )
                            in_out_["turn_confirm_bs_dict_gen"] = turn_confirm_bs_dict_gen
                            in_out.append(in_out_)
                            if td:
                                if turn_domain:
                                    assert turn_domain == td
                                turn_domain = td
                        else:
                            example_messages = build_divide_examples(
                                None, domain_examples, domain2function_mapping, EXPERIMENT_DOMAINS)
                            chat_response, in_out = ChatCompletion.complete(
                                messages=messages,
                                functions=functions,
                                function_call={"name": current_function["name"]}
                                if current_function
                                else {},
                                required=["function_call"],
                                examples=example_messages,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=128,
                                n_seqs=1,
                            )
                            assistant_message = chat_response[0]
                            td, turn_bs_dict_gen = parse_dst_response(
                                assistant_message, domain2function_mapping, schema, user_goal
                            )
                            if td:
                                if turn_domain:
                                    assert turn_domain == td
                                turn_domain = td
                            '''
                            if "function_call" in assistant_message:
                                function_call = assistant_message["function_call"]
                                try:
                                    # step 1: get the domain
                                    if "name" in function_call:
                                        pred_function = function_call["name"].strip()
                                    elif "function" in function_call:
                                        pred_function = function_call["function"].strip()

                                    for d, f in domain2function_mapping.items():
                                        if pred_function == f:
                                            turn_domain = "[" + d + "]"
                                            break
                                    assert turn_domain is not None

                                    # step 2: get the current function
                                    for service in schema:
                                        if service["service_name"] == turn_domain[1:-1]:
                                            current_function = schema2function(
                                                service,
                                                template=ChatCompletion.template,
                                                rename_mapping=domain2function_mapping,
                                            )
                                    assert current_function is not None

                                    # step 3: get the arguments
                                    turn_bs_dict_gen = function_call["arguments"]
                                    if isinstance(turn_bs_dict_gen, str):
                                        turn_bs_dict_gen = json.loads(turn_bs_dict_gen)
                                    assert isinstance(turn_bs_dict_gen, dict)
                                except:
                                    print("Can not parse:", function_call)
                                    turn_bs_dict_gen = {}

                                # update user goal
                                if turn_domain in EXPERIMENT_DOMAINS:
                                    if turn_domain not in user_goal:
                                        user_goal[turn_domain] = {}

                                    # clean the generation and update user goal
                                    for slot, value in turn_bs_dict_gen.items():
                                        slot = slot.strip().lower()
                                        value = str(value).strip().lower()
                                        # only update the valid generations
                                        if slot in current_function["parameters"]["properties"]:
                                            if (
                                                "enum"
                                                not in current_function["parameters"][
                                                    "properties"
                                                ][slot]
                                            ):
                                                user_goal[turn_domain][slot] = value
                                            elif (
                                                value
                                                in current_function["parameters"]["properties"][
                                                    slot
                                                ]["enum"]
                                            ):
                                                user_goal[turn_domain][slot] = value
                            '''

                    # record
                    # print(user_goal)
                    bspn_gen = paser_dict_to_bs(user_goal)
                    eval_turn["bspn_gen"] = bspn_gen  # for evaluation
                    eval_turn["bspn_dict_gen"] = copy.deepcopy(user_goal)

                    if args.ind_dst or args.divide_inform_confirm or args.track_slot_status:
                        eval_in_out[dial_id][idx]["dst"] = in_out
                        eval_in_out[dial_id][idx]["bspn_dict"] = eval_in_out[dial_id][idx].pop("bspn_dict")
                        eval_in_out[dial_id][idx]["bspn_dict_gen"] = eval_turn["bspn_dict_gen"]
                    else:
                        eval_in_out[dial_id][idx]["dst"] = {
                            "prompt": in_out["prompt"],
                            "output": in_out["output"],
                            "bspn_dict": eval_turn["bspn_dict"],
                            "bspn_dict_gen": eval_turn["bspn_dict_gen"]
                        }

                    if save_interval <= 0:
                        with open(eval_result_path, "w") as file:
                            json.dump(eval_data, file, indent=4)
                        with open(eval_messages_path, "w") as file:
                            json.dump(eval_in_out, file, indent=4)

                    # debug
                    if args.verbose:
                        duration = time.time() - turn_st
                        print("=" * 25 + f" {dial_id}-{idx} (consume {duration:.2f} secs)" + "=" * 25)
                        print(f"User: {eval_turn['user']}")
                        print(f"Oracle domain:", eval_turn["dspn"])
                        print(f"Detect domain:", turn_domain)
                        print(f"Oracle bspn: {eval_turn['bspn']}")
                        print(f"Generated bspn: {bspn_gen}")

                    if args.debug:
                        _ = input()

                if args.divide_inform_confirm:
                    turn_inform_bs_dict_gen = eval_in_out[dial_id][idx]["dst"][0]["turn_inform_bs_dict_gen"]
                    dict_update(inform_user_goal, copy.deepcopy(turn_inform_bs_dict_gen))
                    eval_turn["inform_bspn_gen"] = paser_dict_to_bs(inform_user_goal)
                    turn_confirm_bs_dict_gen = eval_in_out[dial_id][idx]["dst"][1]["turn_confirm_bs_dict_gen"]
                    dict_update(confirm_user_goal, copy.deepcopy(turn_confirm_bs_dict_gen))
                    eval_turn["confirm_bspn_gen"] = paser_dict_to_bs(confirm_user_goal)

                """
                Step 3: Response generation (NLG) (DELX)
                """
                if eval_turn["resp_gen"]:
                    resp_gen = eval_turn["resp_gen"]
                elif args.task in ["e2e", "nlg"]:  # inference
                    """
                    Construct prompt for inference
                    """
                    messages = []
                    # system instruction
                    system_messages = [random.choice(tod_instructions)]
                    system_messages.extend(tod_notes)
                    system_message = "\n".join(system_messages)
                    messages.append({"role": "system", "content": system_message})

                    # select examples for the current domain
                    if not args.multi_domain and turn_domain in examples:
                        domain_examples = examples[turn_domain][: args.nlg_nshot]
                    else:
                        domain_examples = []

                    # previous example conversations (DELX, w/ DB Info)
                    example_messages = []
                    for bs_example in domain_examples:
                        example_message = []
                        for turn in bs_example:
                            domain = turn["dspn"]
                            user = turn["user"]
                            resp = turn["resp"]
                            db_num = turn["db"]
                            bs_dict = turn["bspn_dict"]

                            # add user message
                            example_message.append({"role": "user", "content": user})
                            # add assistant message
                            if domain in bs_dict and args.add_prev:
                                function_call_dict = {
                                    "function": domain2function_mapping[domain[1:-1]],
                                    "arguments": bs_dict[domain],
                                    "results": f"{db_num} {domain[1:-1]} matched",
                                }
                                example_message.append(
                                    {
                                        "role": "assistant",
                                        "content": resp,
                                        "function_call": function_call_dict,
                                    }
                                )
                            else:
                                example_message.append(
                                    {"role": "assistant", "content": resp}
                                )
                        example_messages.append(example_message)

                    # history message in the current conversation
                    for prev_turn in eval_turns[:idx]:
                        usr = prev_turn["user"]
                        resp = prev_turn["resp"]
                        prev_domain = (
                            prev_turn["dspn"]
                            if args.ref_domain
                            else prev_turn["dspn_gen"]
                        )
                        prev_bs_dict = (
                            prev_turn["bspn_dict"]
                            if args.ref_bs
                            else prev_turn["bspn_dict_gen"]
                        )
                        # database results
                        db = reader.db.get_match_num(prev_bs_dict)
                        db_num = len(db)

                        # add user message
                        messages.append({"role": "user", "content": usr})

                        # add assistant turn
                        assistant_message = {"role": "assistant", "content": resp}

                        if args.add_prev:
                            if args.multi_domain:
                                if prev_domain in prev_bs_dict:
                                    function_call_dict = {
                                        "function": domain2function_mapping[
                                            prev_domain[1:-1]
                                        ],
                                        "arguments": prev_bs_dict[prev_domain],
                                        "results": f"{db_num} {prev_domain[1:-1]} matched",
                                    }
                                    assistant_message[
                                        "function_call"
                                    ] = function_call_dict
                            else:
                                if turn_domain in prev_bs_dict:
                                    function_call_dict = {
                                        "function": domain2function_mapping[
                                            turn_domain[1:-1]
                                        ],
                                        "arguments": prev_bs_dict[turn_domain],
                                        "results": f"{db_num} {turn_domain[1:-1]} matched",
                                    }
                                    assistant_message[
                                        "function_call"
                                    ] = function_call_dict
                        messages.append(assistant_message)

                    # current turn
                    usr = eval_turn["user"]
                    # database results
                    db = reader.db.get_match_num(user_goal)
                    db_num = len(db)
                    # add user message
                    messages.append({"role": "user", "content": usr})
                    # add assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": "",
                    }
                    if turn_domain in user_goal:
                        if user_goal[turn_domain]:
                            assistant_message["function_call"] = {
                                "function": domain2function_mapping[turn_domain[1:-1]],
                                "arguments": user_goal[turn_domain],
                                "results": f"{db_num} {turn_domain[1:-1]} matched",
                            }
                    messages.append(assistant_message)

                    # generate response
                    chat_response, in_out = ChatCompletion.complete(
                        messages=messages,
                        functions=functions,
                        function_call="none" if len(functions) else {},
                        examples=example_messages,
                        required=["content"],
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=128,
                        n_seqs=1,
                    )
                    assistant_message = chat_response[0]
                    assistant_response = assistant_message["content"]

                    # record
                    resp_gen = assistant_response
                    eval_turn["resp_gen"] = assistant_response  # for evaluation

                    eval_in_out[dial_id][idx]["nlg"] = {
                        "prompt": in_out["prompt"],
                        "output": in_out["output"],
                        "resp": eval_turn["resp"],
                        "resp_gen": eval_turn["resp_gen"]
                    }

                    if save_interval <= 0:
                        with open(eval_result_path, "w") as file:
                            json.dump(eval_data, file, indent=4)
                        with open(eval_messages_path, "w") as file:
                            json.dump(eval_in_out, file, indent=4)

                    # debug
                    if args.verbose:
                        duration = time.time() - turn_st
                        print("=" * 25 + f" {dial_id}-{idx} (consume {duration:.2f} secs)" + "=" * 25)
                        print(f"User: {eval_turn['user']}")
                        print(f"Detect domain:", turn_domain)
                        print(f"Oracle bspn: {eval_turn['bspn']}")
                        print(f"Generated bspn: {eval_turn['bspn_gen']}")
                        print(f"Oracle response: {eval_turn['resp']}")
                        print(f"Generated response: {eval_turn['resp_gen']}")

                    if args.debug:
                        _ = input()

                if not args.verbose:
                    num_completed_turns += 1
                    d_pbar.set_postfix(
                        c_turns=num_completed_turns, n_turns=num_turns
                    )

            if save_interval > 0 and (didx % save_interval == 0 or didx == len(eval_data) - 1):
                with open(eval_result_path, "w") as file:
                    json.dump(eval_data, file, indent=4)
                with open(eval_messages_path, "w") as file:
                    json.dump(eval_in_out, file, indent=4)

            if not args.verbose:
                d_pbar.update(1)
    else:
        # load the model
        ChatCompletion = chat_completion(
            model=args.model,
            function_type=args.function_type,
            function_call_prefix=fc_prefix,
            function_call_suffix=fc_suffix,
            verbose=args.verbose,
        )
        dst_results = None
        if args.dst_result_path and os.path.isfile(args.dst_result_path):
            print(f"Use the '{args.gen_state_channel}' channel of dst results from '{args.dst_result_path}'.")
            with open(args.dst_result_path, "r") as file:
                dst_results = json.load(file)

        num_invalid_dst = 0
        d_pbar = tqdm(total=len(eval_data), desc=f"Evaluation {args.split}")
        for didx, (dial_id, eval_turns) in enumerate(eval_data.items()):
            user_goal = {}

            for idx, eval_turn in enumerate(eval_turns):
                turn_domain = eval_turn["dspn_gen"]
                # Find the domain schema, examples for the prompt construction
                functions = []
                current_function = None
                for domain in EXPERIMENT_DOMAINS:
                    for service in schema:
                        if service["service_name"] == domain[1:-1]:
                            function = schema2function(
                                service,
                                template=ChatCompletion.template,
                                rename_mapping=domain2function_mapping,
                            )
                            if args.multi_domain:
                                functions.append(function)
                            elif domain == turn_domain:  # only the current turn domain
                                current_function = function
                                functions.append(current_function)
                            break

                bspn_dict_gen = user_goal
                if args.track_slot_status:
                    status_messages, status_functions, domain2mapping = adapt_track_slot_status(
                        [], functions, domain2function_mapping, domain2desc)
                    if "dst" in eval_in_out[dial_id][idx]:
                        chat_response = eval_in_out[dial_id][idx]["dst"]["output"]
                        turn_status_dict_gen = parse_status_response(
                            chat_response[0], domain2mapping, status_functions, user_goal, EXPERIMENT_DOMAINS
                        )
                    else:
                        num_invalid_dst += 1

                    if dst_results:
                        external_bs = dst_results[dial_id][idx][args.gen_state_channel]
                        external_bs_dict = paser_bs_to_dict(external_bs)
                        bspn_dict_gen = filter_state_based_on_status(external_bs_dict, user_goal)
                else:
                    bspn_dict_gen = eval_turn["bspn_dict_gen"]

                bspn_gen = paser_dict_to_bs(bspn_dict_gen)
                eval_turn[args.gen_state_channel] = bspn_gen  # for evaluation
                eval_turn[args.gen_state_channel.replace("bspn_gen", "bspn_dict_gen")] = copy.deepcopy(bspn_dict_gen)
        print(f"Number of invalid DST results: {num_invalid_dst}")


    """ Evaluations """
    print(args)
    eval_turns = unzip_session_data(eval_data)
    all_metrics = {}

    """
    Domain Prediction evaluation
    """
    print("=" * 25 + f" DP Evaluation " + "=" * 25)

    valid_turns = []
    for data in eval_turns:
        if data["bspn_gen"]:
            valid_turns.append(data)
    valid, total = len(valid_turns), len(eval_turns)
    if args.eval_valid:
        print("using valid turns")
        evaluated_turns = valid_turns
    else:
        evaluated_turns = eval_turns

    total, correct = 0.0, 0.0
    for eval_turn in evaluated_turns:
        if eval_turn["dspn_gen"] == eval_turn["dspn"]:
            correct += 1
        total += 1
    dp_acc = correct / total if total > 0 else 0
    print("Test Domain Prediction Accuracy is {}. ".format(dp_acc))
    all_metrics["Domain"] = {
        "accuracy": dp_acc,
        "correct_turns": correct,
        "total_turns": total,
    }

    """
    DST evaluation
    """
    print("=" * 25 + f" DST Evaluation " + "=" * 25)

    valid_turns = []
    for data in eval_turns:
        if data["bspn_gen"]:
            valid_turns.append(data)
    valid, total = len(valid_turns), len(eval_turns)
    if args.eval_valid:
        print("using valid turns")
        evaluated_turns = valid_turns
    else:
        evaluated_turns = eval_turns

    from src.multiwoz.utils.compute_joint_acc import compute_jacc, zip_result

    all_dev_result = zip_result(evaluated_turns)
    (
        dev_score,
        dev_f1,
        dev_precision,
        dev_recall,
        per_domain_jga,
        avg_domain_jga,
        per_slot_acc,
        dev_error,
        overlap_dict,
        per_domain_overlap
    ) = compute_jacc(
        data=all_dev_result,
        gen_state_channel=args.gen_state_channel if args.divide_inform_confirm or args.track_slot_status else "bspn_gen",
        ignore_dontcare_in_pred=True
    )
    dev_score *= 100
    print(
        "Test Joint Accuracy is {}. Slot F1 is {}, Precision is {}, Recall is {}".format(
            dev_score, dev_f1, dev_precision, dev_recall
        )
    )
    print(
        "Number of total turns is {}, Number of valid turns is {}".format(total, valid)
    )

    print(f"\nTurn-level Activate Slot Overlap Analysis: {json.dumps(overlap_dict, indent=2)}")
    print(f"\nPer Domain Overlap Analysis:")
    for domain, overlap in per_domain_overlap.items():
        simplied_overlap = {k: overlap[k] for k in ["match_ratio", "value_match_ratio", "cover_ratio", "value_cover_ratio"]}
        print(f"{domain}: {json.dumps(simplied_overlap)}")

    print("\nPer Domain Accuracy:")
    for domain in per_domain_jga:
        print(
            "Domain: {}, Test Joint Accuracy is {}, F1 is {}, Total turns {}".format(
                domain,
                per_domain_jga[domain][0],
                per_domain_jga[domain][2],
                per_domain_jga[domain][1],
            )
        )
    print("\nAverage Domain Accuracy:")
    print(json.dumps(avg_domain_jga, indent=2))

    print("\nPer Slot Accuracy:")
    for domain in per_slot_acc:
        for slot in per_slot_acc[domain]:
            print(
                "{}-{}, Test Joint Accuracy is {}, Total turns {}".format(
                    domain,
                    slot,
                    per_slot_acc[domain][slot][0],
                    per_slot_acc[domain][slot][1],
                )
            )
    with open(dst_eval_error_path, "w") as file:
        json.dump(dev_error, file, indent=4)
    all_metrics["DST"] = {
        "gen_state_channel": args.gen_state_channel,
        "total_turns": total,
        "non_empty_turns": valid,
        "joint_acc": dev_score,
        "slot_precision": dev_precision,
        "slot_recall": dev_recall,
        "slot_f1": dev_f1,
        "per_domain_jga": per_domain_jga,
        "avg_domain_jga": avg_domain_jga,
        "per_slot_acc": per_slot_acc,
        "turn_level_active_slot_overlap": overlap_dict,
        "per_domain_overlap": per_domain_overlap
    }

    """
    NLG evaluation, inform, success, combined score
    """
    print("=" * 25 + f" NLG Evaluation " + "=" * 25)

    valid_turns = []
    for data in eval_turns:
        if data["resp_gen"]:
            valid_turns.append(data)
    valid, total = len(valid_turns), len(eval_turns)
    if args.eval_valid:
        print("using valid turns")
        evaluated_turns = valid_turns
    else:
        evaluated_turns = eval_turns

    from src.multiwoz.utils.eval import MultiWozEvaluator

    evaluator = MultiWozEvaluator(dataset_version=args.dataset_version)
    (
        dev_bleu,
        dev_success,
        dev_match,
        total_successes,
        total_matches,
        dial_nums,
    ) = evaluator.validation_metric(evaluated_turns)
    dev_score = 0.5 * (dev_success + dev_match) + dev_bleu
    print(
        "Test bleu {}, success rate {}, inform rate {}".format(
            dev_bleu, dev_success, dev_match
        )
    )
    print("Test combined score {}".format(dev_score))
    print("Test total successes {}, matches {}".format(total_successes, total_matches))
    print(
        "Number of total turns is {}, Number of valid turns is {}".format(total, valid)
    )
    all_metrics["NLG"] = {
        "total_turns": total,
        "non_empty_turns": valid,
        "bleu": dev_bleu,
        "inform": dev_match,
        "success": dev_success,
        "combined": dev_score,
        "total_successes": total_successes,
        "total_matches": total_matches,
    }

    with open(eval_metrics_path, "w") as file:
        json.dump(all_metrics, file, indent=4)
