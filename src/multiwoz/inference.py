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
import copy
import random
import asyncio
import logging
import argparse
from tqdm import tqdm
from functools import partial

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.multiwoz.inference_auxiliaries import *
from src.multiwoz.utils import *
# from src.multiwoz.utils.config import *
# from src.multiwoz.utils.reader import *
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


async def main(args, data_ns, vital_ns):
    eval_data = data_ns.eval_data
    eval_in_out = data_ns.eval_in_out

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
            user_goal, inform_user_goal, confirm_user_goal = await infer_each_turn(
                args, vital_ns, dial_id, eval_turns, eval_in_out[dial_id], idx, ChatCompletion,
                user_goal, inform_user_goal, confirm_user_goal, async_mode=False
            )
            if not args.verbose:
                num_completed_turns += 1
                d_pbar.set_postfix(
                    c_turns=num_completed_turns, n_turns=num_turns
                )

        # if save_interval > 0 and (didx % save_interval == 0 or didx == len(eval_data) - 1):
        if didx % save_interval == 0 or didx == len(eval_data) - 1:
            save_data(args, data_ns, vital_ns)

        if not args.verbose:
            d_pbar.update(1)
    d_pbar.close()


if __name__ == "__main__":
    args = get_args()
    print(args)

    data_ns, vital_ns = load_eval_data(args)
    print_namespace(data_ns)
    print_namespace(vital_ns)

    if args.generate:
        asyncio.run(main(args, data_ns, vital_ns))
    else:
        not_generate(args, data_ns, vital_ns)

    # run metric evaluation
    run_metric_evaluation(args, data_ns, vital_ns)

    exit(0)

    eval_data = data_ns.eval_data
    eval_in_out = data_ns.eval_in_out
    reader = vital_ns.reader
    schema = vital_ns.schema
    examples = vital_ns.examples
    div_examples = vital_ns.div_examples
    domain2desc = vital_ns.domain2desc
    eval_messages_path = vital_ns.eval_messages_path
    eval_result_path = vital_ns.eval_result_path
    dst_eval_error_path = vital_ns.dst_eval_error_path
    eval_metrics_path = vital_ns.eval_metrics_path

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
                        dst_examples = examples
                        # dst_examples = div_examples
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
                                # if domain in bs_dict and args.add_prev:
                                if domain in bs_dict:
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
                                ChatCompletion, chat_response[0], domain2mapping, schema, user_goal
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
                                ChatCompletion, chat_response[0], domain2mapping, schema, user_goal
                            )
                            in_out_["turn_confirm_bs_dict_gen"] = turn_confirm_bs_dict_gen
                            in_out.append(in_out_)
                            if td:
                                if turn_domain:
                                    assert turn_domain == td
                                turn_domain = td
                        else:
                            # example_messages = build_divide_examples(
                            #     None, domain_examples, domain2function_mapping, EXPERIMENT_DOMAINS)
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
                                ChatCompletion, assistant_message, domain2function_mapping, schema, user_goal
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
                        save_data(args, data_ns)

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
                            # if domain in bs_dict and args.add_prev:
                            if domain in bs_dict:
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
                        save_data(args, data_ns)

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

            # if save_interval > 0 and (didx % save_interval == 0 or didx == len(eval_data) - 1):
            if didx % save_interval == 0 or didx == len(eval_data) - 1:
                save_data(args, data_ns, vital_ns)

            if not args.verbose:
                d_pbar.update(1)
    else:
        not_generate(args, data_ns, vital_ns)

    # run metric evaluation
    run_metric_evaluation(args, data_ns, vital_ns)
