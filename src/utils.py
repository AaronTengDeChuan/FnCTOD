#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re
import json
import argparse
from copy import deepcopy


############## Utilities ##############
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [item.strip() for item in v.split("+")]
    else:
        raise argparse.ArgumentTypeError("List value expected.")


def word2num(word):
    word_to_num = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }
    return word_to_num.get(word)


def string2int(s):
    # Check if it's an integer
    if s.isdigit():
        return s
    # Check if it's a spelled out number
    num = word2num(s.lower())
    if num is not None:
        return num
    return None


def add_bracket(api_dict, level=1):
    if level == 1:
        return {f"[{key}]": value for key, value in api_dict.items()}
    elif level == 2:
        api_dict = {f"[{key}]": value for key, value in api_dict.items()}
        return {
            key: {f"[{sub_key}]": sub_value for sub_key, sub_value in value.items()}
            for key, value in api_dict.items()
        }
    else:
        raise NotImplementedError


def remove_bracket(api_dict, level=1):
    if level == 1:
        return {key[1:-1]: value for key, value in api_dict.items()}
    elif level == 2:
        api_dict = {key[1:-1]: value for key, value in api_dict.items()}
        return {
            key: {sub_key[1:-1]: sub_value for sub_key, sub_value in value.items()}
            for key, value in api_dict.items()
        }
    else:
        raise NotImplementedError


def dict_update(ori_dict, new_dict):
    # recursively update the dictionary
    for key, value in new_dict.items():
        if key in ori_dict:
            if not isinstance(ori_dict[key], dict) or not isinstance(value, dict):
                ori_dict[key] = value
            else:
                dict_update(ori_dict[key], value)
        else:
            ori_dict[key] = value


def round_floats(d, decimals=4):
    if isinstance(d, float):
        return round(d, decimals)
    elif isinstance(d, list):
        return [round_floats(i, decimals) for i in d]
    elif isinstance(d, dict):
        new_d = {}
        for key, value in d.items():
            new_d[key] = round_floats(value, decimals)
        return new_d
    else:
        return d


############################################

############## regex patterns ##############
space = ' '
domain_str = '|'.join(f"{space}{d}" for d in ["taxi", "train", "attraction", "hotel", "restaurant", "hospital", "police", "general"])
domain_prediction_regex = rf"({domain_str})"

def get_argument_regex(function, add_extra=False, independent=False):
    extra_values = ["none", "dontcare"]
    regex_dict = {}
    for slot, slot_info in function["parameters"]["properties"].items():
        if "enum" in slot_info:
            enum_values = deepcopy(slot_info["enum"])
            for extra_value in extra_values[::-1]:
                if add_extra and extra_value not in enum_values:
                    enum_values.insert(0, extra_value)
            regex_dict[slot] = re.compile(rf"({'|'.join(enum_values)})")
        elif slot in ["time", "leave_at_or_after", "leaveat", "leave" "arrive_by", "arriveby", "arrive"]:
            time_pattern = r"([01][0-9]|2[0-4]):[0-5][0-9]"
            if add_extra:
                time_pattern = rf"({'|'.join(extra_values)}|{time_pattern})"
            regex_dict[slot] = re.compile(time_pattern)

        else:
            regex_dict[slot] = re.compile(r'[^"\\\x00-\x1F\x7F-\x9F]{0,50}')

    # whitespace_pattern = r",\n    "
    whitespace_pattern = ", "

    if independent:
        json_regex = []
        for slot, regex in regex_dict.items():
            json_regex.append(r" \{" + whitespace_pattern.lstrip(", ") + f'"{slot}": "{regex.pattern}"' + r"\}")
    else:
        json_regex = (r" \{" + whitespace_pattern.lstrip(", ")
                      + whitespace_pattern.join([f'"{slot}": "{regex.pattern}"' for slot, regex in regex_dict.items()])
                      + r"\}")
    # print(json_regex)
    return regex_dict, json_regex


def fill_inactive_slots(domain_goal, domain_slots, fill_value="none"):
    new_goal = {}
    for slot in domain_slots:
        new_goal[slot] = domain_goal.get(slot, fill_value)
    return new_goal


############## Configurations ##############
domain_prefix = "<domain>"
domain_suffix = "</domain>"

fc_prefix = "<function_call> "
fc_suffix = "  </function_call> "

tod_instructions = [
    "You are a task-oriented assistant. You can use the given functions to fetch further data to help the users.",
    "Your role as an AI assistant is to assist the users with the given functions if necessary.",
    "You are a task-oriented assistant, concentrating on assisting users with the given functions if necessary.",
    "You are a task-oriented assistant to provide users with support using the given functions if necessary.",
    "You are a task-focused AI. Your primary function is to help the users to finish their tasks using the given function(s) to gather more information if necessary.",
    "You are a task-oriented assistant. Your primary objective is assisting users to finish their tasks, using the given function(s) if necessary.",
    "Your primary role is to assist users using the given function (s), as a specialized task-oriented assistant.",
    "As an AI with a task-focused approach, your primary focus is assisting users to finish their tasks using the given functions.",
]

tod_notes = [
    "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    "Use only the argument values explicitly provided or confirmed by the user instead of the assistant. Don't add or guess argument values.",
    "Ensure the accuracy of arguments when calling functions to effectively obtain information of entities requested by the user.",
]

inform_notes = [
    "You are a task-oriented assistant. You need to use the given functions to extract the specific constraints informed by the user.",
    "Don't make assumptions about what values to plug into functions.",
    # "Track only constraints explicitly provided by the user, not those suggested by the assistant.",
    "Don't add or guess argument values.",
    "Ensure the accuracy of arguments when calling functions."
]

confirm_notes = [
    "You are a task-oriented assistant. You need to use the given functions to identify the specific constraints that are informed by the assistant and then are explicitly confirmed by the user.",
    "Don't make assumptions about what values to plug into functions.",
    # "Track only constraints that meet the following two conditions: 1) Provided solely by the assistant, 2) Explicitly confirmed by the user.",
    "Don't add or guess argument values.",
    "Ensure the accuracy of arguments when calling functions."
]


def adapt_inform_or_confirm(type, functions, domain2function_mapping):
    assert type in ["inform", "confirm"]
    reverse_mapping = {v: k for k, v in domain2function_mapping.items()}
    if type == "inform":
        domain2mapping = {d: f"track_{d}_constraints_informed_by_user" for d in domain2function_mapping}
        extra_desc = "Track only constraints explicitly provided by the user, not those suggested by the assistant!"
    elif type == "confirm":
        domain2mapping = {d: f"track_{d}_constraints_confirmed_by_user" for d in domain2function_mapping}
        extra_desc = "Track only constraints that meet the following two conditions: 1) Provided solely by the assistant, 2) Explicitly confirmed by the user!"
    else:
        raise NotImplementedError

    new_functions = deepcopy(functions)
    for function in new_functions:
        domain = reverse_mapping.get(function["name"], function["name"])
        function["name"] = domain2mapping[domain]
        function["description"] += f" {extra_desc}"

    system_message = "\n".join(inform_notes if type == "inform" else confirm_notes)

    return system_message, new_functions, domain2mapping


def build_divide_examples(type, domain_examples, domain2mapping, EXPERIMENT_DOMAINS):
    assert type in ["inform", "confirm", None]
    example_messages = []
    for bs_example in domain_examples:
        example_message = []
        last_fc_idx = None
        for tidx, turn in enumerate(bs_example):
            domain = turn["dspn"]
            user = turn["user"]
            resp = turn["nodelx_resp"]
            if type is None:
                bs_dict = turn["bspn_dict"]
            else:
                bs_dict = turn[f"{type}_bspn_dict"]
            db_num = turn["db"]

            # add user message
            example_message.append({"role": "user", "content": user})
            # add assistant message
            if domain in EXPERIMENT_DOMAINS:
                # if domain in bs_dict:
                if True:
                    function_call_dict = {
                        "function": domain2mapping[
                            domain[1:-1]
                        ],
                        "arguments": bs_dict.get(domain, {}),
                    }
                    example_message.append(
                        {
                            "role": "assistant",
                            "content": resp,
                            "function_call": function_call_dict,
                        }
                    )
                    last_fc_idx = 2 * tidx + 1
                else:
                    example_message.append(
                        {"role": "assistant", "content": resp}
                    )
            else:
                example_message.append(
                    {"role": "assistant", "content": resp}
                )
        # only remain last function_call
        # if last_fc_idx is not None:
        #     for idx, message in enumerate(example_message):
        #         if idx != last_fc_idx and "function_call" in message:
        #             message.pop("function_call")

        example_messages.append(example_message)
    return example_messages


def build_status_examples(
        args, domain2inform_slots, domain_examples, domain2mapping, EXPERIMENT_DOMAINS):
    example_messages = []
    for bs_example in domain_examples:
        previous_message = []
        all_domains = []
        for tidx, turn in enumerate(bs_example):
            domain = turn["dspn"]
            user = turn["user"]
            resp = turn["nodelx_resp"]
            turn_svs_dict = turn["turn_svs_dict"]
            previous_message.append({"role": "user", "content": user})

            possible_domain = None
            if domain in EXPERIMENT_DOMAINS:
                possible_domain = domain
            elif len(all_domains) > 0:
                possible_domain = all_domains[-1]

            if possible_domain is not None:
                domain_svs = turn_svs_dict.get(possible_domain, {})
                arguments = fill_inactive_slots(
                    domain_svs, domain2inform_slots[possible_domain[1:-1]],
                    fill_value=slot_status[0]) if args.fill_inactive else {
                    k: domain_svs[k] for k in domain2inform_slots[possible_domain[1:-1]] if k in domain_svs}
                function_call_dict = {
                    "function": domain2mapping[
                        possible_domain[1:-1]
                    ],
                    "arguments": {k: status2option[status2text[v]] for k, v in arguments.items()},
                }
                example_messages.append(
                    deepcopy(previous_message[-2:]) + [
                        {"role": "assistant", "content": "", "function_call": function_call_dict}])

            previous_message.append({"role": "assistant", "content": resp})
            if domain in EXPERIMENT_DOMAINS:
                all_domains.append(domain)
    return example_messages


user_tag = "USER"
# user_tag = "User"
assistant_tag = "ASSISTANT"
# assistant_tag = "Assistant"

usr_confirm = "user_confirm"
# usr_confirm = "user_accept"
slot_status = ["not_informed", "user_inform", "assistant_inform", usr_confirm]
# status2text = {
#     "not_informed": "Not Informed",
#     "user_inform": f"Informed by {user_tag}",
#     "assistant_inform": f"Informed by {assistant_tag}",
#     usr_confirm: f"Confirmed by {user_tag}",
# }
status_space = ''
status2text = {
    "not_informed": f"{status_space}not_informed{status_space}",
    "user_inform": f"{status_space}user_inform{status_space}",
    "assistant_inform": f"{status_space}assistant_inform{status_space}",
    usr_confirm: f"{status_space}user_confirm{status_space}",
}

text2status = {v: k for k, v in status2text.items()}

update_status = ["user_inform", usr_confirm]

option2status = {
    "A": status2text[slot_status[0]],
    "B": status2text[slot_status[1]],
    "C": status2text[slot_status[2]],
    "D": status2text[slot_status[3]],
}
status2option = {v: k for k, v in option2status.items()}
status2desc = {
    "A": "The slot has not been mentioned in the conversation yet.",
    "B": f"The slot value is{{modifier}} informed by the {user_tag} and may be referred to by the {user_tag} using pronouns or coreferences, such as 'the hotel', 'the restaurant' and so on.",
    # "B": "The slot value is directly informed by the user.",
    "C": f"The slot value is directly informed by the {assistant_tag} .",
    # "D": f"The slot value is informed by the {assistant_tag} and then explicitly confirmed by the {user_tag} . For example, the {assistant_tag} recommended a 'name' to the {user_tag} , and the {user_tag} accepted the suggested 'name'.",
    "D": f"The slot value is first informed by the {assistant_tag} and then explicitly confirmed by the {user_tag} .{{confirm_extra}}",
}
# confirm_example = f" For example, the {assistant_tag} recommended a '{{domain}}name' to the {user_tag} , and then the {user_tag} expresses acceptance of the recommendation."
confirm_example = f" For example, the {assistant_tag} recommended a '{{domain}}name' to the {user_tag} , and then the {user_tag} accepted the suggested 'name'."

status_desc_str = "; ".join([f" - {k.strip()} : {v}" for k, v in status2desc.items()])
slot_status_notes = [
    "You are a task-oriented assistant. You need to use the given functions to track the source of slot values for all slots in the given domain.",
    # "You are a task-oriented ASSISTANT. You need to use the given functions to track the source of slot values for all slots in the given domain.",
    # "You are required to utilize the provided functions to track the source of slot values across all slots within the specified domain during the following dialogue between the USER and the ASSISTANT.",
    # "If the value of a slot has not yet appeared in the conversation, set the status of that slot to 'not_informed'."
    # f"The status of slots is described as follows:\n{status_desc_str}",
    # "Do not miss any slot mentioned in the dialogue.",
    # f"You must accurately identify whether the mentioned slot value is informed by the {user_tag} or the {assistant_tag} .",
    # f"The status of the slots can only be selected from {list(option2status.keys())}, do not use the actual values of the slots."
]


def set_global_variable_for_status(args):
    global status2desc, option2status, status_desc_str, status2option
    if args.status_option == "abstract":
        pass
    elif args.status_option == "concrete":
        status2desc = {option2status[k]: v for k, v in status2desc.items()}
        status_desc_str = "; ".join([f" - {k.strip()} : {v}" for k, v in status2desc.items()])
        option2status = {s: s for s in option2status.values()}
        status2option = {v: k for k, v in option2status.items()}
    else:
        raise NotImplementedError


def adapt_track_slot_status(args, messages, functions, domain2function_mapping, domain2desc):
    # print(status2desc)
    # print(option2status)
    # print(status_desc_str)
    # exit(0)

    status_messages = None
    if messages:
        # modify the messages
        # status_messages = deepcopy(messages)
        # status_messages = deepcopy(messages[:1] + messages[-min(3, len(messages) - 1):])
        status_messages = deepcopy(messages[:1] + messages[-min(args.status_context_window, len(messages) - 1):])
        system_message = "\n".join(slot_status_notes)
        status_messages[0]["content"] = system_message
        for message in status_messages:
            if message["role"] == "user":
                message["content"] = f"{user_tag} : {message['content']}"
            elif message["role"] == "assistant":
                message["content"] = f"{assistant_tag} : {message['content']}"

    no_enum_slots = [
        # "departure", "destination",
        # ["restaurant", ["time", "booktime"]],
        # ["attraction", ["type"]],
        # ["train", ["leave_at_or_after", "leaveat", "arrive_by", "arriveby"]]
    ]

    reverse_mapping = {v: k for k, v in domain2function_mapping.items()}
    # modify the functions
    domain2mapping = {d: f"track_{d}_slot_status" for d in domain2function_mapping}
    # domain2mapping = {d: f"track_{d}_slot_value_source" for d in domain2function_mapping}

    new_functions = deepcopy(functions)
    for function in new_functions:
        domain = reverse_mapping.get(function["name"], function["name"])
        function["name"] = domain2mapping[domain]

        if "name" in function["parameters"]["properties"]:
            # confirm_extra = confirm_example.format(**{"domain": f"{domain}-"})
            confirm_extra = confirm_example.format(**{"domain": ""})
        else:
            confirm_extra = confirm_example.format(**{"domain": ""})
        # confirm_extra = ""
        if domain in []:
            modifier = ""
        else:
            modifier = " directly"
        # if domain in ["train", "taxi"]:
        if domain in []:
            domain_status_desc = status_desc_str.format(**{
                "modifier": modifier, "confirm_extra": ""})
        else:
            domain_status_desc = status_desc_str.format(**{
                "modifier": modifier, "confirm_extra": confirm_extra})
        extra_desc = f"Track the source of slot values for all slots in the '{domain}' domain: {domain2desc[domain]}"
        extra_desc += f" The status of slots is described as follows: {domain_status_desc}"
        # if domain in ["taxi"]:
        #     extra_desc += f"\nYou must accurately identify whether the mentioned slot value is informed by the {user_tag} or the {assistant_tag} ."

        function["description"] = f"{extra_desc}"
        new_properties = {}
        for slot, slot_info in function["parameters"]["properties"].items():
            left_parentheses_idx = slot_info["description"].find("(")
            if left_parentheses_idx != -1:
                slot_info["description"] = slot_info["description"][:left_parentheses_idx].strip()
            slot_info["type"] = "string"

            no_enum_flag = False
            # if domain in ["train"]:
            #     no_enum_flag = True
            for no_enum_slot in no_enum_slots:
                if isinstance(no_enum_slot, list):
                    if domain == no_enum_slot[0] and slot in no_enum_slot[1]:
                        no_enum_flag = True
                        break
                elif slot == no_enum_slot:
                    no_enum_flag = True
                    break

            if "enum" in slot_info and not no_enum_flag:
                slot_info["description"] += f", such as {', '.join(slot_info['enum'][:8])}."
            # slot_info["enum"] = slot_status
            slot_info["enum"] = list(option2status.keys())

            new_slot_info = deepcopy(slot_info)
            new_slot_info["description"] = f"The slot '{slot}' denotes {slot_info['description']}."
            new_properties[f"source_of_{slot}"] = new_slot_info
        # function["parameters"]["properties"] = new_properties

    return status_messages, new_functions, domain2mapping


def parse_status_response(assistant_message, domain2mapping, status_functions, user_status, EXPERIMENT_DOMAINS):
    turn_domain = None
    current_function = None
    turn_status_dict = {}
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
            for function in status_functions:
                if function["name"] == pred_function:
                    current_function = function
            assert current_function is not None

            # step 3: get the arguments
            turn_status_gen = function_call["arguments"]
            if isinstance(turn_status_gen, str):
                turn_status_gen = json.loads(turn_status_gen)
            assert isinstance(turn_status_gen, dict)
        except:
            print("Can not parse:", function_call)
            turn_status_gen = {}

        lower_option2status = {k.strip().lower(): text2status[v] for k, v in option2status.items()}

        if turn_domain in EXPERIMENT_DOMAINS:
            for slot, option in turn_status_gen.items():
                slot = slot.strip().lower()
                option = str(option).strip()
                lower_option = option.lower()
                status = lower_option2status.get(lower_option, option)
                # if status not in slot_status:
                #     status = update_status[0]
                if slot in current_function["parameters"]["properties"]:
                    real_slot = slot
                    if slot.startswith("source_of_"):
                        real_slot = slot[len("source_of_"):]
                    if status in update_status or status not in slot_status:
                    # if status in update_status:
                        if turn_domain not in turn_status_dict:
                            turn_status_dict[turn_domain] = {}
                        if turn_domain not in user_status:
                            user_status[turn_domain] = {}
                        turn_status_dict[turn_domain][real_slot] = status
                        user_status[turn_domain][real_slot] = status

    return turn_status_dict


def filter_state_based_on_status(user_goal, user_status):
    ignored_domains = []
    # ignored_domains = ["[taxi]"]
    # ignored_domains = ["[attraction]"]
    # ignored_domains = ["[restaurant]"]
    # ignored_domains = ["[taxi]", "[attraction]"]
    # ignored_domains = ["[taxi]", "[attraction]", "[restaurant]"]

    ignored_slots = []
    ignored_slots = ["name"]
    # ignored_slots = [["[restaurant]", "name"], ["[hotel]", "name"]]

    new_goal = {}
    for domain, domain_goal in user_goal.items():
        if domain in ignored_domains:
            new_goal[domain] = deepcopy(domain_goal)
        for slot, value in domain_goal.items():
            if slot in ignored_slots or [domain, slot] in ignored_slots:
                if domain not in new_goal:
                    new_goal[domain] = {}
                new_goal[domain][slot] = value

    for domain, domain_status in user_status.items():
        if domain in ignored_domains:
            continue
        domain_goal = user_goal.get(domain, {})
        for slot, status in domain_status.items():
            if slot in ignored_slots or [domain, slot] in ignored_slots:
                continue
            value = domain_goal.get(slot, None)
            if value is not None:
                if domain not in new_goal:
                    new_goal[domain] = {}
                new_goal[domain][slot] = value
    return new_goal


