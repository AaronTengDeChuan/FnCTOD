# coding: utf-8

import os
import json
from copy import deepcopy

def construct_prompt(messages, schema, active_domains, slot_desc_flag=True, PVs_flag=True):
    # cautions
    # exact_name_caution = "Always return the exact value of the 'name' slot when mentioned or confirmed by the user. Avoid using pronouns or coreferences like 'the hotel' or 'the restaurant.'"
    exact_name_caution = ""

    SYSTEM_CAUTIONS = [
        "If the slot is not mentioned in the dialogue, just return NONE.",
        "If the slot is not explicitly provided or confirmed by the user in the dialogue, just return NONE. Don't add or guess slot values.",
        # "Return the slot value as : 'dontcare' ONLY when the user EXPLICITLY states they have no specific preference for a slot.",
    ]
    CAUTIONS = [
        "If the slot is not mentioned in the dialogue, just return NONE.",
        "If the slot is not explicitly provided or confirmed by the user in the dialogue, just return NONE. Don't add or guess slot values.",
    ]

    # construct prompt
    sys_prompt = "Now you need to perform the task of multi-domain dialogue state tracking. You need to return the value of the slot I'm asking about simply based on the content of the dialogue. No explanation!"
    sys_prompt += " " + " ".join(SYSTEM_CAUTIONS)
    assert messages[0]["role"] == "system"
    new_messages = [
        {"role": "system", "content": sys_prompt.strip()},
        {"role": "user", "content": None}
    ]

    # construct context
    dial_context = ""
    usr_speaker = "[USER]"
    # sys_speaker = "[SYSTEM]"
    sys_speaker = "[ASSISTANT]"
    for idx, message in enumerate(messages[1:]):
        if message["role"] == "user":
            speaker = f" {usr_speaker} "
        elif message["role"] == "assistant":
            speaker = f" {sys_speaker} "
        else:
            raise ValueError(f"Invalid role: {message['role']}")
        dial_context += speaker + message["content"]
    dial_context = dial_context.strip()


    # construct slot specifications
    domain_slot_specs = {}
    for domain in active_domains:
        d_name = domain[1:-1]
        for service in schema:
            if service["service_name"] == d_name:
                domain_slot_specs[domain] = {}
                slots = service["slots"]
                for slot in slots:
                    if not slot["is_informable"]:
                        continue
                    s_name = slot["name"].split("-")[-1].strip()
                    domain_slot_specs[domain][s_name] = {}
                    slot_description = slot["description"].strip()
                    assert len(slot_description) > 0
                    slot_desc = f"{s_name}, it indicates {slot_description}" if slot_desc_flag else s_name
                    domain_slot_specs[domain][s_name]["prompt"] = \
                        f"The following is the initial state and the dialogue between {usr_speaker} and {sys_speaker}:\nInitial State: <{d_name}-{s_name}> = NONE\nInput dialogue: {dial_context} \n\n [domain] {d_name}, [slot] {slot_desc}"
                    # domain_slot_specs[domain][s_name]["prompt"] = \
                    #     f"{dial_context} \n\n [domain] {d_name}, [slot] {slot_desc}"

                    slot_PVs = ""
                    if "possible_values" in slot and PVs_flag:
                        if slot["is_categorical"]:
                            candidate_values = [str(v) for v in slot["possible_values"]]
                            domain_slot_specs[domain][s_name]["enum"] = candidate_values
                            slot_PVs += ". This slot is categorical and you can only choose from the following available values: "
                            slot_PVs += ", ".join(candidate_values)
                        else:
                            # pass
                            if slot["possible_values"]:
                                examples = ", ".join(slot["possible_values"][:10])
                                slot_PVs += f", such as {examples}, etc"
                        slot_PVs += ". "
                    else:
                        slot_PVs += ". "

                    domain_slot_specs[domain][s_name]["prompt"] += slot_PVs
                    caution_str = " ".join(([exact_name_caution] if s_name == "name" else []) + CAUTIONS)
                    domain_slot_specs[domain][s_name]["prompt"] += f"{caution_str.strip()} \n "
                    domain_slot_specs[domain][s_name]["prompt"] += "So the value of slot <"+ d_name+ "-" + s_name +"> is "

    return new_messages, domain_slot_specs


def generate_dst(eval_turn, messages, schema, active_domains, completion_func):
    messages, domain_slot_specs = construct_prompt(messages, schema, active_domains)
    state = {}
    eval_in_out = []
    for domain in domain_slot_specs:
        state[domain] = {}
        for slot in domain_slot_specs[domain]:
            # state[domain][slot] = eval_turn["bspn_dict"].get(domain, {}).get(slot, "NONE").strip().lower()
            # continue
            new_messages = deepcopy(messages)
            new_messages[1]["content"] = domain_slot_specs[domain][slot]["prompt"]
            response, in_out = completion_func(messages=new_messages)
            # record value
            predict_value = response[0]["content"].strip().lower()
            # domain_slot_specs[domain][slot]["output"] = predict_value
            if "enum" not in domain_slot_specs[domain][slot]:
                state[domain][slot] = predict_value
            elif predict_value in domain_slot_specs[domain][slot]["enum"]:
                state[domain][slot] = predict_value
            # record input-output
            in_out["target_value"] = eval_turn["bspn_dict"].get(domain, {}).get(slot, "NONE")
            in_out["predict_value"] = predict_value
            eval_in_out.append(in_out)

    return state, eval_in_out

