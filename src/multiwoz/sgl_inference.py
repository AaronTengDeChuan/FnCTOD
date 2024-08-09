# coding: utf-8

import os
import json
import sglang as sgl
from copy import deepcopy
from transformers import AutoTokenizer

from src.multiwoz.inference_auxiliaries import *


all_domains = ["restaurant", "hotel", "attraction", "train", "taxi", "hospital", "police", "general"]
# all_domains = ["restaurant", "hotel", "attraction", "train", "taxi", "general"]


def construct_domain_prompt(
        args, key, eval_turns, current_turn_idx, ChatCompletion):
    space = ' '
    messages, example_messages = prepare_domain_prediction(args, eval_turns, current_turn_idx, space=space)
    # extract the system message
    system_message = ""
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
            break
    # construct prompt from the user utterances
    prompt = ChatCompletion.conversation.get_prompt(
        system_message=system_message,
        messages=messages,
        functions=[],
        function_call={},
        examples=example_messages,
    )
    return {
        "key": key,
        "prompt": prompt.strip() + space
    }


@sgl.function
def predict_domain(s, key, prompt):
    s += prompt
    s += sgl.gen("domain", choices=all_domains)


def get_value_pattern(function, add_extra=False):
    extra_values = ["none", "dontcare"]
    slot_list = []
    for slot, slot_info in function["parameters"]["properties"].items():
        if "enum" in slot_info:
            enum_values = [cand.strip() for cand in slot_info["enum"]]
            for extra_value in extra_values[::-1]:
                if add_extra and extra_value not in enum_values:
                    enum_values.insert(0, extra_value)
            slot_list.append((slot, {"enum": enum_values}))
        elif slot in ["time", "leave_at_or_after", "leaveat", "leave" "arrive_by", "arriveby", "arrive"]:
            time_pattern = r"([01][0-9]|2[0-4]):[0-5][0-9]"
            if add_extra:
                time_pattern = rf"({'|'.join(extra_values)}|{time_pattern})"
            slot_list.append((slot, {"regex": time_pattern}))
        else:
            slot_list.append((slot, {}))
    return slot_list


option_replace = {
    status2text[slot_status[0]].strip(): f"Its value has not yet been mentioned in the current dialogue turn",
    status2text[slot_status[1]].strip(): f"Its value has been provided by the {user_tag} in the current dialogue turn",
    status2text[slot_status[2]].strip(): f"Its value has been informed by the {assistant_tag} in the current dialogue turn",
    status2text[slot_status[3]].strip(): f"Its value has been confirmed by the {user_tag} in the current dialogue turn",
}
option_replace = {k: k for k in option_replace.keys()}
reverse_option_replace = {v: k for k, v in option_replace.items()}


def construct_dst_prompt(
        args, vital_ns, key, meta_info, eval_turns, current_turn_idx, ChatCompletion):
    domain2desc = vital_ns.domain2desc
    domain2inform_slots = vital_ns.domain2inform_slots

    turn_domain = meta_info["turn_domain"]
    functions = meta_info["functions"]
    current_function = meta_info["current_function"]

    messages, example_messages, domain_examples = prepare_dst_prediction(
        args, vital_ns, eval_turns, current_turn_idx, turn_domain
    )
    if args.track_slot_status:
        # adapt messages and functions to track slot status
        status_messages, status_functions, domain2mapping = adapt_track_slot_status(
            args, messages, functions, domain2function_mapping, domain2desc)
        func_call = {"name": status_functions[0]["name"]} if current_function else {}
        slot_patterns = get_value_pattern(status_functions[0]) if func_call else []
        for slot, slot_info in slot_patterns:
            slot_info["enum"] = [option_replace[cand] for cand in slot_info["enum"]]

        meta_info["functions"] = deepcopy(status_functions)
        meta_info["func_call"] = func_call
        meta_info["domain2mapping"] = domain2mapping
        if args.no_enum:
            # remove enum values from the parameters
            for status_function in status_functions:
                for slot, slot_info in status_function["parameters"]["properties"].items():
                    slot_info.pop("enum", None)

        example_messages = build_status_examples(
            args, domain2inform_slots, domain_examples, domain2mapping, EXPERIMENT_DOMAINS)
        # extract the system message
        system_message = ""
        for message in status_messages:
            if message["role"] == "system":
                system_message = message["content"]
                break
        # construct prompt from the user utterances
        prompt = ChatCompletion.conversation.get_prompt(
            system_message=system_message,
            messages=status_messages,
            functions=status_functions,
            function_call=func_call,
            examples=example_messages,
        )
        return {
            "key": key,
            "prompt": prompt,
            "slot_patterns": slot_patterns,
            "track_slot_status": args.track_slot_status,
            "ind_decoding": args.ind_status
        }
    else:
        raise NotImplementedError("Not implemented yet.")


@sgl.function
def predict_dst(s, key, prompt, slot_patterns, track_slot_status, ind_decoding):
    s += prompt

    def _get_gen_params(slot, slot_info):
        if "enum" in slot_info:
            return {"choices": slot_info["enum"]}
        elif "regex" in slot_info:
            return {"regex": slot_info["regex"]}
        else:
            return {"max_tokens": 15, "stop": '"'}

    if not slot_patterns:
        s += sgl.gen("content", max_tokens=50, stop="\n\n")
        res = s["content"]
    else:
        arguments = {}
        whitespace_pattern = ", "
        s += " {" + whitespace_pattern.lstrip(", ")
        if ind_decoding:
            # fork
            forks = s.fork(len(slot_patterns))
            for idx, (fork, (slot, slot_info)) in enumerate(zip(forks, slot_patterns)):
                fork += f'"{slot}": "{status_space}' + sgl.gen(slot, **_get_gen_params(slot, slot_info))
                arguments[slot] = reverse_option_replace[fork[slot]] if track_slot_status else fork[slot]
        else:
            for idx, (slot, slot_info) in enumerate(slot_patterns):
                if track_slot_status:
                    step = s.fork(1)[0]
                    step += f'"{slot}": "{status_space}' + sgl.gen(slot, **_get_gen_params(slot, slot_info))
                    arguments[slot] = reverse_option_replace[step[slot]]
                    s += (f'"{slot}": "{status_space}{arguments[slot]}{status_space}"'
                          + (whitespace_pattern if idx < len(slot_patterns) - 1 else ""))
                else:
                    s += (f'"{slot}": "{status_space}' + sgl.gen(slot, **_get_gen_params(slot, slot_info))
                          + (f'{status_space}"{whitespace_pattern}' if idx < len(slot_patterns) - 1 else f'{status_space}"'))
                    arguments[slot] = s[slot]
            s += "}"
        res = json.dumps(arguments) + "}"

    s["result"] = res


def parse_dst(args, prompt, output, meta_info, ChatCompletion, tokenizer):
    turn_domain = meta_info["turn_domain"]
    functions = meta_info["functions"]
    func_call = meta_info["func_call"]
    domain2mapping = meta_info["domain2mapping"]

    chat_response = ChatCompletion.conversation.get_response(
        output, func_call, required="function_call"
    )
    chat_response["input_tokens"] = len(tokenizer.encode(prompt, add_special_tokens=False))
    chat_response["output_tokens"] = len(tokenizer.encode(output, add_special_tokens=False))
    in_out = {
        "prompt": prompt,
        "output": [chat_response],
    }

    if args.track_slot_status:
        turn_status_dict_gen = {}
        if args.multi_domain or turn_domain in EXPERIMENT_DOMAINS:

            turn_status_dict_gen = parse_status_response(
                chat_response, domain2mapping, functions, meta_info["user_goal"], EXPERIMENT_DOMAINS
            )
        in_out["turn_status_dict_gen"] = turn_status_dict_gen

    else:
        raise NotImplementedError("Not implemented yet.")

    return in_out


def pretty_print(meta_info, tokenizer):
    input_logprobs = meta_info["input_token_logprobs"]
    output_logprobs = meta_info["output_token_logprobs"]
    print("=" * 30, "Input", "=" * 30)
    for segment in input_logprobs:
        for token in segment:
            print(f"{tokenizer.convert_ids_to_tokens([token[1]])[0] + f'({token[1]})': <20}", end="")
        print()
    print("\n", "=" * 30, "Output", "=" * 30)
    for segment in output_logprobs:
        for token in segment:
            print(f"{tokenizer.convert_ids_to_tokens([token[1]])[0] + f'({token[1]})': <20}", end="")
    print("\n")


def main(args, data_ns, vital_ns):

    eval_data = data_ns.eval_data
    eval_in_out = data_ns.eval_in_out
    schema = vital_ns.schema

    # load the model
    ChatCompletion = chat_completion(
        model=args.model,
        api=True,
        host=args.host,
        port=args.port,
        function_type=args.function_type,
        function_call_prefix=fc_prefix,
        function_call_suffix=fc_suffix,
        verbose=args.verbose,
    )
    tokenizer = AutoTokenizer.from_pretrained(ChatCompletion.model_name, use_fast=False, trust_remote_code=True)
    # for domain in all_domains:
    for domain in option2status.values():
        domain = domain.strip()
        print(f"{domain: <15}: {str(tokenizer.encode(domain, add_special_tokens=False)): <20}", end="")
        text = f', "{domain}": '
        print(f"\n\t{text: <20}: {tokenizer.encode(text, add_special_tokens=False)}", end="")
        text = f'" {domain} "'
        print(f"\n\t{text: <20}: {tokenizer.encode(text, add_special_tokens=False)}")

    # exit(0)
    base_url = ChatCompletion.base_url

    sgl.set_default_backend(sgl.RuntimeEndpoint(base_url))

    d_pbar = tqdm(total=len(eval_data), desc=f"Evaluation {args.split}")
    max_turn = max([len(turns) for turns in eval_data.values()])
    num_turns = sum([len(turns) for turns in eval_data.values()])
    num_completed_dials = 0
    num_completed_turns = 0

    meta_data = {}
    for key, eval_turns in eval_data.items():
        meta_data[key] = {
            "user_goal": {},
        }

    for turn_idx in range(max_turn):
        assert len(meta_data) > 0, "No more dials to process."
        # NOTE: domain prompts
        domain_prompts = []
        for key in meta_data.keys():
            domain_prompts.append(construct_domain_prompt(
                args, key, eval_data[key], turn_idx, ChatCompletion))
        # NOTE: predict domain
        domain_states = predict_domain.run_batch(
            domain_prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            progress_bar=True)
        # NOTE: parse domain
        for task_idx, (task, state) in enumerate(zip(domain_prompts, domain_states)):
            key, prompt = task["key"], task["prompt"]
            assert state.text().startswith(prompt), f"Prompt mismatch: {state.text()} vs {prompt}"
            if task_idx < 1:
                print("###" + state["domain"] + "###")
                pretty_print(state.get_meta_info("domain"), tokenizer)

            eval_turn = eval_data[key][turn_idx]
            turn_domain, functions, current_function = parse_domain(
                args, state["domain"], eval_turn, schema, ChatCompletion)
            eval_in_out[key][turn_idx]["domain"] = {
                "prompt": prompt,
                "output": [{
                    "role": "assistant",
                    "content": state["domain"],
                    "function_call": {},
                    "input_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                    "output_tokens": len(tokenizer.encode(state["domain"], add_special_tokens=False)),
                    }],
                "dspn": eval_turn["dspn"],
                "dspn_gen": turn_domain
            }
            meta_data[key]["turn_domain"] = turn_domain
            meta_data[key]["functions"] = functions
            meta_data[key]["current_function"] = current_function

        # NOTE: dst prompts
        dst_prompts = []
        for key in meta_data.keys():
            dst_prompts.append(construct_dst_prompt(
                args, vital_ns, key, meta_data[key], eval_data[key], turn_idx, ChatCompletion))
        # NOTE: predict dst
        dst_states = predict_dst.run_batch(
            dst_prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            progress_bar=True)
        # NOTE: parse dst
        for task_idx, (task, state) in enumerate(zip(dst_prompts, dst_states)):
            key, prompt = task["key"], task["prompt"]
            in_out = parse_dst(
                args, prompt, state["result"], meta_data[key], ChatCompletion, tokenizer
            )

            eval_turn = eval_data[key][turn_idx]
            bspn_gen = paser_dict_to_bs(meta_data[key]["user_goal"])
            eval_turn["bspn_gen"] = bspn_gen
            eval_turn["bspn_dict_gen"] = deepcopy(meta_data[key]["user_goal"])

            conv_in_out = eval_in_out[key]
            if args.ind_dst or args.divide_inform_confirm or args.track_slot_status:
                conv_in_out[turn_idx]["dst"] = in_out
                conv_in_out[turn_idx]["bspn_dict"] = conv_in_out[turn_idx].pop("bspn_dict")
                conv_in_out[turn_idx]["bspn_dict_gen"] = eval_turn["bspn_dict_gen"]
            else:
                conv_in_out[turn_idx]["dst"] = {
                    "prompt": in_out["prompt"],
                    "output": in_out["output"],
                    "bspn_dict": eval_turn["bspn_dict"],
                    "bspn_dict_gen": eval_turn["bspn_dict_gen"]
                }

        num_completed_turns += len(meta_data)
        d_pbar.set_postfix(
            c_turns=num_completed_turns, n_turns=num_turns
        )

        # NOTE: remove completed dials
        for key in list(meta_data.keys()):
            if len(eval_data[key]) == turn_idx + 1:
                meta_data.pop(key)
                num_completed_dials += 1
                d_pbar.update(1)

        save_data(args, data_ns, vital_ns)

    assert len(meta_data) == 0, f"Number of remaining dials {len(meta_data)} is not zero."
    assert num_turns == num_completed_turns, \
        f"Number of completed turns {num_completed_turns} does not match the total number of turns {num_turns}."
    assert num_completed_dials == len(eval_data), \
        f"Number of completed dials {num_completed_dials} does not match the total number of dials {len(eval_data)}."

    d_pbar.close()


if __name__ == '__main__':
    args = get_args(sgl_infer=True)
    print(args)

    data_ns, vital_ns = load_eval_data(args)
    print_namespace(data_ns)
    print_namespace(vital_ns)

    if args.generate:
        if os.path.isfile(vital_ns.eval_metrics_path):
            print(f"Metrics file '{vital_ns.eval_metrics_path}' already exists, representing that the evaluation has been done.")
            print("If you want to re-generate the evaluation, please remove the existing result and metrics files.\n\n")
            exit(0)
        else:
            main(args, data_ns, vital_ns)
    else:
        not_generate(args, data_ns, vital_ns)

    # run metric evaluation
    run_metric_evaluation(args, data_ns, vital_ns)