# coding: utf-8

import os
import json
import copy
import argparse
from argparse import Namespace
from collections import defaultdict

from src.multiwoz.utils.config import *
from src.multiwoz.utils.reader import MultiWozReader
from src.multiwoz.postprocess import get_data_split, load_schema

EXPERIMENT_DOMAINS = ["[taxi]", "[train]", "[attraction]", "[hotel]", "[restaurant]"]


def load_data(args):
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

    train_data, val_data, test_data = get_data_split(
        dataset_version=args.dataset_version,
        reader=reader,
        n_train=-1,
        n_val=-1,
        n_test=-1,
        return_list=False,
    )

    schema = load_schema(args.dataset_version)

    domain2inform_slots = {}
    for service in schema:
        dn = service["service_name"]
        if f"[{dn}]" in EXPERIMENT_DOMAINS:
            domain2inform_slots[dn] = [slot["name"].split("-")[-1].strip() for slot in service["slots"] if
                                       slot["is_informable"]]
    config_namespace = Namespace(
        domain2inform_slots=domain2inform_slots,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )
    return config_namespace



def init_data(data):
    eval_data = {}
    for dial_id, turns in data.items():
        eval_turns = []
        for turn in turns:
            eval_turn = {}
            for key in [
                "dial_id",
                "turn_num",
                "resp",
                "nodelx_resp",
                "aspn",
                "aspn_dict",
                "user",
                "dspn",
                "bspn",
                "bsdx",
                "bspn_dict",
                "turn_bspn",
                "turn_bspn_dict",
                "db",
                "all_domains",
            ]:
                if key in turn:
                    eval_turn[key] = turn[key]
                elif "dict" in key:
                    eval_turn[key] = {}
                else:
                    eval_turn[key] = ""
            eval_turns.append(eval_turn)
        eval_data[dial_id] = eval_turns
    return eval_data


slot_renaming = {
    "[train]": {
        "arrive": "arrive_by",
        "leave": "leave_at_or_after",
    },
    "[taxi]": {
        "arrive": "arrive_by",
        "leave": "leave_at_or_after",
    },
    "[hotel]": {"type": "accommodation_type", "price": "pricerange"},
    "[restaurant]": {"price": "pricerange"},
}

untracked_slots = {
    "[taxi]": ['car', 'phone', 'reference', 'time'],
    "[train]": ['price', 'time', 'id', 'reference', 'choice'],
    "[attraction]": [
        'address', 'phone', 'postcode', 'open', 'price', 'choice'],
    "[hotel]": ['postcode', 'phone', 'address', 'reference', 'choice'],
    "[restaurant]": ['address', 'reference', 'phone', 'postcode', 'choice']
}


def rename_slots(domain, slot):
    if domain in slot_renaming:
        if slot in slot_renaming[domain]:
            slot = slot_renaming[domain][slot]
    return slot


status2option = {
    "not_informed": "not_informed",
    "user_inform": "user_inform",
    "assistant_inform": "assistant_inform",
    "user_confirm": "user_confirm",
}





def main(args, action_data):
    config = load_data(args)
    dom2slots = config.domain2inform_slots
    src_data = getattr(config, f"{args.split}_data")
    new_data = init_data(src_data)

    # variables for checking
    unconsidered_acts = defaultdict(int)
    unconsidered_act_slots = defaultdict(lambda: defaultdict(lambda : defaultdict(int)))
    dom2aspn_slots = defaultdict(lambda: defaultdict(set))

    assistant_inform_acts = ["[inform]", "[recommend]", "[select]", "[nooffer]", "[nobook]", "[offerbook]", "[offerbooked]"]
    other_acts = ["[request]"]

    for dial_id, turns in new_data.items():
        prev_resp = None
        prev_nodelx_resp = None
        prev_aspn = None
        prev_aspn_dict = None
        for turn_idx, turn in enumerate(turns):
            temp_turn = copy.deepcopy(turn)

            turn["resp"] = prev_resp
            turn["nodelx_resp"] = prev_nodelx_resp
            turn["aspn"] = prev_aspn
            turn["aspn_dict"] = prev_aspn_dict

            turn_svs_dict = {}
            # assign "assistant_inform" to slots in aspn_dict
            if prev_aspn_dict:
                for domain, act_slots in prev_aspn_dict.items():
                    if domain in EXPERIMENT_DOMAINS:
                        for act, slots in act_slots.items():
                            dom2aspn_slots[domain][act].update(slots)
                            if act not in (assistant_inform_acts + other_acts):
                                # print(f"Unconsidered act: {act}")
                                unconsidered_acts[act] += 1
                            if act in assistant_inform_acts:
                                for slot in slots:
                                    slot = rename_slots(domain, slot)
                                    if slot not in dom2slots[domain[1:-1]]:
                                        if slot not in untracked_slots[domain]:
                                            unconsidered_act_slots[domain][act][slot] += 1
                                        continue
                                    if domain not in turn_svs_dict:
                                        turn_svs_dict[domain] = {}
                                    turn_svs_dict[domain][slot] = status2option["assistant_inform"]
            # assign "user_inform" or "user_confirm" to slots in turn_bspn_dict
            # TODO: how to find the slot values that are mentioned in the current user utterance but not in turn_bspn_dict (possibly beacuse those slot values have existed in turn_bspn_dict ?)
            turn_bspn_dict = turn["turn_bspn_dict"]
            for domain, slot2value in turn_bspn_dict.items():
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                for slot, value in slot2value.items():
                    # assert slot in dom2slots[domain[1:-1]], f"Invalid slot: {slot} in {turn_bspn_dict}"
                    if slot not in dom2slots[domain[1:-1]]:
                        print(f"Invalid slot: {slot} in {turn_bspn_dict}")
                        continue
                    if value.strip().lower() in ["none"]:
                        continue
                    if domain not in turn_svs_dict:
                        turn_svs_dict[domain] = {}
                    if slot not in turn_svs_dict[domain]:
                        turn_svs_dict[domain][slot] = status2option["user_inform"]
                    elif turn_svs_dict[domain][slot] == status2option["assistant_inform"]:
                        turn_svs_dict[domain][slot] = status2option["user_confirm"]
                    else:
                        print(f"Unconsidered slot status: {turn_svs_dict[domain][slot]}")

            turn["turn_svs_dict"] = turn_svs_dict

            prev_resp = temp_turn["resp"]
            prev_nodelx_resp = temp_turn["nodelx_resp"]
            prev_aspn = temp_turn["aspn"]
            prev_aspn_dict = temp_turn["aspn_dict"]


    print(f"Unconsidered acts: {json.dumps(unconsidered_acts, indent=4)}", end="\n\n")
    print(f"Unconsidered act-slot pairs: {json.dumps(unconsidered_act_slots, indent=4)}", end="\n\n")
    print("Domain to act-slot pairs:")
    for domain, act_slots in dom2aspn_slots.items():
        print(f"{domain}")
        for act, slots in act_slots.items():
            print(f"\t{act}: {slots}")

    return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_version", type=str, default="2.2",
        choices=["2.0", "2.1", "2.2", "2.3"])
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"])
    args = parser.parse_args()
    print(args)

    action_file_path = f"./data/multiwoz/dialog_acts.json"
    tgt_file_path = f"outputs/multiwoz{args.dataset_version}/{args.split}_slot_source.json"

    with open(action_file_path, "r") as f:
        action_data = json.load(f)

    data = main(args, action_data)

    with open(tgt_file_path, "w") as f:
        json.dump(data, f, indent=4)
