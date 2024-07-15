#!/bin/python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
from sklearn.metrics import f1_score, accuracy_score
import sys
import numpy as np
import argparse
from collections import defaultdict

from src.multiwoz.utils.dst import (
    ignore_none,
    ignore_dontcare,
    default_cleaning,
    IGNORE_TURNS_TYPE2,
    paser_bs,
)
from src.multiwoz.utils.utils import informable_slots


def zip_result(prediction):
    result = {}
    for turn in prediction:
        dial_id = turn["dial_id"]
        turn_idx = turn["turn_num"]
        try:
            result[dial_id][turn_idx] = turn
        except KeyError:
            result[dial_id] = {}
            result[dial_id][turn_idx] = turn
    return result


def paser_per_domain_bs(bs_list):
    per_domain_bs = {
        "[restaurant]": [],
        "[taxi]": [],
        "[hotel]": [],
        "[attraction]": [],
        "[train]": [],
    }
    for bs in bs_list:
        domain, slot, value = bs.split("->")
        if domain in per_domain_bs and value:
            per_domain_bs[domain].append(f"{slot}->{value}")
    return per_domain_bs


def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = (
            2 * precision * recall / float(precision + recall)
            if (precision + recall) != 0
            else 0
        )
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count


def compute_overlap(gold, pred, accm_dict):
    def get_slot(bs):
        return bs.rsplit("->", maxsplit=1)[0]
    gold_slots = set(get_slot(bs) for bs in gold)
    pred_slots = set(get_slot(bs) for bs in pred)

    accm_dict["match"] += int(gold_slots == pred_slots)
    accm_dict["value_match"] += int(set(gold) == set(pred))
    accm_dict["cover"] += int(gold_slots.issubset(pred_slots))
    accm_dict["value_cover"] += int(set(gold).issubset(set(pred)))
    accm_dict["tot"] += 1
    accm_dict["valid_tot"] += int(len(gold) > 0)
    tp, fp, fn = len(gold_slots & pred_slots), len(pred_slots - gold_slots), len(gold_slots - pred_slots)
    accm_dict["tp"] += tp
    accm_dict["fp"] += fp
    accm_dict["fn"] += fn
    precision = tp / (tp + fp) if tp != 0 else 0
    recall = tp / (tp + fn) if tp != 0 else 0
    accm_dict["macro_precision"] += precision
    accm_dict["macro_recall"] += recall
    accm_dict["macro_f1"] += 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    accm_dict["macro_value_recall"] += len(set(gold) & set(pred)) / len(set(gold)) if len(set(gold)) != 0 else 0


def postprocess_overlap_accm(overlap_accm_dict):
    # overlap
    overlap_accm_dict["match_ratio"] = overlap_accm_dict["match"] / overlap_accm_dict["tot"]
    overlap_accm_dict["value_match_ratio"] = overlap_accm_dict["value_match"] / overlap_accm_dict["tot"]
    overlap_accm_dict["cover_ratio"] = overlap_accm_dict["cover"] / overlap_accm_dict["tot"]
    overlap_accm_dict["value_cover_ratio"] = overlap_accm_dict["value_cover"] / overlap_accm_dict["tot"]
    overlap_accm_dict["micro_precision"] = overlap_accm_dict["tp"] / (
            overlap_accm_dict["tp"] + overlap_accm_dict["fp"]) if overlap_accm_dict["tp"] != 0 else 0
    overlap_accm_dict["micro_recall"] = overlap_accm_dict["tp"] / (
            overlap_accm_dict["tp"] + overlap_accm_dict["fn"]) if overlap_accm_dict["tp"] != 0 else 0
    overlap_accm_dict["micro_f1"] = 2 * overlap_accm_dict["micro_precision"] * overlap_accm_dict["micro_recall"] / (
            overlap_accm_dict["micro_precision"] + overlap_accm_dict["micro_recall"]) if (
            (overlap_accm_dict["micro_precision"] + overlap_accm_dict["micro_recall"]) != 0) else 0
    overlap_accm_dict["macro_precision"] /= overlap_accm_dict["tot"]
    overlap_accm_dict["macro_recall"] /= overlap_accm_dict["tot"]
    overlap_accm_dict["macro_f1"] /= overlap_accm_dict["tot"]
    overlap_accm_dict["macro_value_recall"] /= overlap_accm_dict["valid_tot"]


def compute_jacc(
    data,
    gen_state_channel="bspn_gen",
    default_cleaning_flag=True,
    type2_cleaning_flag=False,
    ignore_dontcare_in_pred=False,
):
    num_turns = 0
    recall, total_target = 0, 0
    precision, total_pred = 0, 0
    joint_acc = 0
    per_domain_jga = {
        "[restaurant]": [0, 0, 0, 0],  # accuracy, f1
        "[taxi]": [0, 0, 0, 0],  # accuracy, f1
        "[hotel]": [0, 0, 0, 0],  # accuracy, f1
        "[attraction]": [0, 0, 0, 0],  # accuracy, f1
        "[train]": [0, 0, 0, 0],  # accuracy, f1
    }
    avg_domain_jga = {
        "micro_jga": [0, 0, 0],
        "macro_jga": [0, 0, 0],
        "micro_f1": [0, 0, 0],
        "macro_f1": [0, 0, 0],
    }
    per_slot_acc = {
        "[restaurant]": {},
        "[taxi]": {},
        "[hotel]": {},
        "[attraction]": {},
        "[train]": {},
    }
    overlap_accm_dict = defaultdict(lambda : 0)
    per_domain_overlap = defaultdict(lambda : defaultdict(lambda : 0))

    error = {}

    clean_tokens = ["<|endoftext|>"]
    for file_name in data:
        all_domains = list(data[file_name].values())[-1]["all_domains"]
        for turn_id, turn_data in data[file_name].items():
            turn_target = turn_data["bspn"]
            turn_pred = turn_data[gen_state_channel]
            turn_target = paser_bs(turn_target)
            turn_pred = paser_bs(turn_pred)
            # clean
            for bs in turn_pred:
                if bs in clean_tokens + ["", " "] or bs.split("->")[-1] == "none":
                    turn_pred.remove(bs)

            new_turn_pred = []
            for bs in turn_pred:
                for tok in clean_tokens:
                    bs = bs.replace(tok, "").strip()
                    new_turn_pred.append(bs)
            turn_pred = new_turn_pred

            turn_pred, turn_target = ignore_none(turn_pred, turn_target)

            # MultiWOZ default cleaning
            if default_cleaning_flag:
                turn_pred, turn_target = default_cleaning(turn_pred, turn_target)

            turn_pred, turn_target = ignore_none(turn_pred, turn_target)
            if ignore_dontcare_in_pred:
                turn_pred = ignore_dontcare(turn_pred)
                turn_target = ignore_dontcare(turn_target)

            # precision
            wrong_predbs = []
            for p in list(set(turn_pred)):
                if p in list(set(turn_target)):
                    precision += 1
                else:
                    wrong_predbs.append(p)
                total_pred += 1

            # recall
            missed_targets = []
            for p in list(set(turn_target)):
                if p in list(set(turn_pred)):
                    recall += 1
                else:
                    missed_targets.append(p)
                total_target += 1

            # per domain joint accuracy
            per_domain_turn_target = paser_per_domain_bs(turn_target)
            per_domain_turn_pred = paser_per_domain_bs(turn_pred)
            for domain in per_domain_jga:
                """
                Per-domain JGA
                """
                if domain in all_domains:
                    # F1
                    per_domain_jga[domain][1] += 1
                    if set(per_domain_turn_target[domain]) == set(
                        per_domain_turn_pred[domain]
                    ):
                        per_domain_jga[domain][0] += 1
                    # F1
                    temp_f1, temp_r, temp_p, count = compute_prf(
                        set(per_domain_turn_target[domain]),
                        set(per_domain_turn_pred[domain]),
                    )
                    per_domain_jga[domain][2] += temp_f1
                    per_domain_jga[domain][3] += count

                    compute_overlap(
                        per_domain_turn_target[domain],
                        per_domain_turn_pred[domain],
                        per_domain_overlap[domain]
                    )


            # per slot accuracy
            per_domain_turn_target = paser_per_domain_bs(turn_target)
            per_domain_turn_pred = paser_per_domain_bs(turn_pred)
            for domain in per_slot_acc:
                if per_domain_turn_target[domain]:
                    for target_bs in per_domain_turn_target[domain]:
                        slot, value = target_bs.split("->")
                        if slot not in per_slot_acc[domain]:
                            per_slot_acc[domain][slot] = [0, 0]
                        per_slot_acc[domain][slot][1] += 1

                        if domain in per_domain_turn_pred:
                            for pred_bs in per_domain_turn_pred[domain]:
                                if pred_bs == target_bs:
                                    # correct prediction
                                    per_slot_acc[domain][slot][0] += 1

            # MULTI-DOMAIN Joint Accuracy
            join_flag = False
            if set(turn_target) == set(turn_pred):
                joint_acc += 1
                join_flag = True

            elif type2_cleaning_flag:  # check for possible Type 2 noisy annotations
                flag = True
                for bs in turn_target:
                    if bs not in turn_pred:
                        flag = False
                        break
                if flag:
                    for bs in turn_pred:
                        if bs not in turn_target:
                            flag = False
                            break

                if (
                    flag
                ):  # model prediction might be correct if found in Type 2 list of noisy annotations
                    dial_name = dial.split(".")[0]
                    if (
                        dial_name in IGNORE_TURNS_TYPE2
                        and turn_id in IGNORE_TURNS_TYPE2[dial_name]
                    ):  # ignore these turns
                        pass
                    else:
                        joint_acc += 1
                        join_flag = True

            if not join_flag:
                if file_name not in error:
                    error[file_name] = {}
                turn_data["gtbs"] = turn_target
                turn_data["predbs"] = turn_pred
                turn_data["wrong_predbs"] = wrong_predbs
                turn_data["missed_targets"] = missed_targets
                error[file_name][turn_id] = turn_data

            # Overlap
            compute_overlap(turn_target, turn_pred, overlap_accm_dict)

            num_turns += 1

    joint_acc /= num_turns
    precision = precision / total_pred if total_pred > 0 else 0
    recall = recall / total_target if total_target > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    # per domain jga
    for domain in per_domain_jga:
        # micro avg jga over all domains
        avg_domain_jga["micro_jga"][0] += per_domain_jga[domain][0]
        avg_domain_jga["micro_jga"][1] += per_domain_jga[domain][1]
        avg_domain_jga["micro_f1"][0] += per_domain_jga[domain][2]
        avg_domain_jga["micro_f1"][1] += per_domain_jga[domain][3]

        per_domain_jga[domain][0] = (
            (per_domain_jga[domain][0] / per_domain_jga[domain][1])
            if per_domain_jga[domain][1] > 0
            else 0
        )  # JGA
        per_domain_jga[domain][2] = (
            (per_domain_jga[domain][2] / per_domain_jga[domain][3])
            if per_domain_jga[domain][3] > 0
            else 0
        )  # F1
        # macro avg jga over all domains
        avg_domain_jga["macro_jga"][0] += per_domain_jga[domain][0]
        avg_domain_jga["macro_jga"][1] += 1
        avg_domain_jga["macro_f1"][0] += per_domain_jga[domain][2]
        avg_domain_jga["macro_f1"][1] += 1

    # avg domain jga
    for mn, mv in avg_domain_jga.items():
        mv[2] = mv[0] / mv[1] if mv[1] > 0 else 0

    # per slot jga
    for domain in per_slot_acc:
        for slot in per_slot_acc[domain]:
            per_slot_acc[domain][slot][0] = (
                (per_slot_acc[domain][slot][0] / per_slot_acc[domain][slot][1])
                if per_slot_acc[domain][slot][1] > 0
                else 0
            )

    postprocess_overlap_accm(overlap_accm_dict)
    for domain in per_domain_overlap:
        postprocess_overlap_accm(per_domain_overlap[domain])

    # print('joint accuracy: {}, f1: {}, precision: {}, recall: {}'.format(joint_acc, f1, precision, recall))
    return (joint_acc, f1, precision, recall, per_domain_jga, avg_domain_jga, per_slot_acc, error,
            overlap_accm_dict, per_domain_overlap)
