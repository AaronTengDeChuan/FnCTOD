# coding: utf-8

import os
import json
import argparse
import pandas as pd
from itertools import product
from collections import defaultdict
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, computed_field

# display settings: center, column separator, float precision
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.colheader_justify', 'center')
# pd.set_option('display.precision', 2)
pd.set_option('display.float_format', lambda x: f"{x:.1f}")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_version", type=str, default="2.2", choices=["2.0", "2.1", "2.2", "2.3"])
parser.add_argument("--track_slot_status", action="store_true", default=False)
parser.add_argument("--latex_text_out", type=str, default="temp_results/latex_tables.txt")
parser.add_argument("--pandas_out", type=str, default="temp_results/tables.csv")
args = parser.parse_args()


class Domain_Metric(BaseModel):
    Acc: float = Field(default=None, alias="accuracy")


class per_domain_jga(BaseModel):
    Attr: Any = Field(default=None, alias="[attraction]")
    Hotel: Any = Field(default=None, alias="[hotel]")
    Rest: Any = Field(default=None, alias="[restaurant]")
    Taxi: Any = Field(default=None, alias="[taxi]")
    Train: Any = Field(default=None, alias="[train]")

    @field_validator("Attr", "Hotel", "Rest", "Taxi", "Train")
    def check_jga(cls, v, values):
        if isinstance(v, list) and len(v) == 4:
            return v[0]
        else:
            raise ValueError(f"Invalid jga value: {v}")


class avg_domain_jga(BaseModel):
    Mic: Any = Field(default=None, alias="micro_jga")
    Mac: Any = Field(default=None, alias="macro_jga")

    @field_validator("Mic", "Mac")
    def check_jga(cls, v, values):
        if isinstance(v, list) and len(v) == 3:
            return v[2]
        else:
            raise ValueError(f"Invalid jga value: {v}")


class turn_level_active_slot_overlap(BaseModel):
    MR: float = Field(default=None, alias="match_ratio")
    CR: float = Field(default=None, alias="cover_ratio")
    if not args.track_slot_status:
        VCR: float = Field(default=None, alias="value_cover_ratio")


class per_domain_overlap(BaseModel):
    Attr: turn_level_active_slot_overlap = Field(default_factory=turn_level_active_slot_overlap, alias="[attraction]")
    Hotel: turn_level_active_slot_overlap = Field(default_factory=turn_level_active_slot_overlap, alias="[hotel]")
    Rest: turn_level_active_slot_overlap = Field(default_factory=turn_level_active_slot_overlap, alias="[restaurant]")
    Taxi: turn_level_active_slot_overlap = Field(default_factory=turn_level_active_slot_overlap, alias="[taxi]")
    Train: turn_level_active_slot_overlap = Field(default_factory=turn_level_active_slot_overlap, alias="[train]")


class DST_Metric(BaseModel):
    JGA: float = Field(default=None, alias="joint_acc")
    if args.track_slot_status:
        Dom_Slot_Overlap: per_domain_overlap = Field(default_factory=per_domain_overlap, alias="per_domain_overlap")
        # Avg_SO: per_domain_overlap = Field(default_factory=turn_level_active_slot_overlap, alias="per_domain_overlap")

        # @field_validator("Dom_Slot_Overlap")
        # def post_process_dom_slot_overlap(cls, v, values):
        #     delete_keys = ["CR"]
        #     for key in delete_keys:
        #         for dom, overlap in v.dict().items():
        #             delattr(getattr(v, dom), key)
        #     return v

        @computed_field(return_type=turn_level_active_slot_overlap)
        @property
        def Avg_SO(self):
            new_v = turn_level_active_slot_overlap()
            for key in new_v.dict().keys():
                vs = [v[key] for v in self.Dom_Slot_Overlap.dict().values() if isinstance(v[key], float)]
                if vs:
                    assert len(vs) == len(self.Dom_Slot_Overlap.dict()), f"Invalid values: {vs}"
                    new_v.__setattr__(key, sum(vs) / len(vs))
            return new_v
    else:
        Dom_JGA: per_domain_jga = Field(default_factory=per_domain_jga, alias="per_domain_jga")
        Avg_JGA: avg_domain_jga = Field(default_factory=avg_domain_jga, alias="avg_domain_jga")
    TASO: turn_level_active_slot_overlap = Field(
        default_factory=turn_level_active_slot_overlap, alias="turn_level_active_slot_overlap")


class Metric(BaseModel):
    Dom: Domain_Metric = Field(default_factory=Domain_Metric, alias="Domain")
    DST: DST_Metric = Field(default_factory=DST_Metric, alias="DST")


def collect_metrics(eval_metrics_file):
    if os.path.exists(eval_metrics_file):
        with open(eval_metrics_file, "r") as f:
            eval_metrics = json.load(f)
        eval_metrics = Metric.model_validate(eval_metrics)
        print(f"[Loaded] Metrics file '{eval_metrics_file}'.")
    else:
        eval_metrics = Metric()
        print(f"Metrics file '{eval_metrics_file}' does not exist.")
    return eval_metrics


def flatten_metrics(metrics):
    keys = []
    values = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            sub_keys, sub_values = flatten_metrics(value)
            keys.extend([f"{key}.{sub_key}" for sub_key in sub_keys])
            values.extend(sub_values)
        else:
            keys.append(key)
            values.append(value if key == "JGA" or not isinstance(value, (int, float)) else value * 100)
    return keys, values


def construct_multi_index(metrics):
    col_tuples = []
    key2span = {}
    span_start = 0
    for key, value in metrics.items():
        if isinstance(value, dict):
            sub_col_tuples, sub_key2span = construct_multi_index(value)
            col_tuples.extend([(key, *sub_key) for sub_key in sub_col_tuples])
            for sub_key, sub_span in sub_key2span.items():
                key2span[f"{key} -> {sub_key}"] = (span_start + sub_span[0], span_start + sub_span[1])
            key2span[key] = (span_start, span_start + len(sub_col_tuples) - 1)
            span_start += len(sub_col_tuples)
        else:
            col_tuples.append((key,))
            key2span[key] = (span_start, span_start)
            span_start += 1

    return col_tuples, key2span


idx2tag = {
    0: "✔", # \usym{1F5F8} \Checkmark \faCheckSquareO
    1: "✘" # \ding{55} \XSolidBrush \faTimesCircleO
}

models = [
    "llama-2-13b-chat", "zephyr-7b-beta",
    "llama-3-8b-instruct", "llama-3-70b-instruct",
    "llama-3.1-8b-instruct", "llama-3.1-70b-instruct"
]


def shortstack(text):
    stack_text = "\\\\".join(text)
    return f"\\shortstack{{{stack_text}}}"


replace_dict = {
    idx2tag[0]: "\\Checkmark", # "\\usym{1F5F8}",
    idx2tag[1]: "\\XSolidBrush", # "\\ding{55}"
    "\\multirow[t]": "\\multirow",
    "\\cline": "\\cmidrule",
    # **{model: shortstack(model) for model in models}
}


def render_latex_table(metrics, row_tuples, col_names):
    pd.set_option('styler.latex.multicol_align', 'c')
    pd.set_option('styler.latex.multirow_align', 'c')
    # pd.set_option('styler.sparse.columns', False)

    model = ''
    models = set([row[0] for row in row_tuples])
    print(models)
    if len(models) == 1:
        model = f" {models.pop()}"
        row_tuples = [row[1:] for row in row_tuples]
        col_names = col_names[1:]

    # flatten the metrics
    keys, values = zip(*[(flatten_metrics(metric)) for metric in metrics])
    keys = list(map(lambda x: '; '.join(x), keys))
    assert len(set(keys)) == 1, f"The keys of metrics should be the same: {json.dumps(keys, indent=4)}"
    # construct multi-index
    col_tuples, key2span = construct_multi_index(metrics[0])
    print(col_tuples)

    offset = 1 + len(row_tuples[0])
    level2spans = defaultdict(str)
    for key, span in key2span.items():
        level = key.count(" -> ")
        level2spans[level] += f"\\cmidrule(lr){{{span[0] + offset}-{span[1] + offset}}} "
    print(level2spans)

    metric_df = pd.DataFrame(
        values,
        columns=pd.MultiIndex.from_tuples(col_tuples),
        index=pd.MultiIndex.from_tuples(row_tuples, names=col_names)
    )
    # replace NaN with '-'
    metric_df = metric_df.fillna("-")
    print(metric_df)


    latex_table_text = metric_df.to_latex(
        caption=f"Evaluation results of{model} on MultiWOZ 2.2",
        multicolumn=True, multirow=True,
        column_format='c' * (len(col_tuples) + len(row_tuples[0])), float_format="%.1f", escape=True)

    if "\\centering" not in latex_table_text:
        latex_table_text = latex_table_text.replace("\\begin{table}\n", "\\begin{table}\n\\centering\n")

    level_1_end = latex_table_text.find("\\\\")
    assert level_1_end != -1
    latex_table_text = latex_table_text[:level_1_end] + f"\\\\{level2spans[0].strip()}" + latex_table_text[level_1_end + 2:]

    level_2_end = latex_table_text.find("\\\\", level_1_end + 2)
    assert level_2_end != -1
    latex_table_text = latex_table_text[:level_2_end] + f"\\\\{level2spans[1].strip()}" + latex_table_text[level_2_end + 2:]

    # for linerule in ["toprule", "midrule", "bottomrule"]:
    #     latex_table_text = latex_table_text.replace(f"\\{linerule}", f"\\hline")
    for before, after in replace_dict.items():
        latex_table_text = latex_table_text.replace(before, after)

    if models:
        for model in models:
            latex_table_text = latex_table_text.replace(model, shortstack(model))

    return metric_df, latex_table_text


def enumerate_hyperparameters(args):
    if args.track_slot_status:
        channels = [
            ["-regex", ""],
            ["-fill", ""],
            ["abs_", ""],
            [0, 1]
        ]
        col_names = ["M", "Re", "F", "OT", "K"]
        args.latex_text_out = os.path.join(os.path.dirname(args.latex_text_out), "status_" + os.path.basename(args.latex_text_out))
        args.pandas_out = os.path.join(os.path.dirname(args.pandas_out), "status_" + os.path.basename(args.pandas_out))
    else:
        channels = [
            ["True", "False"],
            ["-regex", ""],
            ["-fill", ""],
            [0, 1, 3, 5],
        ]
        col_names = ["M", "Pr", "Re", "F", "K"]

    all_metrics, all_row_tuples = [], []
    dataframes = []
    latex_tables = []
    for model in models[:]:
        metrics = []
        row_tuples = []
        # product
        for config in product(*channels):
            if args.track_slot_status:
                r, f, o, s = config
                metric_file = f"outputs/multiwoz{args.dataset_version}/test1000-multiFalse-refFalse-prevFalse-json{r}{f}-{o}status_dst{s}shot-nlg0shot-{model}-metrics.json"
            else:
                p, r, f, s = config
                metric_file = f"outputs/multiwoz{args.dataset_version}/test1000-multiFalse-refFalse-prev{p}-json{r}{f}-dst{s}shot-nlg0shot-{model}-metrics.json"
            metric = collect_metrics(metric_file)
            metrics.append(metric.dict())
            row_tuples.append((
                model, *[idx2tag[channels[i].index(c)] for i, c in enumerate(config) if isinstance(c, str)],
                config[-1]))
        all_metrics.extend(metrics)
        all_row_tuples.extend(row_tuples)
        df, table = render_latex_table(metrics, row_tuples, col_names)
        dataframes.append(df)
        latex_tables.append(table)

    os.makedirs(os.path.dirname(args.latex_text_out), exist_ok=True)
    with open(args.latex_text_out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(latex_tables))

    total_df, _ = render_latex_table(all_metrics, all_row_tuples, col_names)
    total_df.to_csv(args.pandas_out, index=True, header=True)


if __name__ == '__main__':

    enumerate_hyperparameters(args)
    exit(0)

    metric_file = "outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-fill-dst1shot-nlg0shot-llama-2-13b-chat-metrics.json"

    metrics = collect_metrics(metric_file)
    print(json.dumps(metrics.dict(), indent=4))

    print(render_latex_table(metrics))


    # 将多级索引的列名转换为单行字段名
    # 重置索引，并将列名作为新的索引值
    # 输出转换后的单行字段名
    # metric_df.columns = metric_df.columns.map(lambda x: '_'.join([i for i in x if isinstance(i, str)]))
    # metric_df = metric_df.reset_index(drop=True)
    # print(metric_df)
