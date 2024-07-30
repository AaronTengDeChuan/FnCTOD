#!/bin/bash

export OPENAI_API_KEY='sk-zrooiSVDXe47pAH5212bDc66E4F9458e9058B6410d49A4D6'
export OPENAI_BASE_URL='https://api.xiaoai.plus/v1/'

devices=0

# dst results of gpt-3.5-0125
dst_result_path=None
#dst_result_path="outputs/multiwoz2.2/test100-multiFalse-refFalse-prevFalse-json-dst0shot-gpt-3.5-0125.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevFalse-json-dst0shot-gpt-3.5-0125.json"

# dst results of zephyr-7b-beta
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-dst5shot-zephyr-7b-beta.json"

# dst results of llama-2-7b-chat
#dst_result_path="outputs/multiwoz2.2/test100-multiFalse-refFalse-prevTrue-json-dst5shot-llama-2-7b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test100-multiFalse-refFalse-prevFalse-json-dst5shot-llama-2-7b-chat.json"

# dst results of llama-2-13b-chat
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevFalse-json-dst5shot-llama-2-13b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-dst5shot-llama-2-13b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-dst2shot-llama-2-13b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-dst3shot-llama-2-13b-chat.json"

# dst results of llama-3.1-8b-instruct
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-dst0shot-llama-3.1-8b-instruct.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-dst1shot-llama-3.1-8b-instruct.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-dst5shot-llama-3.1-8b-instruct.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-dst0shot-llama-3.1-8b-instruct.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-fill-dst1shot-llama-3.1-8b-instruct.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-fill-dst5shot-llama-3.1-8b-instruct.json"


infer_entry="src.multiwoz.inference"
#infer_entry="src.multiwoz.async_inference"


for dataset_version in 2.2
do
    for split in test
    do
        for n_eval in 1000
        do
            for multi_domain in False
            do
                for ref_domain in False
                do
                    for ref_bs in False
                    do
                        for add_prev in False
                        do
                            for task in dst # dst, e2e
                            do
                                for ind_dst in False # False, True
                                do
                                    for dst_nshot in 0
                                    do
                                        for nlg_nshot in 0
                                        do
                                            for function_type in json
                                            do
                                                for model in gpt-3.5-0125 # gpt-3.5-1106 gpt-3.5-0125
                                                do
                                                    CUDA_VISIBLE_DEVICES=$devices python -m ${infer_entry} \
                                                        --dst_result_path ${dst_result_path} \
                                                        --dataset_version $dataset_version \
                                                        --target_domains $target_domains \
                                                        --split $split \
                                                        --n_eval $n_eval \
                                                        --model $model \
                                                        --task $task \
                                                        --ind_dst $ind_dst \
                                                        --divide_inform_confirm False \
                                                        --track_slot_status True \
                                                        --gen_state_channel "bspn_gen" \
                                                        --dst_nshot $dst_nshot \
                                                        --nlg_nshot $nlg_nshot \
                                                        --add_prev $add_prev \
                                                        --ref_domain $ref_domain \
                                                        --ref_bs $ref_bs \
                                                        --multi_domain $multi_domain \
                                                        --function_type $function_type \
#                                                        --generate \
#                                                        --verbose \
#                                                        --debug
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done