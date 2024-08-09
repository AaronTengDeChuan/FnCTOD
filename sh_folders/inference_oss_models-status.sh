#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

export TRANSFORMERS_CACHE='HOME_PATH/.cache/huggingface/transformers'
export HF_HOME='HOME_PATH/.cache/huggingface'

#devices=2,4,6
devices=0

#infer_entry="src.multiwoz.inference"
infer_entry="src.multiwoz.async_inference"
#infer_entry="src.multiwoz.sgl_inference"

network_params=""
# get params from command line in the form of "--key1=value1 --key2=value2 ..."
for arg in "$@"
do
    case $arg in
        --model=*)
            models="${arg#*=}"
            echo "model: $models"
            shift
            ;;
        --host=*)
            host="${arg#*=}"
            network_params="${network_params} --host ${host}"
            echo "host: $host"
            shift
            ;;
        --port=*)
            port="${arg#*=}"
            network_params="${network_params} --port ${port}"
            echo "port: $port"
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

echo "network_params: $network_params"


dst_result_path=None
# dst results of llama-2-13b-chat
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-dst0shot-llama-2-13b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-dst0shot-llama-2-13b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-fill-dst1shot-llama-2-13b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevFalse-json-regex-dst1shot-llama-2-13b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-fill-dst1shot-llama-2-13b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-fill-dst3shot-llama-2-13b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevFalse-json-regex-dst3shot-llama-2-13b-chat.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-fill-dst3shot-llama-2-13b-chat.json"

# dst results of llama-3.1-70b-instruct
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-dst0shot-llama-3.1-70b-instruct.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-regex-dst5shot-llama-3.1-70b-instruct.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-dst0shot-llama-3.1-70b-instruct.json"
#dst_result_path="outputs/multiwoz2.2/test1000-multiFalse-refFalse-prevTrue-json-dst5shot-llama-3.1-70b-instruct.json"

# main DST results
for dataset_version in 2.2 # 2.1 2.2
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
                        for add_prev in False # True False
                        do
                            for task in dst
                            do
                                for dst_nshot in 0 1
                                do
                                    for nlg_nshot in 0
                                    do
                                        for function_type in json # text
                                        do
                                            for model in ${models} # zephyr-7b-beta vicuna-7b-v1.5 vicuna-13b-v1.5 baichuan-13b-chat llama-2-7b-chat llama-2-13b-chat llama-3-8b llama-3-8b-instruct llama-3-70b-instruct llama-3.1-8b-instruct llama-3.1-70b-instruct
                                            do
                                                for regex in True False  # False True
                                                do
                                                    for fill_inactive in True False  # False True
                                                    do
                                                        for status_option in abstract concrete # abstract concrete
                                                        do
                                                        CUDA_VISIBLE_DEVICES=$devices python -m ${infer_entry} \
                                                            ${network_params} \
                                                            --regex $regex \
                                                            --fill_inactive $fill_inactive \
                                                            --dst_result_path ${dst_result_path} \
                                                            --dataset_version $dataset_version \
                                                            --target_domains $target_domains \
                                                            --split $split \
                                                            --n_eval $n_eval \
                                                            --model $model \
                                                            --task $task \
                                                            --track_slot_status True \
                                                            --ind_status False \
                                                            --no_enum True \
                                                            --status_option $status_option \
                                                            --status_context_window 2 \
                                                            --temperature 0.3 \
                                                            --gen_state_channel "bspn_gen" \
                                                            --dst_nshot $dst_nshot \
                                                            --nlg_nshot $nlg_nshot \
                                                            --add_prev $add_prev \
                                                            --ref_domain $ref_domain \
                                                            --ref_bs $ref_bs \
                                                            --multi_domain $multi_domain \
                                                            --function_type $function_type \
                                                            --generate \
        #                                                    --verbose \
        #                                                    --debug
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
    done
done

exit 0

# e2e with generated bs on multiwoz 2.2
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
                        for add_prev in True
                        do
                            for task in e2e # dst nlg
                            do
                                for dst_nshot in 5
                                do
                                    for nlg_nshot in 5
                                    do
                                        for function_type in json # text
                                        do
                                            for model in zephyr-7b-beta baichuan-13b-chat vicuna-7b-v1.5 vicuna-13b-v1.5 llama-2-7b-chat llama-2-13b-chat llama-2-13b-chat
                                            do
                                                CUDA_VISIBLE_DEVICES=$devices python -m src.multiwoz.inference \
                                                                                        --dataset_version $dataset_version \
                                                                                        --target_domains $target_domains \
                                                                                        --split $split \
                                                                                        --n_eval $n_eval \
                                                                                        --model $model \
                                                                                        --task $task \
                                                                                        --dst_nshot $dst_nshot \
                                                                                        --nlg_nshot $nlg_nshot \
                                                                                        --add_prev $add_prev \
                                                                                        --ref_domain $ref_domain \
                                                                                        --ref_bs $ref_bs \
                                                                                        --multi_domain $multi_domain \
                                                                                        --function_type $function_type \
                                                                                        --generate \
                                                                                        --verbose \
                                                                                        # --debug
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