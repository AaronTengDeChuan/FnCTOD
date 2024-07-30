# coding: utf-8
import os.path
import time
import asyncio
import aiohttp
from collections import OrderedDict

from src.multiwoz.inference_auxiliaries import *

# TODO: There are two ways to get regex - one from Pydantic (define BaseModel and then build_regex_from_object) and one from manually writing regex.


async def monitor(args, data_ns, vital_ns):
    save_interval = 10
    min_seconds = 30

    num_dials = len(data_ns.eval_data)
    num_turns = sum([len(turns) for turns in data_ns.eval_data.values()])
    d_pbar = tqdm(total=num_dials, desc=f"Evaluation {args.split}")
    last_saved = 0
    last_time = None

    def _update_pbar():
        d_pbar.set_postfix(OrderedDict(
            c_turns=global_counter.completed_turns,
            n_turns=num_turns,
            reqs=global_counter.completed_requests,
            saved_dials=last_saved,
        ))
        d_pbar.update(global_counter.completed_dials - d_pbar.n)

    # monitor the completion process
    while global_counter.completed_dials < num_dials:
        if (global_counter.completed_dials - last_saved >= save_interval
                and (last_time is None or time.time() - last_time >= min_seconds)):
            save_data(args, data_ns, vital_ns)
            last_saved = global_counter.completed_dials
            last_time = time.time()
        _update_pbar()
        await asyncio.sleep(1)
    save_data(args, data_ns, vital_ns)
    last_saved = global_counter.completed_dials
    _update_pbar()
    d_pbar.close()


async def validate_model_info(base_url, specified_model, num_dials):
    url = f"{base_url}/get_model_info"
    current_base = os.path.basename(specified_model)
    while global_counter.completed_dials < num_dials:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response = await response.json()
                remote_base = os.path.basename(response["model_path"])
                if remote_base != current_base:
                    raise ValueError(f"Remote model '{response['model_path']}' is different from current model '{specified_model}', please check the remote server.")
        await asyncio.sleep(10)


async def core_task(
        args, vital_ns, key, eval_turns, conv_in_out, ChatCompletion, semaphore=None):
    user_goal = {}
    inform_user_goal = {}
    confirm_user_goal = {}

    for cur_turn_idx, eval_turn in enumerate(eval_turns):
        # async with semaphore:
        user_goal, inform_user_goal, confirm_user_goal = await infer_each_turn(
            args, vital_ns, key, eval_turns, conv_in_out, cur_turn_idx, ChatCompletion,
            user_goal, inform_user_goal, confirm_user_goal, async_mode=True
        )
        global_counter.completed_turns += 1
    global_counter.completed_dials += 1


async def main(args, data_ns, vital_ns):

    eval_data = data_ns.eval_data
    eval_in_out = data_ns.eval_in_out

    semaphore = asyncio.Semaphore(200)

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
        counter=global_counter,
        semaphore=semaphore,
    )
    base_url = ChatCompletion.base_url

    # create task
    monitor_task = asyncio.create_task(monitor(args, data_ns, vital_ns))
    validate_task = asyncio.create_task(validate_model_info(base_url, ChatCompletion.model_name, len(eval_data)))
    core_tasks = [core_task(args, vital_ns, key, eval_data[key], eval_in_out[key], ChatCompletion) for key in eval_data.keys()]
    await asyncio.gather(*core_tasks)
    await monitor_task
    await validate_task


if __name__ == '__main__':
    global_counter = Namespace(
        completed_requests=0,
        completed_dials=0,
        completed_turns=0,
    )

    args = get_args()
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
            asyncio.run(main(args, data_ns, vital_ns))
    else:
        not_generate(args, data_ns, vital_ns)

    # run metric evaluation
    run_metric_evaluation(args, data_ns, vital_ns)