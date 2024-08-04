import json
import os
import socket
import regex as re
import pynvml
from subprocess import PIPE
import time
import wandb
import argparse
from psutil import Popen

LLMC_COMMAND_MAPPINGS = {
    "train_file": "-i",
    "dev_file": "-j",
    "out_dir": "-o",
    "val_loss_every": "-v",
    "sample_every": "-s",
    "num_tokens_to_sample": "-g",
    "run_hellaswag": "-h",
    "batch_size": "-b",
    "total_tokens_per_step": "-d",
    "seq_len": "-t",
    "use_act_recomputation": "-r",
    "weight_decay": "-c",
    "learning_rate": "-l",
    "lr_decay_final_percentage": "-q",
    "lr_warmup_iters": "-u",
    "checkpoint_every": "-n",
    "resume_optimization": "-y",
    "load_weights": "-e",
    "max_loss_outlier": "-sl",
    "max_norm_outlier": "-sg",
    "gelu_fusion": "-ge"
}

REBUILD_LLMC_SCRIPT = """
make clean
rm -r build/
rm train_gpt2cu
make train_gpt2cu
"""


FORCE_RECOMPILE_LLMC = True

def main():
    # argparse json run config path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help="path to json training run configuration file")
    args = parser.parse_args()

    if args.config_path is None:
        # print help
        parser.print_help()
        return

    # load json run config
    with open(args.config_path) as f:
        config = json.load(f)

    # compiling the llm.c program
    if not os.path.exists("train_gpt2cu") or FORCE_RECOMPILE_LLMC:
        process = Popen(
            "export NO_MULTI_GPU=" + ("1" if config["compile_no_multi_gpu"] else "0") + "\n" +
            "export USE_CUDNN=" + ("1" if config["compile_use_cudnn"] else "0") + "\n" +
            "export FP_PRECISION=" + config["fp_precision"] + "\n" +
            REBUILD_LLMC_SCRIPT, shell=True, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True)
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()
        print("Finished compiling llm.c")
    else:
        print("train_gpt2cu already compiled, skipping...")

    # if multi-node, copy to all other nodes
    if not config["compile_no_multi_gpu"]:
        for node_host, node_config in config["nodes"].items():
            if node_config.get("is_master", False):
                continue
            
            # copy ./train_gpt2cu to all other nodes
            abs_path = os.path.abspath("./train_gpt2cu")
            print(f"Copying {abs_path} to {node_host}...")
            os.system(f"scp {abs_path} {node_host}:{abs_path}")

    pynvml.nvmlInit()

    # apply overclock
    if config["apply_overclock"]:
        for gpu_name, overclock in config["overclocks"].items():
            # apply overclock to gpus with the specified name
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                if gpu_name == pynvml.nvmlDeviceGetName(handle):
                    core_mhz_plus = overclock["core_mhz+"]
                    mem_mhz_plus = overclock["mem_mhz+"]
                    print(f"Applied overclock to {gpu_name} with core_mhz+={core_mhz_plus} and mem_mhz+={mem_mhz_plus}")

                    try:
                        pynvml.nvmlDeviceSetGpcClkVfOffset(handle, core_mhz_plus)
                        pynvml.nvmlDeviceSetMemClkVfOffset(handle, mem_mhz_plus)
                    except pynvml.NVMLError as e:
                        print(f"Failed to apply overclock to {gpu_name}: {e}")
   
    # force fan speed to 100%
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        # iterate over all fans
        fan_count = pynvml.nvmlDeviceGetNumFans(handle)
        for fan_index in range(fan_count):
            try:
                pynvml.nvmlDeviceSetFanSpeed_v2(handle, fan_index, 100)
            except pynvml.NVMLError as err:
                print(f"Failed to set fan speed of fan {fan_index} of GPU {i} to 100%")

    # create log dir
    log_dir = config["out_dir"]
    os.makedirs(log_dir, exist_ok=True)

    # build command
    if config["compile_no_multi_gpu"]:
        command = ["./train_gpt2cu"]
    else:
        host_str = ""
        for node_name, node_config in config["nodes"].items():
            num_gpus = node_config["num_gpus"]

            if len(host_str) > 0:
                host_str += ","
            host_str += f"{node_name}:{num_gpus}"
        
        command = ['mpirun', '-np', str(config['num_total_gpus']), "--host", host_str, './train_gpt2cu']
    
    for key, value in config.items():
        if key in LLMC_COMMAND_MAPPINGS:
            command.append(LLMC_COMMAND_MAPPINGS[key])
            command.append(str(value))

    if not config["compile_no_multi_gpu"]:
        command.extend(["-pi", "mpi"])

    enable_wandb = config.get("enable_wandb", False)

    if enable_wandb:
        # init wandb
        wandb.init(project=config["wandb_project"], config=config)

    # run command
    print("Launching: " + " ".join(command))
    
    env = os.environ.copy()
    env["NCCL_DEBUG"] = "INFO"

    process = Popen(command, shell=False, stdout=PIPE, stderr=PIPE,
                    env=env)
    # intercept stdout and stderr in real-time in a blocking manner
    trainstep_pattern = re.compile(
        r"step +(\d+)/(\d+) \| loss ([+-]?[na]*[0-9]*\.?[0-9]*[eE]?[+-]?[0-9]*?) \([+-]?[na]*[0-9]*\.?[0-9]*[eE]?[+-]?[0-9]*?z\)\| norm ([+-]?[na]*[0-9]*\.?[0-9]*[eE]?[+-]?[0-9]*?) \([+-]?[na]*[0-9]*\.?[0-9]*[eE]?[+-]?[0-9]*?z\)\| lr ([+-]?[na]*[0-9]*\.?[0-9]*[eE]?[+-]?[0-9]*?) \| ([+-]?[na]*[0-9]*\.?[0-9]*[eE]?[+-]?[0-9]*?) ms \| ([+-]?[na]*[0-9]*\.?[0-9]*[eE]?[+-]?[0-9]*?)% b?fp?\d\d MFU \| (\d+) tok/s"
    )
    hella_swag_pattern = re.compile(r"HellaSwag: +(\d+)/(\d+) = ([+-]?[0-9]*\.?[0-9]*[eE]?[+-]?[0-9]+?)")
    val_loss_pattern = re.compile(r"val loss ([+-]?[0-9]*\.?[0-9]*[eE]?[+-]?[0-9]+?)")
    total_steps = -1
    
    step = -1
    stdout = process.stdout
    
    line_bytes = bytearray()

    while True:
        stdout_bytes = stdout.read(1)
        
        if len(stdout_bytes) == 0:
            break

        line_bytes.append(stdout_bytes[0])

        line = line_bytes.decode('utf-8', errors='replace')
        if not line.endswith('\n'):
            continue
        
        line_bytes = bytearray()

        print(line, end="")
        train_match_result = trainstep_pattern.match(line)
       
        any_wandb_log = False
        if train_match_result:
            groups = train_match_result.groups()
            step, current_total_steps, loss, norm, lr, ms, mfu, tok_s = groups

            if total_steps == -1:
                total_steps = int(current_total_steps)

            # log to wandb
            if enable_wandb:
                wandb.log({
                    "Step": int(step),
                    "train/loss": float(loss),
                    "train/norm": float(norm),
                    "train/lr": float(lr),
                    "train/ms": float(ms),
                    "train/mfu": float(mfu),
                    "train/tok_s": int(tok_s)
                })
                any_wandb_log = True

        if val_loss_pattern.match(line) and step != -1:
            val_loss = val_loss_pattern.match(line).groups()[0]

            # log to wandb
            if enable_wandb:
                wandb.log({
                    "Step": int(step),
                    "val/loss": float(val_loss)
                })
                any_wandb_log = True

        if hella_swag_pattern.match(line) and step != -1:
            _, _, hswg_score = hella_swag_pattern.match(line).groups()
            # log to wandb
            if enable_wandb:
                wandb.log({
                    "Step": int(step),
                    "val/hella_swag": float(hswg_score)
                })
                any_wandb_log = True

        # sleep for a bit to save time for seperate wandb thread
        if any_wandb_log:
            time.sleep(0.15)


    for line in process.stderr:
        print(line, end="")


if __name__ == '__main__':
    main()
