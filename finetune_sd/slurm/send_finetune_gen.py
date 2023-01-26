import os
from collections import Counter
from time import sleep
import submitit
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def track_jobs_with_pbar(jobs):
    num_completed = 0
    with tqdm(total=len(jobs)) as pbar:
        while any(job.state not in ["COMPLETED", "FAILED", "DONE"] for job in jobs):
            sleep(2)
            job_infos = [j.get_info() for j in jobs]
            state2count = Counter([info['State'] if 'State' in info else "None" for info in job_infos])
            newly_completed = state2count["COMPLETED"] - num_completed
            pbar.update(newly_completed)
            num_completed = state2count["COMPLETED"]
            s = [f"{k}: {v}" for k, v in state2count.items()]
            pbar.set_description(" | ".join(s))


@hydra.main(config_path="../../configs/finetune", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=5900, stdoutToServer=True, stderrToServer=True)

    output_dir = cfg.general.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.realpath(f"{output_dir}/config.yaml")
    OmegaConf.save(cfg, config_path)
    cmd_path = os.path.realpath(f"{output_dir}/cmd.sh")
    run_cmd = f"""#!/bin/bash
accelerate launch finetune_sd/training/finetune_gen.py
"""
    with open(cmd_path, 'w') as f:
        f.write(run_cmd)

    slurm_kwargs = {
        "slurm_job_name": cfg.slurm.job_name,
        "slurm_partition": cfg.slurm.partition,
        "slurm_nodes": 1,
        "slurm_additional_parameters": {
            "ntasks": cfg.slurm.n_processes,
            "gpus": cfg.slurm.n_processes,
            "tasks_per_node": cfg.slurm.n_processes,
        },
        "slurm_cpus_per_task": cfg.training.num_workers + 2,
        "slurm_time": cfg.slurm.time_limit,
        "slurm_exclude": "n-202,n-203,n-204,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01",
        "stderr_to_stdout": True
    }

    executor = submitit.AutoExecutor(folder=os.path.join(output_dir, "logs", "finetune_gen"))  # , cluster="debug")
    executor.update_parameters(**slurm_kwargs)

    function = submitit.helpers.CommandFunction(
        [
            "bash",
            f"{cmd_path}",
            # f"{config_path}",
        ]
    )
    job = executor.submit(function)


if __name__ == '__main__':
    main()
