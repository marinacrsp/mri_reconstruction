import os
import time
from itertools import chain

import torch
import torch.multiprocessing as mp
from config.config_utils import handle_reproducibility, load_config, parse_args
from datasets import KCoordDataset, seed_worker
from model import *
from torch.distributed import destroy_process_group, init_process_group
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from train_utils_meta import *


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    torch.cuda.set_device(rank)

    # `init_process_group()` is blocking. That means the code wait until all processes have reached that line and the command is succesfully executed before going on.
    # nccl = NVIDIA Collective Communications Library.
    init_process_group("nccl", rank=rank, world_size=world_size)


MODEL_CLASSES = {
    "Siren": Siren,
}

LOSS_CLASSES = {
    "MAE": MAELoss,
    "DMAE": DMAELoss,
    "MSE": MSELoss,
    "MSEDist": MSEDistLoss,
    "HDR": HDRLoss,
    "LogL2": LogL2Loss,
    "MSEL2": MSEL2Loss,
}

OPTIMIZER_CLASSES = {
    "Adam": Adam,
    "AdamW": AdamW,
    "SGD": SGD,
}

SCHEDULER_CLASSES = {"StepLR": StepLR}


def main(rank: int, world_size: int, config: dict):
    ddp_setup(rank, world_size=world_size)

    dataset = KCoordDataset(**config["dataset"])
    loader_config = config["dataloader"]
    

    # N.B. Since we are using a sampler, we need to set shuffle to False.
    
    num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])//world_size - 1
    print(num_workers)
    print(world_size)
    print(int(os.environ['SLURM_CPUS_PER_TASK']))
    if num_workers < 0:
        num_workers = 0
    #################### DATALOADING TO MULTIGPU ######################
    dataloader = DataLoader(
        dataset,
        batch_size=loader_config["effective_batch_size"]//world_size,
        num_workers=num_workers, # This is needed to make processing faster 
        shuffle=False,
        sampler=DistributedSampler(dataset),
        pin_memory=True,
    )
    #####################################################################
    #####################################################################
    
    model_params = config["model"]["params"]
    model_params["n_volumes"] = config["dataset"]["n_volumes"]
    
    model = MODEL_CLASSES[config["model"]["id"]](**model_params)

    if config["runtype"] == "test":
        assert (
            "model_checkpoint" in config.keys()
        ), f"[GPU{rank}] Error: Trying to start a test run without a model checkpoint."

        # Load checkpoint.
        model_state_dict = torch.load(config["model_checkpoint"])["model_state_dict"]
        model.load_state_dict(model_state_dict)
        print(f"GPU{rank}] Checkpoint loaded successfully.")

        # Only embeddings are optimized.
        for param in model.parameters():
            param.requires_grad = False


    elif config["runtype"] == "train":
        if "model_checkpoint" in config.keys():
            model_state_dict = torch.load(config["model_checkpoint"])[
                "model_state_dict"
            ]
            model.load_state_dict(model_state_dict)
            
            print(f"GPU{rank}] Checkpoint loaded successfully.")

        optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
            model.parameters(),
            **config["optimizer"]["params"],
        )

    else:
        raise ValueError(f"[GPU{rank}] Incorrect runtype (must be `train` or `test`).")

    loss_fn = LOSS_CLASSES[config["loss"]["id"]](**config["loss"]["params"])
    scheduler = SCHEDULER_CLASSES[config["scheduler"]["id"]](
        optimizer, **config["scheduler"]["params"]
    )

## Only print the characteristics of the main running GPU 
    if rank == 0:
        
        print(f"model {model}")
        print(f"loss {loss_fn}")
        print(f"optimizer {optimizer}")
        print(f"scheduler {scheduler}")
        print(config)
        print(f'Total n# sampled points for training : {len(dataset)}')
        print(f'Total n# of steps per epoch : {len(dataloader)}')

    trainer = Trainer(
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=rank,
        config=config,
    )
    trainer.train()

    destroy_process_group()


#### MAIN SCRIPT BEING RUN:
###################################################################
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    rs_numpy, rs_torch = handle_reproducibility(config["seed"])
    torch.set_default_dtype(torch.float32)

    world_size = torch.cuda.device_count()
    assert world_size == int(os.environ["SLURM_GPUS_ON_NODE"])
    
    print(f'Number of GPUs used for training: {world_size}')
    print(f"Starting {config['runtype']} process...")
    
    t0 = time.time()

    mp.spawn(main, args=(world_size, config), nprocs = world_size) #NOTE: This is run per GPU

    t1 = time.time()
    print(f"Time it took to run: {(t1-t0)/60} min")