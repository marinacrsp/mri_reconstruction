import time

import torch
from config.config_utils import (
    handle_reproducibility,
    load_config,
    parse_args,
    save_config,
)
from datasets import KCoordDataset, seed_worker
from model import *
from train_utils_tboard import *
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


MODEL_CLASSES = {
    "Siren0": Siren_skip,
    # "Siren3.0": Siren_skip_emb,
    "SirenHash": Siren_skip_hash
}

LOSS_CLASSES = {
    "MAE": MAELoss,
    "DMAE": DMAELoss,
    "MSE": MSELoss,
    "MSEDist": MSEDistLoss,
    "HDR": HDRLoss,
    "LogL2": LogL2Loss,
    "MSE_2": MSELoss_transformed
}

OPTIMIZER_CLASSES = {"Adam": Adam, "AdamW": AdamW, "SGD": SGD}

SCHEDULER_CLASSES = {"StepLR": StepLR}


def main():
    args = parse_args()
    config = load_config(args.config)
    config["device"] = args.device
    rs_numpy, rs_torch = handle_reproducibility(config["seed"])

    torch.set_default_dtype(torch.float32)
    
    ##################################################
    # Initialization
    ##################################################
    dataset_undersampled = KCoordDataset(**config["dataset"])
    dataset_fullysampled = KCoordDataset(**config["dataset_full"])
    loader_config = config["dataloader"]
    
    dataloader_undersampled = DataLoader(
        dataset_undersampled,
        batch_size=loader_config["batch_size"],
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        shuffle=True,
        pin_memory=loader_config["pin_memory"],
    )
    
    dataloader_fullysampled = DataLoader(
        dataset_fullysampled,
        batch_size=loader_config["batch_size"],
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        shuffle=False,
        pin_memory=loader_config["pin_memory"],
    )

    model_params = config["model"]["params"]
    model = MODEL_CLASSES[config["model"]["id"]](**model_params)
    
    if "model_checkpoint" in config.keys():
        model_state_dict = torch.load(config["model_checkpoint"])["model_state_dict"]
        model.load_state_dict(model_state_dict)
        print("Checkpoint loaded successfully.")

    loss_fn = LOSS_CLASSES[config["loss"]["id"]](**config["loss"]["params"])
    optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
        model.parameters(), **config["optimizer"]["params"]
    )
    scheduler = SCHEDULER_CLASSES[config["scheduler"]["id"]](
        optimizer, **config["scheduler"]["params"]
    )
    
    print(f"model {model}")
    print(f"loss {loss_fn}")
    print(f"optimizer {optimizer}")
    print(f"scheduler {scheduler}")
    print(config)

    ##################################################
    # Training Process
    ##################################################
    print("Starting training process...")
    t0 = time.time()

    trainer = Trainer(
        dataloader_consistency=dataloader_undersampled,
        dataloader_pisco = dataloader_fullysampled,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    trainer.train()

    save_config(config)

    t1 = time.time()
    print(f"Time it took to train: {(t1-t0)/60} min")


if __name__ == "__main__":
    main()