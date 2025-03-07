import time
from itertools import chain

import torch
from config.config_utils import (
    handle_reproducibility,
    load_config,
    parse_args,
    save_config,
)
from datasets import KCoordDataset, seed_worker
from model import *
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from train_utils_meta import *
import re

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


def main():
    args = parse_args()
    config = load_config(args.config)
    config["device"] = args.device

    rs_numpy, rs_torch = handle_reproducibility(config["seed"])

    torch.set_default_dtype(torch.float32)

    dataset = KCoordDataset(**config["dataset"])
    loader_config = config["dataloader"]
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, collate_fn=collate_fn, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker, generator=RS_TORCH)
    dataloader = DataLoader(
        dataset,
        batch_size=loader_config["batch_size"],
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        shuffle=True,
        pin_memory=loader_config["pin_memory"],
    )

    model_params = config["model"]["params"]
    model = MODEL_CLASSES[config["model"]["id"]](**model_params)
    
    ### LOAD HASH TABLES
    # Load checkpoint.
    model_state_dict = torch.load(config["model_checkpoint"], map_location=torch.device('cpu'))["model_state_dict"]
    # for layer_name,tensor in model.named_parameters():
    #     if layer_name in model_state_dict:
    #         tensor.data.copy_(model_state_dict[layer_name])

    levels = [[] for _ in range(config['model']['params']['levels'])]  # Creates n# empty lists according to the number of levels stated in config

    ## GET THE TABLES FOR EACH VOLUME AND EACH LEVEL
    for layer_name, tensor in model_state_dict.items():
        if 'embed_fn' in layer_name:
            numbers = re.findall(r'\d+', layer_name)
            # Convert the last number to an integer - last number corresponds to the level ID
            last_number = int(numbers[-1])
            
            # Assign tensor to the corresponding level if the number is valid
            if 0 <= last_number < len(levels):
                levels[last_number].append(tensor)

    # AVERAGE EACH LEVEL ACROSS ALL VOLUMES
    table_dict = []
    for level in levels:
        mean_of_level = np.mean(level, axis = 0)
        table_dict.append(torch.tensor(mean_of_level))

    # INITIALIZE THE HASH TABLES WITH EACH MEAN OF THE LEVEL
    for layer_name, tensor in model.named_parameters():
        if 'embed_fn' in layer_name:
            numbers = re.findall(r'\d+', layer_name)
            level = int(numbers[-1]) # Find out the level id
            tensor.data.copy_(table_dict[level]) # Locate the level ID in the table dictionary
        else:
            tensor.data.copy_(model_state_dict[layer_name])
            print(layer_name)
    print('Hash tables initialized from checkpoint, model loaded')

    ##### Optimizer
    optimizer = torch.load(config["model_checkpoint"])["optimizer_state_dict"]

    print("Checkpoint loaded successfully.")
    print('Optimizing hash embeddings...')
    # Freeze the parameters in sine layers
    for layer_name, tensor in model.named_parameters():
        if 'embed_fn' in layer_name:
            tensor.requires_grad = True
        else:
            tensor.requires_grad = False
        
    # Only Hash encoders are optimized.
    optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
        model.parameters(), **config["optimizer"]["params"]
    )
        

    loss_fn = LOSS_CLASSES[config["loss"]["id"]](**config["loss"]["params"])
    scheduler = SCHEDULER_CLASSES[config["scheduler"]["id"]](
        optimizer, **config["scheduler"]["params"]
    )

    print(f"model {model}")
    print(f"loss {loss_fn}")
    print(f"optimizer {optimizer}")
    print(f"scheduler {scheduler}")
    print(config)
    print(f"Number of steps per epoch: {len(dataloader)}")

    print(f"Starting {config['runtype']} process...")
    t0 = time.time()

    trainer = Trainer(
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    trainer.train()

    save_config(config)

    t1 = time.time()
    print(f"Time it took to run: {(t1-t0)/60} min")


if __name__ == "__main__":
    main()
