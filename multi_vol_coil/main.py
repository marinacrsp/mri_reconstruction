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
    print(f'Connected to GPU: {torch.cuda.is_available()}')
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
    print('Dataset loaded')
    model_params = config["model"]["params"]
    
    #####################################################################
    # volume embedding
    #####################################################################
    embeddings_vol = torch.nn.Embedding(
        len(dataset.metadata), model_params["vol_embedding_dim"]
    )
    print(f'Shape of volume embeddings: {embeddings_vol.weight.shape}')

    ## Coil embeddings initialization
    ########################################################
    coil_sizes = []
    for i in range(len(dataset.metadata)):
        _, n_coils, _, _ = dataset.metadata[i]["shape"]
        coil_sizes.append(n_coils)
        
    total_n_coils = torch.cumsum(torch.tensor(coil_sizes), dim=0)[-1]
    
    # Create the indexes to access the embedding coil table
    start_idx = torch.tensor([0] + list(torch.cumsum(torch.tensor(coil_sizes), dim=0)[:-1]))
    embeddings_coil = torch.nn.Embedding(total_n_coils.item(), model_params["coil_embedding_dim"])
    print(f'Shape of coil embeddings: {embeddings_coil.weight.shape}')

    model = MODEL_CLASSES[config["model"]["id"]](**model_params)

## NOTE : Train or inference
    if config["runtype"] == "test":
        assert (
            "model_checkpoint" in config.keys()
        ), "Error: Trying to start a test run without a model checkpoint."

        # Load checkpoint.
        model_state_dict = torch.load(config["model_checkpoint"])["model_state_dict"]
        model.load_state_dict(model_state_dict)
        
        # print("Initialization from dictionary of embeddings: ")
        # print("Loading the dictionary of embeddings from pretrained checkpoint")
        # embeddings_vol.weight.data.copy_(phi_vol_zero[:len(dataset.metadata)])
        # embeddings_coil.weight.data.copy_(phi_coil_zero[:total_n_coils.item()])        

        # print("Initialization from mean of embeddings: ")
        # embeddings_coil.weight.data.copy_(
        #     torch.load(config["model_checkpoint"])["embedding_coil_state_dict"]["weight"].mean(0)
        #     )
        # embeddings_vol.weight.data.copy_(
        #     torch.load(config["model_checkpoint"])["embedding_vol_state_dict"]["weight"].mean(0)
        #     )

        # print("Initialization from dictionary of embeddings (from meta training): ")
        # embeddings_coil.weight.data.copy_(phi_coil_zero)
        # embeddings_vol.weight.data.copy_(phi_vol_zero)


        print("Reinitialization of embeddings: ")
        torch.nn.init.uniform_(embeddings_coil.weight.data, 0, 2*math.pi)
        torch.nn.init.uniform_(embeddings_vol.weight.data, 0, 2*math.pi)
    
    
        print("Checkpoint loaded successfully.")
        
        # Only embeddings are optimized.
        optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
        chain(embeddings_vol.parameters(), embeddings_coil.parameters()), **config["optimizer"]["params"]
    )
                

    elif config["runtype"] == "train":
        # phi_coil_zero = torch.normal(0.0, config["loss"]["params"]["sigma"], size=(model_params["coil_embedding_dim"],))
        # embeddings_coil.weight.data.copy_(phi_coil_zero.unsqueeze(0).repeat(total_n_coils.item(), 1))
        
        # phi_vol_zero = torch.normal(0.0, config["loss"]["params"]["sigma"], size=(model_params["vol_embedding_dim"],))
        # embeddings_vol.weight.data.copy_(phi_vol_zero.unsqueeze(0).repeat(len(dataset.metadata), 1))
        
        torch.nn.init.uniform_(embeddings_coil.weight.data, 0, 2*math.pi)
        torch.nn.init.uniform_(embeddings_vol.weight.data, 0, 2*math.pi)

        
        if "model_checkpoint" in config.keys():
            model_state_dict = torch.load(config["model_checkpoint"])[
                "model_state_dict"
            ]
            model.load_state_dict(model_state_dict)
            
            embeddings_vol.weight.data.copy_(torch.load(config["model_checkpoint"])["embedding_vol_state_dict"]["weight"])
            embeddings_coil.weight.data.copy_(torch.load(config["model_checkpoint"])["embedding_coil_state_dict"]["weight"])
            
            print("Checkpoint loaded successfully.")

        optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
            chain(embeddings_vol.parameters(), embeddings_coil.parameters(), model.parameters()),
            **config["optimizer"]["params"],
        )

    else:
        raise ValueError("Incorrect runtype (must be `train` or `test`).")

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
        embeddings_vol=embeddings_vol,
        embeddings_coil = embeddings_coil,
        embeddings_start_idx=start_idx,
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