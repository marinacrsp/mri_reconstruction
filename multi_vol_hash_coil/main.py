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
    
    ## Volume embeddings initialization
    ########################################################
    embeddings_vol = torch.nn.Embedding(
        len(dataset.metadata), model_params["vol_embedding_dim"]
    )

    
    ## Coil embeddings initialization
    ########################################################
    coil_sizes = []
    for i in range(len(dataset.metadata)):
        _, n_coils, _, _ = dataset.metadata[i]["shape"]
        coil_sizes.append(n_coils)
        
    total_n_coils = torch.cumsum(torch.tensor(coil_sizes), dim=0)[-1]
    
    # Create the indexes to access the embedding coil table
    start_idx = torch.tensor([0] + list(torch.cumsum(torch.tensor(coil_sizes), dim=0)[:-1]))

    # Create the table of embeddings for the coils
    embeddings_coil = torch.nn.Embedding(total_n_coils.item(), model_params["coil_embedding_dim"])

    model = MODEL_CLASSES[config["model"]["id"]](**model_params)
    
    
## NOTE : Train or inference
    if config["runtype"] == "test":
        assert (
            "model_checkpoint" in config.keys()
        ), "Error: Trying to start a test run without a model checkpoint."

        # Load checkpoint.
        model_state_dict = torch.load(config["model_checkpoint"])["model_state_dict"]
        model.load_state_dict(model_state_dict)
        
        #### Embedding
        phi_coil_zero = torch.load(config["model_checkpoint"])["embedding_coil_state_dict"]["weight"]
        phi_vol_zero = torch.load(config["model_checkpoint"])["embedding_vol_state_dict"]["weight"]
        print("Loading the dictionary of embeddings from pretrained checkpoint")
        embeddings_vol.weight.data.copy_(phi_vol_zero[:len(dataset.metadata)])
        embeddings_coil.weight.data.copy_(phi_coil_zero[:total_n_coils.item()])        

        # phi_coil_zero = torch.load(config["model_checkpoint"])["embedding_coil_state_dict"]["weight"].mean(0)
        # phi_vol_zero = torch.load(config["model_checkpoint"])["embedding_vol_state_dict"]["weight"].mean(0)
        # print("Initialization from mean of embeddings: ")
        # embeddings_coil.weight.data.copy_(phi_coil_zero.unsqueeze(0).repeat(total_n_coils.item(), 1))
        # embeddings_vol.weight.data.copy_(phi_vol_zero.unsqueeze(0).repeat(len(dataset.metadata), 1))

        # print("Reinitialization of embeddings: ")
        # phi_coil_zero = torch.normal(0.0, config["loss"]["params"]["sigma"], size=(model_params["coil_embedding_dim"],))
        # phi_vol_zero = torch.normal(0.0, config["loss"]["params"]["sigma"], size=(model_params["vol_embedding_dim"],))
        # embeddings_coil.weight.data.copy_(phi_coil_zero.unsqueeze(0).repeat(total_n_coils.item(), 1))
        # embeddings_vol.weight.data.copy_(phi_vol_zero.unsqueeze(0).repeat(len(dataset.metadata), 1))
    
        ##### Optimizer
        optimizer = torch.load(config["model_checkpoint"])["optimizer_state_dict"]
        print("Initialization of optimizer from mean of checkpoint dictionary: ")
        vol_optim = [optimizer["state"][0]["exp_avg"].mean(0).unsqueeze(0).expand(len(dataset.metadata), model_params["vol_embedding_dim"]), 
                    optimizer["state"][0]["exp_avg_sq"].mean(0).unsqueeze(0).expand(len(dataset.metadata), model_params["vol_embedding_dim"])]
        
        coil_optim = [optimizer["state"][1]["exp_avg"].mean(0).unsqueeze(0).expand(total_n_coils.item(), model_params["coil_embedding_dim"]), 
                    optimizer["state"][1]["exp_avg_sq"].mean(0).unsqueeze(0).expand(total_n_coils.item(), model_params["coil_embedding_dim"])]
        

        
        # print("Loading optimizer checkpoint")
        # vol_optim = [optimizer["state"][0]["exp_avg"][:len(dataset.metadata)], optimizer["state"][0]["exp_avg_sq"][:len(dataset.metadata)]]
        # coil_optim = [optimizer["state"][1]["exp_avg"][:total_n_coils.item()], optimizer["state"][1]["exp_avg_sq"][:total_n_coils.item()]]
        

    
        print("Checkpoint loaded successfully.")
        
        
        if config["hash_freeze"]:
            print('Optimizing volume and coil embeddings...')
            # Freeze the whole model
            for param in model.parameters():
                param.requires_grad = False

            optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
                        chain(embeddings_vol.parameters(), embeddings_coil.parameters()), **config["optimizer"]["params"]
                    )
        else:
            print('Optimizing volume, coil and hash embeddings...')
            # Freeze the parameters in sine layers
            for param in model.parameters():
                param.requires_grad = False
                
            for param in model.embed_fn.parameters():
                param.requires_grad = True
                
            # Only embeddings and Hash encoders are optimized.
            optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
                chain(embeddings_vol.parameters(), embeddings_coil.parameters(), model.embed_fn.parameters(),), **config["optimizer"]["params"]
            )
            

    
    elif config["runtype"] == "train":
        phi_coil_zero = torch.normal(0.0, config["loss"]["params"]["sigma"], size=(model_params["coil_embedding_dim"],))
        embeddings_coil.weight.data.copy_(phi_coil_zero.unsqueeze(0).repeat(total_n_coils.item(), 1))
        
        phi_vol_zero = torch.normal(0.0, config["loss"]["params"]["sigma"], size=(model_params["vol_embedding_dim"],))
        embeddings_vol.weight.data.copy_(phi_vol_zero.unsqueeze(0).repeat(len(dataset.metadata), 1))
        
        if "model_checkpoint" in config.keys():
            model_state_dict = torch.load(config["model_checkpoint"])[
                "model_state_dict"
            ]
            model.load_state_dict(model_state_dict)
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
        phi_vol=phi_vol_zero,
        vol_momentums = vol_optim,
        embeddings_coil = embeddings_coil,
        phi_coil=phi_coil_zero,
        coil_momentums = coil_optim,
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
