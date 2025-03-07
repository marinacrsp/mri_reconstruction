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
from model import Siren
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from train_utils import *
from models.d2c_vae.autoencoder_unet import Autoencoder

MODEL_CLASSES = {
    "Siren": Siren,
}

LOSS_CLASSES = {
    "MAE": MAELoss,
    "DMAE": MAELoss,
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
        num_workers= 2, #int(os.environ["SLURM_CPUS_PER_TASK"]),
        shuffle=True,
        pin_memory=loader_config["pin_memory"],
    )
    print('Dataset loaded')
    model_params = config["model"]["params"]
    latent_dim = config['ddconfig']['out_ch']
    
    model = MODEL_CLASSES[config["model"]["id"]](coord_dim=model_params['coord_dim'], 
                                                latent_dim=latent_dim, 
                                                hidden_dim=model_params['hidden_dim'], 
                                                n_layers=model_params['n_layers'],
                                                )
    vaemodel = Autoencoder(ddconfig=config["ddconfig"], 
                        embed_dim=latent_dim)
    
## NOTE : Train or inference
    if config["runtype"] == "test":
        assert (
            "model_checkpoint" in config.keys()
        ), "Error: Trying to start a test run without a model checkpoint."

        # Load checkpoint.
        model_state_dict = torch.load(config["model_checkpoint"])["model_state_dict"]
        model.load_state_dict(model_state_dict)

        print("Checkpoint loaded successfully.")
        
        # Only embeddings are optimized.
        optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
        chain(embeddings_vol.parameters(), embeddings_coil.parameters()), **config["optimizer"]["params"]
    )
                

    elif config["runtype"] == "train":
        
        if "model_checkpoint" in config.keys():
            model_state_dict = torch.load(config["model_checkpoint"])[
                "model_state_dict"
            ]
            model.load_state_dict(model_state_dict)

            print("Checkpoint loaded successfully.")

        optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
            chain(model.parameters(), vaemodel.parameters()),
            **config["optimizer"]["params"],
        )

    else:
        raise ValueError("Incorrect runtype (must be `train` or `test`).")

    loss_fn = LOSS_CLASSES[config["loss"]["id"]](**config['loss']['params'])
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
        model = model,
        vae = vaemodel,
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