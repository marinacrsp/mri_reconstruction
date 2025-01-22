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
from train_utils import *

MODEL_CLASSES = {
    "Siren": Siren,
    "Siren_v2": Siren_v2,
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
    embeddings = torch.nn.Embedding(
        len(dataset.metadata), model_params["embedding_dim"]
    )
    torch.nn.init.normal_(
        embeddings.weight.data, 0.0, config["loss"]["params"]["sigma"]
    )

    model = MODEL_CLASSES[config["model"]["id"]](**model_params)

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
        # Freeze the model parameters
        for param in model.parameters():
            param.requires_grad = False

        optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
            embeddings.parameters(), **config["optimizer"]["params"]
        )

    elif config["runtype"] == "train":
        if "model_checkpoint" in config.keys():
            model_state_dict = torch.load(config["model_checkpoint"])[
                "model_state_dict"
            ]
            model.load_state_dict(model_state_dict)
            print("Checkpoint loaded successfully.")

        optimizer = OPTIMIZER_CLASSES[config["optimizer"]["id"]](
            chain(embeddings.parameters(), model.parameters()),
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
        embeddings=embeddings,
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
