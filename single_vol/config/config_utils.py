import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from numpy.random import default_rng
from torch import cuda, manual_seed


def load_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    config["timestamp"] = datetime.now().strftime("%m-%d_%Hh%Mm")
    return config


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to a YAML file."""
    path = Path(config["path_to_outputs"]) / config["timestamp"]
    # Ensure path exists
    path.mkdir(parents=True, exist_ok=True)

    filename = path / "config.yaml"
    with open(filename, "w") as file:
        yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file.",
        default="./single_vol/config/config.yaml",
    )

    device = "cuda" if cuda.is_available() else "cpu"
    parser.add_argument("-d", "--device", type=str, default=device)

    return parser.parse_args()


def handle_reproducibility(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    rs_numpy = default_rng(seed)
    rs_torch = manual_seed(seed)

    # Warning: the settings commented below may slow down the execution time.

    # # PyTorch will only use deterministic operations.
    # # If no deterministic alternative exist, an error will be raised.
    # torch.use_deterministic_algorithms(True)

    # # Reproducibility when using GPUs

    # # Choice of algorithms (in `cuDNN`) is deterministic.
    # torch.backends.cudnn.benchmark = False

    # # Algorithms themselves (only the ones in `cuDNN`) are deterministic.
    # torch.backends.cudnn.deterministic = True

    # In some CUDA versions:
    # - Set the CUBLAS_WORKSPACE_CONFIG environment variable
    # - Be aware that RNN and LSTM networks may have non-deterministic behavior

    return rs_numpy, rs_torch
