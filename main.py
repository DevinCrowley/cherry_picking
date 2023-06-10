import argparse
from pathlib import Path
import sys

import ruamel.yaml as yaml
import numpy as np
import torch

from src.util import args_type, recursive_update
from src.cherry_picking import Cherry_Picking_Algo


def main(config):
    Cherry_Picking_Algo.run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", required=False)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (Path(sys.argv[0]).parent / "src" / "configs.yaml").read_text()
    )

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args(remaining)
    main(config)