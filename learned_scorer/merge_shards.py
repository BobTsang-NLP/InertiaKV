#!/usr/bin/env python3
"""Merge scorer checkpoints from parallel shards (federated averaging)."""

import argparse
import json
import copy
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    state_dicts = []
    config = None

    for i in range(args.num_shards):
        ckpt_path = shard_dir / f"shard_{i}" / "scorer_final.pt"
        if not ckpt_path.exists():
            print(f"WARNING: {ckpt_path} not found, skipping")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dicts.append(ckpt["model_state_dict"])
        if config is None:
            config = ckpt["config"]
        print(f"  Loaded shard {i}: {ckpt_path}")

    if not state_dicts:
        raise RuntimeError("No shards found!")

    print(f"Averaging {len(state_dicts)} shards...")
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg_state[key] = stacked.mean(dim=0)

    torch.save({
        "model_state_dict": avg_state,
        "config": config,
        "num_shards": len(state_dicts),
    }, args.output)

    # Also save config.json
    config_path = Path(args.output).parent / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved merged model: {args.output}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
