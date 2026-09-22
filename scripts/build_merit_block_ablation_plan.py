from __future__ import annotations

import argparse

from merit_feddg.block_ablation import build_block_ablation_plan
from merit_feddg.io import load_yaml, save_json


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a one-factor-at-a-time MERIT-Block ablation plan. "
            "This script never reads benchmark labels or scores."
        )
    )
    parser.add_argument("--config", default="configs/merit_block.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_block_ablation_plan(load_yaml(args.config))
    save_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
