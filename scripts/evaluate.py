#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from coces.evaluation import compute_metrics
from coces.utils.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CoCES predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    metrics = compute_metrics(read_jsonl(args.predictions)).to_dict()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        write_json(metrics, args.output)


if __name__ == "__main__":
    main()

