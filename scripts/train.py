#!/usr/bin/env python3
from __future__ import annotations

import argparse

from coces.config import CoCESConfig
from coces.data.dataset import CoCESDataset
from coces.models.coces import CoCESModel
from coces.training.trainer import CoCESTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoCES in three stages.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = CoCESConfig.from_yaml(args.config)
    model = CoCESModel(config)
    trainer = CoCESTrainer(model, config)
    trainer.fit(
        CoCESDataset(config.data.train_file),
        CoCESDataset(config.data.dev_file),
    )


if __name__ == "__main__":
    main()

