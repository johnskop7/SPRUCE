#!/usr/bin/env python
import argparse
import logging
import os
import json
import sys
# Add project root to PYTHONPATH so modules like utils, data, models, etc. can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import load_config
from utils.logging import setup_logger
from data.loaders import load_dataloaders
from training.trainer import Trainer
from models.convnext import CustomConvNeXt
from models.resnet50 import CustomResNet50


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN classifier on medical images using a unified Trainer"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Initialize logging
    setup_logger(cfg.output_dir)
    logging.info("Loaded configuration from %s", args.config)
    logging.info("Output directory: %s", cfg.output_dir)

    # Load data
    logging.info("Building data loaders...")
    train_loader, val_loader, test_loader = load_dataloaders(
        data_dir=cfg.data.data_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        dataset_type=cfg.data.dataset_type,
        normalized=True,
        num_workers=cfg.data.__dict__.get('num_workers', 2)
    )
    logging.info("Data loaders ready.")

    # Build model
    logging.info("Initializing model architecture: %s", cfg.model_arch)
    if cfg.model_arch.lower() == 'convnext':
        model = CustomConvNeXt(num_classes=cfg.num_classes)
    elif cfg.model_arch.lower() == 'resnet50':
        model = CustomResNet50(num_classes=cfg.num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.model_arch}")

    # Create trainer and run training
    trainer = Trainer(
        model=model,
        loaders=(train_loader, val_loader, test_loader),
        cfg=cfg
    )

    logging.info("Starting training for %d epochs", cfg.train.epochs)
    history = trainer.fit()
    logging.info("Training complete.")

    # Evaluate on test set
    logging.info("Evaluating on test set...")
    metrics = trainer.evaluate()
    logging.info("Test metrics: %s", metrics)

    # Save metrics to file
    metrics_path = os.path.join(cfg.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info("Saved test metrics to %s", metrics_path)


if __name__ == '__main__':
    main()
