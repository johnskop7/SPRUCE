import torch
import torch.nn as nn
from load_dataloaders import load_data
from models.ConvNeXt import CustomConvNeXt
from models.ResNet50 import CustomResNet50
from train_functions import setup_logger, train_model, evaluate_model, validate_clean
import argparse
import os
import logging

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        setup_logger(args.output_dir)
        with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("No output directory provided. Logging only to console.")

    logging.info("Starting the training process...")

    # Load data with updated batch size and image size
    logging.info("Loading data...")
    train_loader, val_loader, test_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        normalized=True,
        dataset_type=args.dataset_type
    )
    logging.info("Data loaded successfully.")
    #Adressing class imbalance
    class_weights = args.class_weights
    class_weights_tensor = torch.tensor(class_weights).float().to(device)
    logging.info(f"Class weights tensor: {class_weights_tensor}")

    # Initialize the model
    logging.info("Initializing the model...")
    if args.model_arch == 'convnext':
        model = CustomConvNeXt(num_classes=args.num_classes)
    elif args.model_arch == 'resnet50':
        model = CustomResNet50(num_classes=args.num_classes)
    else:
        raise ValueError("Unsupported model architecture specified.")

    # Define criterion
    if args.method == 'plain':
        criterion = nn.CrossEntropyLoss( weight=class_weights_tensor,label_smoothing=args.label_smoothing)
    elif args.method == 'trades':
        # For TRADES, we'll handle the loss inside the training loop
        criterion = None
    else:
        raise ValueError("Unsupported training method specified.")

    # Initialize optimizer: switch from Adam to SGD with momentum
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported optimizer specified.")

    model.to(device)
    logging.info(f"Model initialized. Using device: {device}")

    # Full modified scheduler section:
    if args.scheduler == 'step':
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs,eta_min=1e-7  # Minimum learning rate
        )
    elif args.scheduler == 'reduce_on_plateau':
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5, verbose=True
        )
    else:
        main_scheduler = None

    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=args.warmup_epochs
    )

    # Combine schedulers
    if main_scheduler is not None:
        from torch.optim.lr_scheduler import SequentialLR
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[args.warmup_epochs]
        )
    else:
        scheduler = warmup_scheduler

    # Train the model
    logging.info("Starting model training...")
    model, train_losses, val_losses_clean, val_accuracies_clean = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        method=args.method,
        num_epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir,
        adv_eval_freq=args.adv_eval_freq,
        trades_args=args  # Pass all args for TRADES-specific parameters

    )
    logging.info("Model training completed.")

    # Save the trained model
    logging.info("Saving the trained model...")
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'trained_model.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

    # Evaluate the model
    logging.info("Starting model evaluation...")
    accuracy, precision, recall, f1 = evaluate_model(
        test_loader, model, device, dataset=args.dataset_type, output_dir=args.output_dir
    )
    logging.info("Model evaluation completed.")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN classifier on medical images")

    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, required=True,
                        help="Type of dataset (cardiomegaly, lung_opacity, oct, dementia)")
    parser.add_argument("--data_dir", type=str, required=True, help="Image directory")

    # Model parameters
    parser.add_argument("--model_arch", type=str, choices=['convnext', 'resnet50'], default='convnext',
                        help="Model architecture to use: 'convnext' or 'resnet50'")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes of the dataset")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate for the classifier training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for the classifier training")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay during training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    parser.add_argument('--class_weights', nargs='+', type=float, default=None,help="Manual class weights for unbalanced datasets. Must match --num_classes.")

    parser.add_argument('--label_smoothing', type=float, default=0.1,help = "Label smoothing parameter")
    # Early stopping
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs to wait before ending the training if the val loss does not improve")

    # Optimizer and scheduler
    parser.add_argument("--optimizer", type=str, choices=['sgd', 'adamw'], default='sgd',
                        help="Optimizer to use: 'sgd' or 'adam'")
    parser.add_argument("--scheduler", type=str, choices=['step', 'cosine', 'reduce_on_plateau'], default='step',
                        help="Learning rate scheduler to use")
    parser.add_argument("--step_size", type=int, default=30, help="Step size for StepLR scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of warmup epochs for learning rate ramp-up")
    # Output directory
    parser.add_argument("--output_dir", type=str, help="Output directory to save the training info")

    # Training method
    parser.add_argument("--method", type=str, choices=['plain', 'trades'], default='plain',
                        help="Select training method: 'plain' or 'trades'")

    # TRADES-specific hyperparameters
    parser.add_argument("--adv_eval_freq", type=int,default=5, help="Adversarial evaluation frequency")
    parser.add_argument("--beta", type=float, default=1.0, help="Trade-off parameter for TRADES")
    parser.add_argument("--epsilon", type=float, default=0.08, help="Maximum perturbation for TRADES")
    parser.add_argument("--alpha", type=float, default=0.02, help="Step size for PGD in TRADES")
    parser.add_argument("--pgd_steps", type=int, default=10, help="Number of PGD steps for TRADES")
    parser.add_argument("--norm", type=str, choices=['l2', 'l_inf'], default='l_inf',
                        help="Norm to use for adversarial training (l2 or l_inf)")

    args = parser.parse_args()
    if args.class_weights is not None and len(args.class_weights) != args.num_classes:
        parser.error(f"--class_weights expects {args.num_classes} values, but {len(args.class_weights)} were given.")

    main(args)
