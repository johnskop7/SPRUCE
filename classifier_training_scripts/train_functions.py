import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from trades import trades_loss
import os
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

###############################################################################
# 1. Logger Setup
###############################################################################

def setup_logger(output_dir):
    """
    Sets up the logger to output logs to both console and a file.

    Args:
        output_dir (str): Directory where the log file will be saved.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info(f"Logger initialized. Logging to console and {log_file}")

###############################################################################
# 2. Training Function for each epoch (Supports TRADES)
###############################################################################
def train_one_epoch(model, train_loader, criterion, optimizer, device, use_trades=False, trades_args=None):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    scaler = GradScaler()

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            if use_trades and trades_args is not None:
                loss = trades_loss(
                    model=model,
                    x_natural=images,
                    y=labels,
                    optimizer=optimizer,
                    step_size=trades_args.alpha,
                    epsilon=trades_args.epsilon,
                    perturb_steps=trades_args.pgd_steps,
                    beta=trades_args.beta,
                    distance=trades_args.norm
                )
            else:
                outputs ,features = model(images)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({'loss': loss.item()})

    return running_loss / len(train_loader.dataset)


###############################################################################
# 3. Validation on Clean Images
###############################################################################
def validate_clean(model, validation_loader, device):
    model.eval()
    running_loss, correct_predictions, total_predictions = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(validation_loader, desc='Val (Clean)', leave=False):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs,features = model(images)
                loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item() * images.size(0)

    return running_loss / len(validation_loader.dataset), (correct_predictions / total_predictions) * 100


###############################################################################
# 4. Validation on Adversarial Images
###############################################################################
def validate_adversarial(model, validation_loader, device, trades_args):
    model.eval()
    running_loss, correct_predictions, total_predictions = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()

    step_size, epsilon, pgd_steps, norm_type = (
        trades_args.alpha, trades_args.epsilon, trades_args.pgd_steps, trades_args.norm
    )

    for images, labels in tqdm(validation_loader, desc='Val (Adversarial)', leave=False):
        images, labels = images.to(device), labels.to(device)

        # Initialize adversarial examples
        x_adv = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # PGD Attack
        for _ in range(pgd_steps):
            x_adv.requires_grad_()
            logits_adv,features = model(x_adv)
            loss_ce = criterion(logits_adv, labels)
            grad_x_adv = torch.autograd.grad(loss_ce, [x_adv])[0]

            if norm_type == 'l_inf':
                x_adv = x_adv.detach() + step_size * torch.sign(grad_x_adv.detach())
                x_adv = torch.min(torch.max(x_adv, images - epsilon), images + epsilon)
            elif norm_type == 'l2':
                grad_norm = torch.norm(grad_x_adv.view(grad_x_adv.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                normalized_grad = grad_x_adv / (grad_norm + 1e-8)
                x_adv = x_adv.detach() + step_size * normalized_grad
                perturbation = x_adv - images
                perturb_norm = torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                factor = torch.min(torch.ones_like(perturb_norm), epsilon / (perturb_norm + 1e-8))
                x_adv = images + perturbation * factor

            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # Evaluate adversarial images
        with torch.no_grad():
            outputs_adv,features = model(x_adv)
            loss_adv = criterion(outputs_adv, labels)

        _, predicted = torch.max(outputs_adv, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        running_loss += loss_adv.item() * images.size(0)

    return running_loss / len(validation_loader.dataset), (correct_predictions / total_predictions) * 100


###############################################################################
# 5. Global training function
###############################################################################

def train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        method='plain',  # 'plain' or 'trades'
        num_epochs=10,
        patience=3,
        output_dir=None,
        trades_args=None,
        adv_eval_freq=5
):
    best_val_acc = 0.0
    best_model_wts = None
    epochs_no_improve = 0
    early_stop = False

    train_losses = []
    val_losses_clean = []
    val_accuracies_clean = []

    # We'll store adv metrics but only fill them every adv_eval_freq epochs
    val_losses_adv = []
    val_accuracies_adv = []

    use_trades = (method == 'trades')

    for epoch in range(num_epochs):
        if early_stop:
            break

        logging.info(f'Epoch {epoch + 1}/{num_epochs}')
        logging.info('-' * 10)

        # 1. Training
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_trades=use_trades,
            trades_args=trades_args
        )

        # 2. Clean Validation
        val_loss_clean, val_acc_clean = validate_clean(model, val_loader, device)

        # 3. Adversarial Validation only every adv_eval_freq epochs
        if use_trades and ((epoch + 1) % adv_eval_freq == 0):
            val_loss_adv, val_acc_adv = validate_adversarial(model, val_loader, device, trades_args)
            logging.info(f'Val (Adversarial) Loss: {val_loss_adv:.4f}')
            logging.info(f'Val (Adversarial) Accuracy: {val_acc_adv:.2f}%')
        else:
            val_loss_adv, val_acc_adv = None, None

        # Logging
        logging.info(f'Training Loss: {train_loss:.4f}')
        logging.info(f'Val (Clean) Loss: {val_loss_clean:.4f}')
        logging.info(f'Val (Clean) Accuracy: {val_acc_clean:.2f}%')

        # 4. Append to lists
        train_losses.append(train_loss)
        val_losses_clean.append(val_loss_clean)
        val_accuracies_clean.append(val_acc_clean)
        val_losses_adv.append(val_loss_adv)
        val_accuracies_adv.append(val_acc_adv)

        # 5. Early Stopping logic
        if val_acc_clean > best_val_acc:
            best_val_acc = val_acc_clean
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
            logging.info(f'Validation (Clean) accuracy improved to {val_acc_clean:.2f}%')
        else:
            epochs_no_improve += 1
            logging.info(f'Validation (Clean) accuracy has not improved from {best_val_acc:.2f}%. '
                         f'Patience: {epochs_no_improve}/{patience}')

        # Step the scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(best_val_acc)
            else:
                scheduler.step()

        if epochs_no_improve >= patience:
            logging.info('Early stopping triggered.')
            early_stop = True

        # Optionally save model weights every 5 epochs
        if (epoch + 1) % 5 == 0:
            if best_model_wts is not None:
                if output_dir:
                    save_path = os.path.join(output_dir, f'best_model_weights.pth')
                else:
                    save_path = f'best_model_weights.pth'
                torch.save(best_model_wts, save_path)
                logging.info(f'Saved best model weights at epoch {epoch + 1} with accuracy {best_val_acc:.2f}%')

    # 6. Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        logging.info(f'Loaded best model weights with validation accuracy: {best_val_acc:.2f}%')

    # 7. Plot training & clean validation curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses_clean, label='Validation (Clean) Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Clean Validation Loss')
    plt.legend()

    if output_dir:
        loss_plot_path = os.path.join(output_dir, 'loss_curves.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f'Saved training-validation loss plot at {loss_plot_path}')
    else:
        plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies_clean, label='Validation (Clean) Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Clean Validation Accuracy')
    plt.legend()

    if output_dir:
        accuracy_plot_path = os.path.join(output_dir, 'accuracy_curves.png')
        plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f'Saved validation accuracy plot at {accuracy_plot_path}')
    else:
        plt.show()

    return model, train_losses, val_losses_clean, val_accuracies_clean



###############################################################################
# 6. Evaluation Function
###############################################################################

def evaluate_model(test_loader, model, device, dataset=None, output_dir=None):
    model.eval()
    predictions = []
    actuals = []
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if dataset not in ["cardiomegaly","lung_opacity","oct","dementia"]:
        logging.error("Invalid dataset provided. Supported datasets:'cardiomegaly', 'lung_opacity', 'oct','dementia")
        raise ValueError("Invalid dataset provided.")

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,features = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.view(-1).tolist())
            actuals.extend(labels.view(-1).tolist())

    # Metrics
    if dataset in ["cardiomegaly", "lung_opacity"]:
        average_type = 'binary' if len(set(actuals)) == 2 else 'weighted'
    elif dataset == "oct" or dataset == "dementia" :
        average_type = 'weighted'

    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average=average_type, zero_division=0)
    recall = recall_score(actuals, predictions, average=average_type, zero_division=0)
    f1 = f1_score(actuals, predictions, average=average_type, zero_division=0)

    logging.info(f"Evaluation Metrics - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, "
                 f"Recall: {recall:.2f}, F1: {f1:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(10, 7))
    label_mapping = {
        "lung_opacity": ['Lung Opacity','Normal'],
        "cardiomegaly": ['Cardiomegaly', 'Normal'],
        "oct": ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
        "dementia": [ 'Mild Impairment', 'Moderate Impairment','No Impairment', 'Very Mild Impairment'],
    }

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_mapping.get(dataset, []),
                yticklabels=label_mapping.get(dataset, []))

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    if output_dir:
        output_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved confusion matrix at {output_path}")
    else:
        plt.show()

    return accuracy, precision, recall, f1
