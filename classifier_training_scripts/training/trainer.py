import os
import logging
import warnings
import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from losses.trades import trades_loss
from training.metrics import compute_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from training.callbacks import EarlyStopping, ModelCheckpoint

# Suppress UserWarnings (e.g. lr_scheduler deprecation)
warnings.simplefilter("ignore", category=UserWarning)


def make_optimizer(model: torch.nn.Module, optim_cfg) -> torch.optim.Optimizer:
    if optim_cfg.optimizer.lower() == 'sgd':
        return SGD(model.parameters(), lr=optim_cfg.lr, momentum=0.9, weight_decay=optim_cfg.weight_decay)
    else:
        return AdamW(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)


def make_scheduler(optimizer, optim_cfg, total_epochs: int):
    # Warmup scheduler
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=optim_cfg.warmup_epochs
    )
    # Main scheduler
    sched_type = optim_cfg.scheduler.lower()
    if sched_type == 'step':
        main = StepLR(optimizer, step_size=optim_cfg.step_size, gamma=optim_cfg.gamma)
    elif sched_type == 'cosine':
        effective_epochs = max(total_epochs - optim_cfg.warmup_epochs, 1)
        main = CosineAnnealingLR(optimizer, T_max=effective_epochs, eta_min=1e-7)
    else:
        main = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    return SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[optim_cfg.warmup_epochs]
    )


class Trainer:
    def __init__(self, model: torch.nn.Module, loaders: tuple, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader, self.val_loader, self.test_loader = loaders

        self.optimizer = make_optimizer(self.model, cfg.optim)
        self.scheduler = make_scheduler(self.optimizer, cfg.optim, cfg.train.epochs)
        self.scaler = GradScaler()

        # Callbacks
        self.early_stopper = EarlyStopping(patience=cfg.train.patience)
        self.checkpointer = ModelCheckpoint(dirpath=cfg.output_dir, filename='best_model.pth', mode='max')

        self.adv_eval_freq = cfg.adv.adv_eval_freq

    def fit(self):
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_adv_loss': [],
            'val_adv_acc': []
        }

        for epoch in range(1, self.cfg.train.epochs + 1):
            logging.info(f"Epoch {epoch}/{self.cfg.train.epochs}")

            # 1) Training
            train_loss = self._train_one_epoch()
            history['train_loss'].append(train_loss)

            # 2) Clean validation
            val_loss, val_acc = self._validate()
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # 3) Adversarial validation
            adv_loss, adv_acc = (None, None)
            if self.cfg.adv.method.lower() == 'trades' and epoch % self.adv_eval_freq == 0:
                adv_loss, adv_acc = self._validate_adversarial()
                logging.info(f"Val (Adversarial) Loss: {adv_loss:.4f}")
                logging.info(f"Val (Adversarial) Accuracy: {adv_acc:.2f}%")
            history['val_adv_loss'].append(adv_loss)
            history['val_adv_acc'].append(adv_acc)

            # 4) Scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            # 5) Checkpointing
            prev_best = self.checkpointer.best_score
            self.checkpointer(self.model, epoch, val_acc)
            if self.checkpointer.best_score is not None and self.checkpointer.best_score > (prev_best or -float('inf')):
                logging.info(f"Validation accuracy improved to {val_acc:.2f}% — saving checkpoint.")

            # 6) Early stopping
            should_stop = self.early_stopper(val_acc)
            if self.early_stopper.num_bad_epochs == 0:
                logging.info(f"Validation accuracy improved; resetting patience.")
            else:
                logging.info(
                    f"Validation accuracy did not improve from {self.early_stopper.best_score:.2f}%. "
                    f"Patience: {self.early_stopper.num_bad_epochs}/{self.early_stopper.patience}"
                )
            if should_stop:
                logging.info("Early stopping triggered. Halting training.")
                break

        # Restore best model
        if self.checkpointer.last_path:
            state = torch.load(self.checkpointer.last_path,weights_only=False)
            self.model.load_state_dict(state)
            logging.info(
                f"Loaded best model from {self.checkpointer.last_path} "
                f"(val_acc={self.checkpointer.best_score:.2f}%)"
            )
        else:
            logging.warning("No checkpoint saved; model weights unchanged.")

        # Save plots
        self._save_plots(history)
        return history

    def _train_one_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for images, labels in tqdm(self.train_loader, desc='Training', leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type):
                if self.cfg.adv.method.lower() == 'trades':
                    loss = trades_loss(
                        self.model,
                        x_natural=images,
                        y=labels,
                        step_size=self.cfg.adv.alpha,
                        epsilon=self.cfg.adv.epsilon,
                        perturb_steps=self.cfg.adv.pgd_steps,
                        beta=self.cfg.adv.beta,
                        distance=self.cfg.adv.norm
                    )
                else:
                    logits = self.model(images)
                    loss = F.cross_entropy(
                        logits,
                        labels,
                        weight=torch.tensor(self.cfg.train.class_weights, device=self.device)
                            if self.cfg.train.class_weights else None,
                        label_smoothing=self.cfg.train.label_smoothing
                    )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(self.train_loader.dataset)
        logging.info(f"Training Loss: {avg_loss:.4f}")
        return avg_loss

    def _validate(self) -> tuple:
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                with autocast(device_type=self.device.type):
                    logits = self.model(images)
                    loss = F.cross_entropy(logits, labels)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(self.val_loader.dataset)
        acc = compute_metrics(all_labels, all_preds)['accuracy']
        logging.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
        return avg_loss, acc

    def _validate_adversarial(self) -> tuple:
        """
        Perform adversarial validation using PGD. Generates adversarial examples with gradients,
        then evaluates model on them without gradients.
        """
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        eps = self.cfg.adv.epsilon
        alpha = self.cfg.adv.alpha
        steps = self.cfg.adv.pgd_steps
        norm = self.cfg.adv.norm

        for images, labels in tqdm(self.val_loader, desc='Adversarial Validating', leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            # Initialize adversarial examples
            x_adv = images + torch.empty_like(images).uniform_(-eps, eps)
            x_adv = x_adv.clamp(0, 1).detach()

            # PGD attack loop (requires grad)
            for _ in range(steps):
                x_adv.requires_grad_(True)
                # Forward
                logits_adv = self.model(x_adv)
                loss_inner = F.cross_entropy(logits_adv, labels)
                # Backward for delta
                grad = torch.autograd.grad(loss_inner, x_adv)[0]
                # Update
                if norm == 'l_inf':
                    x_adv = x_adv.detach() + alpha * grad.sign()
                    x_adv = torch.min(torch.max(x_adv, images - eps), images + eps)
                else:
                    # l2 norm
                    grad_flat = grad.view(grad.size(0), -1)
                    grad_norm = grad_flat.norm(p=2, dim=1).view(-1,1,1,1)
                    x_adv = x_adv.detach() + alpha * (grad / (grad_norm + 1e-8))
                    delta = x_adv - images
                    delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1,1,1,1)
                    factor = torch.min(torch.ones_like(delta_norm), eps / (delta_norm + 1e-8))
                    x_adv = images + delta * factor
                x_adv = x_adv.clamp(0, 1).detach()

            # Final evaluation on adversarial examples
            with torch.no_grad():
                logits_adv = self.model(x_adv)
                loss_final = F.cross_entropy(logits_adv, labels)
                preds = torch.argmax(logits_adv, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                running_loss += loss_final.item() * labels.size(0)

        adv_loss = running_loss / len(self.val_loader.dataset)
        adv_acc = correct / total * 100
        return adv_loss, adv_acc

    def _save_plots(self, history: dict) -> None:
        epochs = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history['train_loss'], label='Training Loss')
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        loss_path = os.path.join(self.cfg.output_dir, 'loss_curves.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved loss curves to {loss_path}")

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
        # if any(a is not None for a in history['val_adv_acc']):
        #     adv_epochs = [e for e in epochs if e % self.adv_eval_freq == 0]
        #     adv_vals = [a for a in history['val_adv_acc'] if a is not None]
        #     plt.plot(adv_epochs, adv_vals, label='Adversarial Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Curve')
        acc_path = os.path.join(self.cfg.output_dir, 'accuracy_curve.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved accuracy curve to {acc_path}")

    def evaluate(self) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        metrics = compute_metrics(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        label_map = {
            'cardiomegaly': ['Cardiomegaly', 'Normal'],
            'lung_opacity': ['Lung_Opacity', 'Normal'],
            'oct': ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
            'dementia': ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
        }
        labels = label_map.get(self.cfg.data.dataset_type, [])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_path = os.path.join(self.cfg.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved confusion matrix to {cm_path}")
        return metrics
