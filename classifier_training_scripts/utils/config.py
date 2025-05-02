from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class DataConfig:
    data_dir: str
    batch_size: int
    image_size: int
    dataset_type: str
    num_workers: int = 2

@dataclass
class OptimConfig:
    lr: float
    weight_decay: float
    optimizer: str
    scheduler: str
    warmup_epochs: int
    step_size: Optional[int] = None
    gamma: Optional[float] = None

@dataclass
class AdvConfig:
    method: str            # 'plain' or 'trades'
    beta: float
    epsilon: float
    alpha: float
    pgd_steps: int
    norm: str
    adv_eval_freq: int

@dataclass
class TrainConfig:
    epochs: int
    patience: int
    class_weights: Optional[List[float]]
    label_smoothing: float

@dataclass
class Config:
    data: DataConfig
    optim: OptimConfig
    adv: AdvConfig
    train: TrainConfig
    model_arch: str
    num_classes: int
    output_dir: str
    device: str = 'cuda'  # or 'cpu'


def load_config(path: str) -> Config:
    """
    Loads a YAML file from `path` and returns a Config object with correct types.
    """
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    # Data config
    data_raw = raw.get('data', {})
    data_cfg = DataConfig(
        data_dir=str(data_raw['data_dir']),
        batch_size=int(data_raw['batch_size']),
        image_size=int(data_raw['image_size']),
        dataset_type=str(data_raw['dataset_type']),
        num_workers=int(data_raw.get('num_workers', 2))
    )

    # Optimizer config
    optim_raw = raw.get('optim', {})
    optim_cfg = OptimConfig(
        lr=float(optim_raw['lr']),
        weight_decay=float(optim_raw['weight_decay']),
        optimizer=str(optim_raw['optimizer']),
        scheduler=str(optim_raw['scheduler']),
        warmup_epochs=int(optim_raw['warmup_epochs']),
        step_size=int(optim_raw.get('step_size')) if 'step_size' in optim_raw else None,
        gamma=float(optim_raw.get('gamma')) if 'gamma' in optim_raw else None
    )

    # Adversarial config
    adv_raw = raw.get('adv', {})
    adv_cfg = AdvConfig(
        method=str(adv_raw['method']),
        beta=float(adv_raw['beta']),
        epsilon=float(adv_raw['epsilon']),
        alpha=float(adv_raw['alpha']),
        pgd_steps=int(adv_raw['pgd_steps']),
        norm=str(adv_raw['norm']),
        adv_eval_freq=int(adv_raw['adv_eval_freq'])
    )

    # Training config
    train_raw = raw.get('train', {})
    class_weights = train_raw.get('class_weights')
    if class_weights is not None:
        class_weights = [float(w) for w in class_weights]
    train_cfg = TrainConfig(
        epochs=int(train_raw['epochs']),
        patience=int(train_raw['patience']),
        class_weights=class_weights,
        label_smoothing=float(train_raw['label_smoothing'])
    )

    cfg = Config(
        data=data_cfg,
        optim=optim_cfg,
        adv=adv_cfg,
        train=train_cfg,
        model_arch=str(raw['model_arch']),
        num_classes=int(raw['num_classes']),
        output_dir=str(raw['output_dir']),
        device=str(raw.get('device', 'cuda'))
    )

    return cfg
