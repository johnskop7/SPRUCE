# Classifier Training Scripts

A collection of scripts and configurations for training a CNN classifier on medical images, extracting penultimate-layer activations, and saving correctly classified test images.

---

## Repository Layout

```
classifier_training_scripts/
├── configs/
│   ├── train.yaml             # training hyperparameters & data/model paths
│   ├── activations.yaml       # settings for dumping penultimate-layer features
│   └── correct_images.yaml    # settings for saving correctly classified test images
├── scripts/
│   ├── train.py               # runs adversarial/plain training
│   ├── save_activations.py    # extracts & pickles penultimate-layer activations
│   └── save_images.py         # saves correctly classified images above threshold
├── data/
│   └── loaders.py             # `load_data` for train/val/test splits
├── models/
│   ├── convnext.py            # ConvNeXt wrapper
│   └── resnet50.py            # ResNet50 wrapper
├── training/
│   ├── trainer.py             # `Trainer` class with LR scheduling, TRADES, callbacks
│   ├── callbacks.py           # `EarlyStopping` & `ModelCheckpoint`
│   └── metrics.py             # classification metrics helpers
└── README.md                  # this file
```

---

## Prerequisites

* Python ≥ 3.8
* PyTorch & torchvision
* pyyaml, tqdm, numpy, Pillow
* (Optional) CUDA for GPU acceleration

Install dependencies:

```bash
pip install torch torchvision pyyaml tqdm numpy pillow
```

---

## Configuration Files

Each script uses its own YAML config under `configs/`.

### `configs/train.yaml`

```yaml
data:
  data_dir:     "/path/to/dataset"
  batch_size:   16
  image_size:   256
  dataset_type: cardiomegaly  # or lung_opacity, oct, dementia

model:
  arch:         convnext      # or resnet50
  num_classes:  2

train:
  lr:             5e-4
  epochs:         100
  optimizer:      adamw        # or sgd
  scheduler:      cosine       # or step, reduce_on_plateau
  warmup_epochs:  5
  patience:       10
  class_weights:  [1.0, 1.0]
  label_smoothing: 0.01

adv:
  method:         trades       # or plain
  epsilon:        0.063
  alpha:          0.0063
  pgd_steps:      10
  adv_eval_freq:  5
  norm:           l_inf

output_dir: "/path/to/exp1"
```

### `configs/activations.yaml`

```yaml
data:
  data_dir:     "/path/to/dataset"
  batch_size:   64
  image_size:   256
  dataset_type: cardiomegaly

model:
  arch:        convnext
  num_classes: 2
  weights:     "/path/to/exp1/best_model.pth"

extract:
  save_path:   "/path/to/outputs/activations.pkl"
```

### `configs/correct_images.yaml`

```yaml
data:
  data_dir:     "/path/to/dataset"
  batch_size:   64
  image_size:   256
  dataset_type: cardiomegaly

model:
  arch:        convnext
  num_classes: 2
  weights:     "/path/to/exp1/best_model.pth"

extract:
  output_dir:   "/path/to/outputs/correct_images"
  class_number: 1
  threshold:    0.5
  num_images:   100
```

---

## Script Usage

All scripts live in `scripts/` and take a `-c`/`--config` argument.

### 1. Train the Model

```bash
python scripts/train.py \
  --config configs/train.yaml
```

* Logs to console & `output_dir/training.log`
* Saves:

  * `best_model.pth`
  * `loss_curves.png` & `accuracy_curves.png`
  * `confusion_matrix.png`
  * `metrics.json`

### 2. Save Penultimate-Layer Activations

```bash
python scripts/save_activations.py \
  -c configs/activations.yaml
```

* Loads the **train** split
* Extracts penultimate-layer features
* Outputs a pickle:
  `{ class_id → ndarray[num_samples,feat_dim] }`

### 3. Save Correctly Classified Images

```bash
python scripts/save_images.py \
  -c configs/correct_images.yaml
```

* Loads the **test** split
* Filters images where `pred == label` for `class_number`
* Applies confidence ≥ `threshold`
* Saves up to `num_images` PNGs in `output_dir`

---

## Contributing

* Add new architectures under `models/`
* Extend data loaders in `data/`
* Improve training strategies & callbacks in `training/`

---

## License

This project is licensed under the MIT License.
