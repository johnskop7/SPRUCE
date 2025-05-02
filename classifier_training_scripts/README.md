# Classifier Training Scripts

This folder contains three main scripts: train.py , save_activations.py and save_images.py . Each script has its own configuration file for passing the necessary arguments. We are going to explain the functioning of each script below.

---

## Layout

```
classifier_training_scripts/
├── configs/
│   ├── train.yaml             # training hyperparameters & data/model paths
│   ├── activations.yaml       # settings for saving penultimate-layer featureactivations
│   └── correct_images.yaml    # settings for saving correctly classified test images
├── scripts/
│   ├── train.py               # runs adversarial/plain training
│   ├── save_activations.py    # extracts & saves penultimate-layer activations
│   └── save_images.py         # saves correctly classified images above threshold
├── data/
│   ├── loaders.py             # builds and returns data_loaders for each train/test/val split
│   ├── transforms.py          # transforms applied on images during training 
├── losses/
│   ├── trades.py              # implementation of TRADES loss based on https://github.com/yaodongyu/TRADES
├── models/
│   ├── convnext.py            # ConvNeXt wrapper
│   └── resnet50.py            # ResNet50 wrapper
├── training/
│   ├── trainer.py             # encapsulates the full train–validate–save loop in one class called Trainer
│   ├── callbacks.py           # `EarlyStopping` & `ModelCheckpoint`
│   └── metrics.py             # classification metrics helpers
├── utils/
│   ├── config.py              # Simple YAML loader that reads a `.yaml` file into a Python `dict`
│   └── logging.py             # Centralized logger setup
└── README.md                  # this file
```

---

## Prerequisites

* Python ≥ 3.8
* PyTorch & torchvision
* pyyaml, tqdm, numpy, Pillow
  * CUDA for GPU acceleration

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

## Dataset Folder Structure

Your data directory should follow the standard `ImageFolder` layout:

```

/path/to/dataset/
├── train/
│   ├── classA/
│   │   ├── img001.png
│   │   └── …
│   └── classB/
│       ├── img002.png
│       └── …
├── val/
│   ├── classA/
│   │   ├── img101.png
│   │   └── …
│   └── classB/
│       ├── img102.png
│       └── …
└── test/
├── classA/
│   ├── img201.png
│   └── …
└── classB/
├── img202.png
└── …

```
Each of the `train/`, `val/`, and `test/` folders must contain one sub-folder per class, with all images for that class inside. This structure ensures compatibility with the provided data loaders.

---

## Script Usage

All scripts live in `scripts/` and take a `-c`/`--config` argument.


### 1. Train the Model

```bash
python scripts/train.py \
  --config configs/train.yaml
```
* Trains the chosen model architecture on the specified dataset either adversarially or not.
* Logs to console & `output_dir/training.log`
* Saves:

  * `best_model.pth`
  * `loss_curves.png` & `accuracy_curve.png`
  * `confusion_matrix.png`
  * `metrics.json`

### 2. Save Penultimate-Layer Activations

```bash
python scripts/save_activations.py \
  -c configs/activations.yaml
```

* Loads the **train** split
* Extracts penultimate-layer feature activations
* Outputs a pickle dictionary:
  `{ class_id → ndarray[num_samples,feat_dim] }`
* This file is necessary for the creation of counterfactual feature vectors based on the PIECE methodology

### 3. Save Correctly Classified Images

```bash
python scripts/save_images.py \
  -c configs/correct_images.yaml
```

* Loads the **test** split
* Filters images where `pred == label` for `class_number`
* Applies confidence ≥ `threshold`
* Saves up to `num_images` images in `output_dir`

---

## Contributing

* Add new architectures under `models/`
* Extend data loaders in `data/`
* Improve training strategies & callbacks in `training/`

---

## License

This project is licensed under the MIT License.
