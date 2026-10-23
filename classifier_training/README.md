# Classifier Training

This directory implements the **classifier-training and classifier-side data-preparation stage of SPRUCE**.

It supports both naturally trained and adversarially robust classifiers and produces the three main artifacts required by the downstream SPRUCE pipeline:

1. a trained classifier checkpoint
2. stored penultimate-layer activations for PIECE
3. correctly classified, high-confidence query images for GAN inversion and counterfactual generation

The overall workflow is:

```text
Medical image dataset
        │
        ▼
Train classifier
        │
        ├──────────────► best_model.pth
        │
        ├──────────────► class_activations.pkl
        │
        └──────────────► selected query images
                              │
                              ▼
                             PTI
                              │
                              ▼
                 counterfactual_generation/
```

---

## Folder Structure

```text
classifier_training/
├── configs/
│   ├── train.yaml
│   ├── activations.yaml
│   └── correct_images.yaml
│
├── scripts/
│   ├── train.py
│   ├── save_activations.py
│   └── save_images.py
│
├── data/
│   ├── loaders.py
│   └── transforms.py
│
├── losses/
│   └── trades.py
│
├── models/
│   ├── convnext.py
│   ├── resnet50.py
│   └── efficientnet.py
│
├── training/
│   ├── trainer.py
│   ├── callbacks.py
│   └── metrics.py
│
├── utils/
│   ├── config.py
│   └── logging.py
│
└── README.md
```

### Main components

- **`configs/`**  
  YAML configuration files for training, activation extraction, and query-image selection.

- **`scripts/`**  
  Entry points for the three main classifier-side stages of SPRUCE.

- **`data/`**  
  Dataset loading, preprocessing, augmentation, and split handling.

- **`losses/`**  
  Contains the TRADES loss used for adversarially robust training.

- **`models/`**  
  Classifier wrappers that expose both logits and penultimate-layer features.

- **`training/`**  
  Training, validation, checkpointing, early stopping, metrics, and visualization utilities.

---

## Role in SPRUCE

The classifier is not only used to produce predictions.

Its internal feature representation is also central to SPRUCE.

After training:

- the classifier checkpoint is used during counterfactual optimization
- penultimate-layer activations are collected to model class-conditional feature distributions
- these distributions are used by PIECE to identify exceptional features
- correctly classified, high-confidence test images are selected as query images
- those query images are later inverted using PTI

This creates the connection:

```text
Classifier training
      │
      ├────────► classifier weights
      │
      ├────────► class activations
      │
      └────────► query images
                         │
                         ▼
                       PTI
                         │
                         ▼
                PIECE feature targeting
                         │
                         ▼
               Counterfactual generation
```

---

# Supported Classifiers

The current implementation supports:

- **ConvNeXt**
- **ResNet-50**
- **EfficientNet-B3**

The model is selected through:

```yaml
model_arch: convnext
```

or:

```yaml
model_arch: resnet50
```

or:

```yaml
model_arch: efficientnet
```

The classifier wrappers support returning both:

```python
logits, features = model(x, return_features=True)
```

where `features` corresponds to the penultimate-layer representation used by PIECE and the SPRUCE feature-alignment objective.

---

# Dataset Structure

The data directory should contain three splits:

```text
dataset/
├── train/
│   ├── class_A/
│   │   ├── image_001.png
│   │   └── ...
│   └── class_B/
│       ├── image_002.png
│       └── ...
│
├── val/
│   ├── class_A/
│   └── class_B/
│
└── test/
    ├── class_A/
    └── class_B/
```

The loaders use the standard torchvision `ImageFolder` structure.

---

## Dataset Types and Class Indices

The current data loader defines the following class mappings:

| Dataset type | Class index | Class |
|---|---:|---|
| `cardiomegaly` | 0 | Cardiomegaly |
| `cardiomegaly` | 1 | Normal |
| `lung_opacity` | 0 | Lung Opacity |
| `lung_opacity` | 1 | Normal |
| `oct` | 0 | CNV |
| `oct` | 1 | DME |
| `oct` | 2 | DRUSEN |
| `oct` | 3 | NORMAL |
| `dementia` | 0 | Mild Impairment |
| `dementia` | 1 | Moderate Impairment |
| `dementia` | 2 | No Impairment |
| `dementia` | 3 | Very Mild Impairment |
| `pleural_effusion` | 0 | Pleural Effusion |
| `pleural_effusion` | 1 | Normal |

These indices are important later when specifying a SPRUCE target class such as:

```yaml
counterfactual_class: 1
```

in the counterfactual-generation configuration.

---

# Image Preprocessing and Augmentation

Training images are processed with:

- random resized crop
- resize to the configured image size
- horizontal flip with probability 0.3
- random affine transformation
- brightness and contrast jitter
- conversion to tensor
- ImageNet normalization

The current training augmentation includes:

```text
RandomResizedCrop(scale = 0.75–1.0)
RandomHorizontalFlip(p = 0.3)
RandomAffine(
    degrees = 10,
    translate = 0.02,
    scale = 0.95–1.05,
    shear = 5
)
ColorJitter(
    brightness = 0.2,
    contrast = 0.2
)
```

Validation and test images are resized and normalized without random augmentation.

ImageNet normalization is:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

# Configuration

The three main scripts use separate YAML files under:

```text
configs/
```

---

## Training Configuration

File:

```text
configs/train.yaml
```

A portable example is:

```yaml
data:
  data_dir: "/path/to/dataset"
  batch_size: 32
  image_size: 256
  dataset_type: cardiomegaly

optim:
  lr: 5e-4
  weight_decay: 1e-2
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 5
  step_size: 30
  gamma: 0.1

adv:
  method: plain
  beta: 6
  epsilon: 0.016
  alpha: 0.0016
  pgd_steps: 10
  norm: l_inf
  adv_eval_freq: 5

train:
  epochs: 100
  patience: 10
  class_weights: [1.0, 1.382]
  label_smoothing: 0.01

model_arch: convnext
num_classes: 2
output_dir: "/path/to/experiment"
device: cuda
```

The committed configuration contains paths from the original experimental environment and should be adapted before use on another machine.

---

## Data Configuration

```yaml
data:
  data_dir: "/path/to/dataset"
  batch_size: 32
  image_size: 256
  dataset_type: cardiomegaly
```

| Parameter | Description |
|---|---|
| `data_dir` | Root directory containing `train/`, `val/`, and `test/` |
| `batch_size` | Number of samples per batch |
| `image_size` | Input image resolution |
| `dataset_type` | Dataset identifier used for class mapping |

---

## Optimizer and Scheduler

```yaml
optim:
  lr: 5e-4
  weight_decay: 1e-2
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 5
  step_size: 30
  gamma: 0.1
```

Supported optimizers include:

- AdamW
- SGD

Supported scheduler settings include:

- cosine
- step
- plateau-style fallback

A linear warmup phase is applied before the main learning-rate scheduler.

---

# Plain vs TRADES Training

The training mode is selected using:

```yaml
adv:
  method: plain
```

or:

```yaml
adv:
  method: trades
```

---

## Plain Training

For naturally trained classifiers, the model is optimized with cross-entropy loss.

The implementation also supports:

- class weighting
- label smoothing

through:

```yaml
train:
  class_weights: [1.0, 1.382]
  label_smoothing: 0.01
```

---

## TRADES Robust Training

SPRUCE also evaluates adversarially robust classifiers trained with TRADES.

The robust objective balances natural classification performance and robustness to adversarial perturbations.

The main configuration parameters are:

```yaml
adv:
  method: trades
  beta: 6
  epsilon: 0.016
  alpha: 0.0016
  pgd_steps: 10
  norm: l_inf
  adv_eval_freq: 5
```

| Parameter | Description |
|---|---|
| `beta` | Trade-off between natural classification loss and robustness |
| `epsilon` | Maximum adversarial perturbation magnitude |
| `alpha` | Adversarial optimization step size |
| `pgd_steps` | Number of perturbation steps |
| `norm` | Perturbation norm, e.g. `l_inf` |
| `adv_eval_freq` | Frequency of adversarial validation during training |

The SPRUCE experiments investigate how classifier robustness affects the realism, semantic alignment, and sparsity of generated counterfactual explanations.

---

# Step 1 — Train the Classifier

Script:

```text
scripts/train.py
```

From the `classifier_training/` directory:

```bash
python scripts/train.py \
    --config configs/train.yaml
```

The training script:

1. loads the train, validation, and test splits
2. initializes the configured classifier
3. trains using either plain or TRADES optimization
4. evaluates clean validation performance
5. optionally performs adversarial validation during TRADES training
6. checkpoints the best model according to validation accuracy
7. applies early stopping
8. restores the best checkpoint
9. evaluates the final model on the test set

---

## Training Outputs

A typical experiment directory contains:

```text
output_dir/
├── best_model.pth
├── loss_curves.png
├── accuracy_curve.png
├── confusion_matrix.png
├── metrics.json
└── training.log
```

### `best_model.pth`

Contains the checkpoint with the best validation accuracy.

This checkpoint is later used by:

- activation extraction
- confident image selection
- PIECE
- counterfactual latent optimization

### `metrics.json`

Stores the final clean test-set classification metrics.

### Training plots

The trainer saves:

- training and validation loss curves
- validation accuracy curve
- final test confusion matrix

---

# Step 2 — Extract Penultimate-Layer Activations

Script:

```text
scripts/save_activations.py
```

Configuration:

```text
configs/activations.yaml
```

Example:

```yaml
data:
  data_dir: "/path/to/dataset"
  batch_size: 128
  image_size: 256
  dataset_type: cardiomegaly

model:
  arch: convnext
  num_classes: 2
  weights: "/path/to/best_model.pth"

extract:
  save_path: "/path/to/class_activations.pkl"
```

Run:

```bash
python scripts/save_activations.py \
    --config configs/activations.yaml
```

The script:

1. loads the training split
2. loads the trained classifier
3. extracts penultimate-layer activations
4. obtains the predicted class for each training image
5. groups the feature vectors according to the classifier's predicted class
6. saves the resulting class-conditional activation dictionary

---

## Activation File Format

The resulting pickle has the form:

```python
{
    class_id: ndarray[num_samples, feature_dim]
}
```

Conceptually:

```text
Training images
      │
      ▼
Trained classifier
      │
      ▼
Penultimate-layer features
      │
      ▼
Grouped by predicted class
      │
      ▼
class_activations.pkl
```

These activations are later consumed by the PIECE component under:

[`../counterfactual_generation/`](../counterfactual_generation/)

to model target-class feature distributions and identify exceptional classifier features.

---

# Step 3 — Select Query Images

Script:

```text
scripts/save_images.py
```

Configuration:

```text
configs/correct_images.yaml
```

Example:

```yaml
data:
  data_dir: "/path/to/dataset"
  batch_size: 64
  image_size: 256
  dataset_type: cardiomegaly

model:
  arch: convnext
  num_classes: 2
  weights: "/path/to/best_model.pth"

extract:
  output_dir: "/path/to/query_images"
  class_number: 1
  threshold: 0.8
  num_images: 200
```

Run:

```bash
python scripts/save_images.py \
    --config configs/correct_images.yaml
```

The script scans the test split and saves images satisfying all of the following:

1. the image is correctly classified
2. its true class matches `class_number`
3. the classifier confidence for that class is at least `threshold`
4. the requested number of images has not yet been reached

Conceptually:

```text
Test image
   │
   ▼
Classifier
   │
   ├── correctly classified?
   ├── requested class?
   └── confidence ≥ threshold?
              │
              ▼
          save image
```

These selected images form the query set used in the next stage of SPRUCE.

---

# Connection to PTI

The query images generated by `save_images.py` are passed to the inversion pipeline under:

[`../PTI/`](../PTI/)

For each query image, PTI produces:

```text
W+ pivot latent
+
per-image tuned StyleGAN2-ADA generator
```

These artifacts are then used by the counterfactual-generation pipeline.

See:

[`../PTI/README.md`](../PTI/README.md)

for detailed inversion instructions.

---

# Connection to Counterfactual Generation

The classifier checkpoint and activation distributions are later used under:

[`../counterfactual_generation/`](../counterfactual_generation/)

The relationship is:

```text
best_model.pth
      │
      ├──────────────► PIECE feature extraction
      │
      └──────────────► counterfactual optimization

class_activations.pkl
      │
      └──────────────► target-class feature distributions

selected query images
      │
      └──────────────► PTI inversion
```

The complete downstream flow is therefore:

```text
classifier_training/
        │
        ├── best_model.pth
        ├── class_activations.pkl
        └── query images
                 │
                 ▼
                PTI/
                 │
                 ▼
counterfactual_generation/
```

---

# Running the Complete Classifier-Side Pipeline

From the repository root:

```bash
cd classifier_training
```

### 1. Train the model

```bash
python scripts/train.py \
    --config configs/train.yaml
```

### 2. Extract class activations

```bash
python scripts/save_activations.py \
    --config configs/activations.yaml
```

### 3. Select correctly classified query images

```bash
python scripts/save_images.py \
    --config configs/correct_images.yaml
```

After these steps, the classifier-side inputs required by SPRUCE are ready.

---

# Notes

- The committed YAML files contain paths from the original experimental environment and should be changed before use on another machine.
- A CUDA-capable GPU is strongly recommended for classifier training and TRADES.
- The same classifier architecture and weights should be used consistently when extracting activations and later generating counterfactuals.
- The class indices used during query selection and counterfactual generation should match the dataset mapping.
- The activation file is constructed from penultimate-layer features of the training split and grouped according to the model's predicted classes.
- High-confidence, correctly classified test images are recommended as query images for the downstream inversion and explanation pipeline.

---

# References

## TRADES

The adversarially robust training implementation is based on the TRADES methodology:

**Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan.**  
*Theoretically Principled Trade-off between Robustness and Accuracy.*

https://arxiv.org/abs/1901.08573

Official implementation:

https://github.com/yaodongyu/TRADES

Please cite the original TRADES work when using the adversarial training component.

For the complete set of SPRUCE references and publication information, see the [main repository README](../README.md).

---

## License

This project is released under the MIT License.

See [`../LICENSE`](../LICENSE) for details.
