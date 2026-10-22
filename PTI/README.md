# Pivotal Tuning Inversion (PTI)

This directory contains the **GAN inversion stage used by SPRUCE**.

For each query image, the inversion pipeline produces the two artifacts required by the downstream counterfactual-generation stage:

1. a **pivot latent code** in StyleGAN's extended latent space \(W+\)
2. a **per-image fine-tuned StyleGAN2-ADA generator**

The pivot is obtained with an **E4E encoder** and is then kept fixed while **Pivotal Tuning Inversion (PTI)** fine-tunes the generator to reconstruct the query image more faithfully.

> **Important:** SPRUCE counterfactual generation requires **both** the saved pivot latent and the corresponding PTI-tuned generator.

---

## Role of PTI in SPRUCE

The inversion stage can be summarized as:

```text
Query medical image
        │
        ▼
      E4E
        │
        ▼
Pivot latent code in W+
        │
        ├──────────────► saved as 0.pt
        │
        ▼
Pivotal Tuning Inversion
(fine-tune generator while keeping pivot fixed)
        │
        ▼
Per-image tuned StyleGAN2-ADA generator
        │
        └──────────────► saved as model_<run_name>_<image_name>.pt
```

The resulting pair

```text
pivot latent + tuned generator
```

is later consumed by the code under [`../counterfactual_generation/`](../counterfactual_generation/) to initialize latent optimization and generate the final counterfactual.

---

## Directory Overview

The most relevant files are:

```text
PTI/
├── configs/
│   ├── global_config.py
│   ├── hyperparameters.py
│   └── paths_config.py
│
├── scripts/
│   ├── run_pti.py
│   └── latent_creators/
│
├── training/
│   ├── coaches/
│   │   ├── base_coach.py
│   │   ├── single_id_coach.py
│   │   └── multi_id_coach.py
│   └── projectors/
│
├── criteria/
├── models/
├── utils/
└── README.md
```

### Main entry point

[`scripts/run_pti.py`](scripts/run_pti.py) is the main script used for SPRUCE inversion.

By default it uses:

- **single-image PTI training**
- **E4E-based W+ initialization**
- optional Weights & Biases logging
- one image per batch

---

## Prerequisites

Before running PTI you need:

- a pretrained **StyleGAN2-ADA generator** for the image domain
- a compatible **E4E encoder checkpoint**
- the medical images to invert
- a CUDA-capable GPU

The repository-level dependencies should be installed first:

```bash
pip install -r requirements.txt
```

Run the commands below from the repository root unless stated otherwise.

---

## 1. Configure the Paths

Edit:

```text
PTI/configs/paths_config.py
```

Adapt the following paths to your environment.

The relevant entries are:

```python
# Pretrained models
e4e = "/path/to/e4e_checkpoint.pt"
stylegan2_ada = "/path/to/stylegan2_ada_generator.pkl"

# Output directories
checkpoints_dir = "/path/to/save/pti_generators"
embedding_base_dir = "/path/to/save/embeddings"

# Input images
input_data_path = "/path/to/images_to_invert"

# Identifier used to group one inversion experiment
input_data_id = "experiment_name"

# Output naming
pti_results_keyword = "PTI"
e4e_results_keyword = "e4e"
```
---

## 2. Configure PTI Hyperparameters

The inversion settings are defined in:

```text
PTI/configs/hyperparameters.py
```

The current configuration includes:

| Parameter | Current value | Description |
|---|---:|---|
| `first_inv_type` | `w+` | use E4E to initialize the W+ pivot |
| `lpips_type` | `vgg` | backbone used by LPIPS |
| `LPIPS_value_threshold` | `0.001` | early-stop threshold for PTI reconstruction |
| `max_pti_steps` | `1000` | maximum PTI steps per image |
| `max_images_to_invert` | `500` | maximum number of input images processed |
| `pti_learning_rate` | `1e-3` | learning rate for generator fine-tuning |
| `pt_l2_lambda` | `1` | L2 reconstruction-loss weight |
| `pt_lpips_lambda` | `1` | LPIPS reconstruction-loss weight |
| `use_locality_regularization` | `True` | enable PTI locality regularization |
| `locality_regularization_interval` | `5` | interval between locality-regularization steps |
| `regulizer_l2_lambda` | `0.1` | L2 locality-regularization coefficient |
| `regulizer_lpips_lambda` | `0.1` | LPIPS locality-regularization coefficient |
| `regulizer_alpha` | `10` | locality-regularization interpolation parameter |

GPU selection is controlled in:

```text
PTI/configs/global_config.py
```

with:

```python
cuda_visible_devices = "0"
device = "cuda:0"
```

Change these values if a different GPU should be used.

---

## 3. Prepare the Input Images

Set `input_data_path` to the directory containing the images that should be inverted.

The PTI data loader processes images one at a time:

```text
input_data_path/
├── image_001.png
├── image_002.png
├── image_003.png
└── ...
```

The main PTI pipeline applies tensor conversion and normalization to `[-1, 1]`. E4E inversion additionally resizes the image to **256 × 256** before encoding it.

Use images drawn from the same domain as the StyleGAN2-ADA generator and E4E model.

---

## 4. Run PTI

From the repository root:

```bash
python PTI/scripts/run_pti.py
```

The script currently calls:

```python
run_PTI(
    run_name="",
    use_wandb=True,
    use_multi_id_training=False
)
```

An empty `run_name` causes a random run identifier to be generated.

If you do not want Weights & Biases logging, change the call to:

```python
run_PTI(
    run_name="my_run",
    use_wandb=False,
    use_multi_id_training=False
)
```

---

## 5. Output Files

### Pivot latent

For each image, the pivot latent is saved as:

```text
<embedding_base_dir>/
└── <input_data_id>/
    └── PTI/
        └── <image_name>/
            └── 0.pt
```

For example:

```text
embeddings/
└── cardiomegaly/
    └── PTI/
        └── image_001.png/
            └── 0.pt
```

The saved tensor is the **W+ pivot** used to reconstruct the original image and initialize the subsequent SPRUCE latent optimization.

### Fine-tuned generator

The corresponding PTI-tuned generator is saved to:

```text
<checkpoints_dir>/
└── model_<run_name>_<image_name>.pt
```

For example:

```text
checkpoints/
└── model_MYRUN_image_001.png.pt
```

Each image therefore has a matching pair:

```text
embeddings/.../PTI/image_001.png/0.pt
checkpoints/model_MYRUN_image_001.png.pt
```

These two files should be kept together for downstream counterfactual generation.

---

## What Happens Internally?

For every query image, the single-image PTI coach performs the following sequence:

1. reload the original pretrained StyleGAN2-ADA generator
2. compute an E4E inversion in W+
3. save the W+ pivot
4. freeze the pivot latent
5. optimize the generator parameters to improve reconstruction
6. minimize a combination of:
   - L2 reconstruction loss
   - LPIPS reconstruction loss
   - locality regularization
7. stop early when LPIPS falls below the configured threshold, or stop after `max_pti_steps`
8. save the tuned generator checkpoint

Conceptually:

```text
w_p = E4E(x)

G* = argmin_G [
    L2(G(w_p), x)
    + LPIPS(G(w_p), x)
    + locality regularization
]
```

where `w_p` remains fixed during pivotal tuning.

---

## Using the PTI Outputs in SPRUCE

After inversion, the counterfactual-generation stage requires:

### 1. The original/query image

Used as the reconstruction and sparsity reference.

### 2. The W+ pivot

```text
<embedding_base_dir>/<input_data_id>/PTI/<image_name>/0.pt
```

### 3. The matching tuned generator

```text
<checkpoints_dir>/model_<run_name>_<image_name>.pt
```

The downstream configuration in [`../counterfactual_generation/`](../counterfactual_generation/) should point to the directories containing these artifacts.

The counterfactual optimization then starts from the PTI pivot rather than from a random latent code. This helps preserve patient-specific structure while allowing the latent representation to move toward the desired counterfactual classifier features.

---

## Notes

- PTI is computationally expensive because the generator is optimized separately for each query image.
- The E4E pivot and tuned generator must correspond to the **same query image**.
- The generator and E4E model should correspond to the same medical-image domain used by the downstream counterfactual experiment.
- If `use_wandb=True`, make sure Weights & Biases is configured in your environment.

---

## References

This implementation builds on:

- **Pivotal Tuning Inversion (PTI)** for high-fidelity GAN inversion
- **E4E** for editable encoder-based StyleGAN inversion
- **StyleGAN2-ADA** for the underlying generator

Please cite the corresponding original works when using these components.

For the complete SPRUCE pipeline, return to the [main repository README](../README.md).
