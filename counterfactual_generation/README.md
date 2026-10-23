# Counterfactual Generation

This directory implements the **counterfactual-generation stage of SPRUCE**.

It assumes that each query image has already been processed by the GAN inversion pipeline under [`../PTI/`](../PTI/) and therefore has:

1. its **original image**
2. a **W+ pivot latent**
3. a corresponding **per-image PTI-tuned StyleGAN2-ADA generator**

Using these artifacts, this module:

1. optionally evaluates the quality of the PTI reconstruction
2. computes a target **counterfactual feature vector** using the PIECE methodology
3. optimizes the latent representation in W+ to generate a sparse counterfactual image

The complete flow is:

```text
Original query image
        +
W+ pivot latent
        +
PTI-tuned generator
        │
        ▼
Reconstruction assessment
        │
        ▼
Classifier feature extraction
        │
        ▼
PIECE exceptional-feature analysis
        │
        ▼
Counterfactual feature vector
        │
        ▼
Latent-space optimization in W+
        │
        ▼
Counterfactual image
```

---

## Required Inputs

Before running the scripts in this directory, make sure you have:

- the original query images
- a PTI-tuned generator for each image
- the matching W+ pivot latent for each image
- a trained classifier checkpoint
- stored penultimate-layer classifier activations
- a configured target counterfactual class

The PTI outputs can be generated using the pipeline documented in:

[`../PTI/README.md`](../PTI/README.md)

For each image, the inversion stage should provide a matching pair:

```text
PTI-tuned generator
+
W+ pivot latent
```

Both are required for downstream counterfactual generation.

---

## Folder Structure

```text
counterfactual_generation/
├── configs/
│   └── counterfactual_config.yaml
│
├── scripts/
│   ├── batch_process_inversion.py
│   ├── compute_ideal_xp.py
│   └── find_counterfactual.py
│
├── models/
│   ├── convnext.py
│   ├── efficientnet.py
│   └── resnet50.py
│
├── utils/
│   ├── io_utils.py
│   ├── model_utils.py
│   ├── optim_utils.py
│   └── piece_utils.py
│
├── dnnlib/
├── torch_utils/
└── README.md
```

### Main components

- **`configs/`**  
  Contains the central YAML configuration shared by the counterfactual-generation scripts.

- **`scripts/`**  
  Implements the three main stages of the SPRUCE counterfactual pipeline.

- **`models/`**  
  Contains classifier wrappers that expose both logits and penultimate-layer features.

- **`utils/`**  
  Contains helper functions for image processing, model loading, PIECE feature analysis, and latent optimization.

- **`dnnlib/` and `torch_utils/`**  
  StyleGAN2-ADA support code required for generator loading and synthesis.

---

# Configuration

All three scripts use:

```text
configs/counterfactual_config.yaml
```

A portable example configuration is:

```yaml
gan:
  original_weights: "/path/to/stylegan2_ada.pkl"
  checkpoints_dir: "/path/to/pti_generators"
  embeddings_dir: "/path/to/pti_embeddings"
  original_images_dir: "/path/to/query_images"
  output_dir: "/path/to/output"

classifier:
  type: "convnext"
  weights: "/path/to/classifier.pth"
  num_classes: 2
  normalized: true

piece:
  counterfactual_class: 1
  class_activations: "/path/to/class_activations.pkl"
  alpha: 0.05
  n_components: 5

opt:
  lr: 0.02
  epochs: 200
  lambda_piece: 1.0
  lambda_perc: 0.01
  lambda_latent: 1e-3
  lambda_image: 1e-6
  image_size: 256
```

Replace the placeholder paths in the configuration file with paths appropriate for your environment.

---

## GAN / Inversion Configuration

```yaml
gan:
  original_weights: "/path/to/stylegan2_ada.pkl"
  checkpoints_dir: "/path/to/pti_generators"
  embeddings_dir: "/path/to/pti_embeddings"
  original_images_dir: "/path/to/query_images"
  output_dir: "/path/to/output"
```

| Parameter | Description |
|---|---|
| `original_weights` | Original pretrained StyleGAN2-ADA generator |
| `checkpoints_dir` | Directory containing the per-image PTI-tuned generators |
| `embeddings_dir` | Directory containing the W+ pivot latents |
| `original_images_dir` | Directory containing the original query images |
| `output_dir` | Root directory for all generated outputs |

The PTI outputs are expected to follow the structure documented in [`../PTI/README.md`](../PTI/README.md).

---

## Classifier Configuration

```yaml
classifier:
  type: "convnext"
  weights: "/path/to/classifier.pth"
  num_classes: 2
  normalized: true
```

| Parameter | Description |
|---|---|
| `type` | Classifier architecture wrapper |
| `weights` | Path to the trained classifier checkpoint |
| `num_classes` | Number of classifier output classes |
| `normalized` | Whether classifier-specific input preprocessing/normalization should be applied |

Supported classifier wrappers currently include:

```text
convnext
efficientnet
resnet50
```

Each wrapper is designed to return both:

- classification logits
- penultimate-layer feature representations

The feature representation is required by both PIECE and the counterfactual feature-alignment loss.

---

## PIECE Configuration

```yaml
piece:
  counterfactual_class: 1
  class_activations: "/path/to/class_activations.pkl"
  alpha: 0.05
  n_components: 5
```

| Parameter | Description |
|---|---|
| `counterfactual_class` | Target class toward which the counterfactual should move |
| `class_activations` | Stored penultimate-layer classifier activations for the reference population |
| `alpha` | Threshold used to identify statistically exceptional features |
| `n_components` | Number of Gaussian-mixture components used to model feature distributions |

The stored class activations are used to estimate the feature distributions of the target class.

For each query image, SPRUCE compares its classifier features against these distributions and identifies features that are exceptional with respect to the desired counterfactual class.

---

## Optimization Configuration

```yaml
opt:
  lr: 0.02
  epochs: 200
  lambda_piece: 1.0
  lambda_perc: 0.01
  lambda_latent: 1e-3
  lambda_image: 1e-6
  image_size: 256
```

| Parameter | Description |
|---|---|
| `lr` | Latent optimization learning rate |
| `epochs` | Number of optimization iterations |
| `lambda_piece` | Weight of the counterfactual feature-alignment loss |
| `lambda_perc` | Weight of the LPIPS perceptual loss |
| `lambda_latent` | Weight of latent-space L1 regularization |
| `lambda_image` | Weight of image-space L1 regularization |
| `image_size` | Image resolution used during optimization |

These values are experiment-dependent and may require tuning for a different dataset, generator, or classifier.

---

# Step 1 — Assess the PTI Reconstruction

Script:

```text
scripts/batch_process_inversion.py
```

This step is **optional but strongly recommended**.

Its purpose is to evaluate whether the PTI inversion reconstructs each query image sufficiently well before counterfactual optimization begins.

The script loads:

- the original pretrained StyleGAN2-ADA generator
- the PTI-tuned generator
- the W+ pivot latent
- the original image

It then synthesizes:

```text
Original image
Before PTI reconstruction
After PTI reconstruction
```

and computes absolute difference maps between the original image and both reconstructions.

### Output

```text
output_dir/
├── inverted_images/
│   └── <image_name>.png
│
└── inversion_comparisons/
    └── <image_name>_comparison.png
```

The comparison grid contains:

```text
Original | Before PTI | After PTI
         | Difference | Difference
```

This step is primarily diagnostic and is not required to compute the counterfactual feature vector itself.

---

# Step 2 — Construct the Counterfactual Feature Vector

Script:

```text
scripts/compute_ideal_xp.py
```

This stage implements the PIECE-based exceptional-feature analysis used by SPRUCE.

For every query image, the script:

1. loads the corresponding PTI-tuned generator
2. loads the matching W+ pivot latent
3. reconstructs the query image
4. passes the reconstructed image through the classifier
5. extracts the penultimate-layer representation
6. compares the query features against the target-class feature distributions
7. identifies statistically exceptional features
8. filters exceptional features according to the PIECE methodology
9. replaces selected exceptional values with expected values under the target class
10. constructs the final counterfactual feature vector

Conceptually:

```text
query feature vector
        │
        ▼
target-class feature distributions
        │
        ▼
exceptional feature detection
        │
        ▼
exceptional → expected replacement
        │
        ▼
counterfactual feature vector
```

---

## Stored Class Activations

The file specified by:

```yaml
piece:
  class_activations: "/path/to/class_activations.pkl"
```

contains stored classifier representations of the reference data.

These activations are used to model target-class feature distributions.

The current SPRUCE implementation uses Gaussian mixture models for feature-distribution estimation, controlled by:

```yaml
n_components: 5
```

---

## Counterfactual Feature Output

For each image, the script creates:

```text
output_dir/
└── feature_vectors/
    └── <image_name>_cf_feature_vector.pickle
```

Each pickle contains:

```python
{
    "cf_feature_vector": cf_x,
    "ws": w_pivot
}
```

where:

- `cf_feature_vector` is the PIECE-derived target classifier representation
- `ws` is the original W+ pivot latent

The pivot is stored together with the feature vector so that the final latent optimization begins from the same inverted representation.

---

## Feature Summary

The script also creates:

```text
output_dir/features_changed.txt
```

This file records:

- the classifier probability vector for each reconstructed query image
- the number of exceptional features modified to form the target counterfactual representation

---

# Step 3 — Generate the Counterfactual

Script:

```text
scripts/find_counterfactual.py
```

This stage performs the final latent-space optimization.

For each query image, the script loads:

- the PTI-tuned generator
- the original query image
- the W+ pivot
- the PIECE counterfactual feature vector
- the classifier

The optimization begins from the pivot:

```text
w_initial = w_pivot
```

and searches for an optimized latent representation:

```text
w_opt
```

such that the generated image approaches the desired counterfactual classifier representation while remaining visually close to the original image.

---

# SPRUCE Optimization Objective

The counterfactual latent is optimized using a combination of four objectives:

```math
\mathcal{L}
=
\lambda_{\mathrm{piece}}\mathcal{L}_{\mathrm{piece}}
+
\lambda_{\mathrm{perc}}\mathcal{L}_{\mathrm{LPIPS}}
+
\lambda_{\mathrm{latent}}\mathcal{L}_{\mathrm{latent}}
+
\lambda_{\mathrm{image}}\mathcal{L}_{\mathrm{image}}

```
### Counterfactual feature-alignment loss

```math
\mathcal{L}_{\mathrm{piece}}
```

encourages the classifier representation of the generated image to approach the PIECE-derived counterfactual feature vector.

This is the main term driving the prediction toward the desired counterfactual state.

### LPIPS perceptual loss

```math
\mathcal{L}_{\mathrm{LPIPS}}
```

encourages perceptual similarity between the generated counterfactual and the original image.

### Latent-space regularization

```math
\mathcal{L}_{\mathrm{latent}}
```

penalizes large deviations from the PTI pivot latent.

This helps preserve the local latent neighborhood associated with the original patient image.

### Image-space sparsity

```math
\mathcal{L}_{\mathrm{image}}
```

penalizes pixel-level differences between the original and generated image.

This term encourages sparse image modifications.

---

# Counterfactual Outputs

For each successfully processed query image, SPRUCE saves the final counterfactual to:

```text
output_dir/
└── counterfactuals/
    └── <image_name>_cf.png
```

---

## Comparison Grids

A qualitative comparison is saved to:

```text
output_dir/
└── cf_comparisons/
    └── <image_name>_comparison.png
```

The grid includes:

```text
Original Image
Inverted Original
Counterfactual Image

Difference: Original vs Inverted
Difference: Original vs Counterfactual
Difference: Inverted vs Counterfactual
```

The difference maps are useful for visually assessing the sparsity and localization of the edits.

---

## Counterfactual Metrics

The script appends results to:

```text
output_dir/counterfactual_metrics.txt
```

with the columns:

```text
Image,Normalized_L1,Pred_Prob,Pred_Class
```

| Field | Description |
|---|---|
| `Image` | Query-image identifier |
| `Normalized_L1` | Mean absolute pixel difference between the original and counterfactual |
| `Pred_Prob` | Maximum classifier probability for the generated counterfactual |
| `Pred_Class` | Final predicted class |

The normalized L1 score provides a simple measure of image-space sparsity.

---

## Experiment Logging

The final generation script also creates:

```text
output_dir/run.log
```

and copies the configuration used for the experiment to:

```text
output_dir/config.yaml
```

This makes it easier to reproduce the exact configuration associated with a generated set of counterfactuals.

---

# Complete Output Structure

After running all stages, a typical experiment directory looks like:

```text
output_dir/
├── inverted_images/
│   └── <image_name>.png
│
├── inversion_comparisons/
│   └── <image_name>_comparison.png
│
├── feature_vectors/
│   └── <image_name>_cf_feature_vector.pickle
│
├── features_changed.txt
│
├── counterfactuals/
│   └── <image_name>_cf.png
│
├── cf_comparisons/
│   └── <image_name>_comparison.png
│
├── counterfactual_metrics.txt
├── run.log
└── config.yaml
```

---

# Running the Pipeline

From the repository root:

```bash
cd counterfactual_generation
```

### 1. Evaluate PTI reconstruction

```bash
python scripts/batch_process_inversion.py \
    --config configs/counterfactual_config.yaml
```

### 2. Compute PIECE counterfactual feature vectors

```bash
python scripts/compute_ideal_xp.py \
    --config configs/counterfactual_config.yaml
```

### 3. Generate counterfactual images

```bash
python scripts/find_counterfactual.py \
    --config configs/counterfactual_config.yaml
```

The first stage is optional if reconstruction quality has already been verified.

The required counterfactual-generation sequence is therefore:

```text
compute_ideal_xp.py
        │
        ▼
find_counterfactual.py
```

provided that the PTI outputs already exist.

---

# Filename Conventions

The scripts associate:

- a PTI generator
- a pivot latent
- an original image
- a counterfactual feature pickle

through the image identifier encoded in their filenames.

A generator named:

```text
model_<run_name>_<image_name>.pt
```

is interpreted as belonging to:

```text
embeddings_dir/<image_name>/0.pt
original_images_dir/<image_name>.png
feature_vectors/<image_name>_cf_feature_vector.pickle
```

For example:

```text
model_ABCD_image_001.pt
```

should correspond to:

```text
embeddings_dir/image_001/0.pt
original_images_dir/image_001.png
feature_vectors/image_001_cf_feature_vector.pickle
```

Maintaining this naming convention is important because the batch scripts infer the corresponding files automatically.

---

# Models

The `models/` directory currently provides wrappers for:

```text
ConvNeXt
EfficientNet
ResNet-50
```

These wrappers expose both:

```text
classification logits
+
penultimate-layer features
```

The penultimate features are required for:

- PIECE exceptional-feature analysis
- construction of the target counterfactual representation
- feature-alignment during latent optimization

---

# Utilities

## `utils/io_utils.py`

Handles:

- image loading
- tensor/image conversion
- difference-image generation
- classifier preprocessing
- visualization support

## `utils/model_utils.py`

Handles:

- classifier loading
- StyleGAN generator loading
- PTI generator loading

## `utils/piece_utils.py`

Implements the PIECE-related logic, including:

- target-class feature-distribution modeling
- exceptional-feature probability estimation
- exceptional-feature filtering
- construction of the counterfactual feature vector

## `utils/optim_utils.py`

Implements the W+ latent optimization used to generate the final SPRUCE counterfactual.

---

# Notes

- The classifier, generator, class activations, and query images should correspond to the same medical-imaging domain.
- Each PTI generator must be paired with the correct pivot latent and original query image.
- The `counterfactual_class` must be a valid classifier class index.
- Optimization hyperparameters are dataset- and classifier-dependent.
- A CUDA-capable GPU is strongly recommended.
- Replace the placeholder paths in the YAML configuration with paths appropriate for your environment.
- If a corresponding pivot, original image, feature vector, or tuned generator cannot be found, the batch scripts skip that image.

---

# References

The counterfactual-generation stage of SPRUCE builds on:

- **PIECE**  
  Eoin M. Kenny and Mark T. Keane.  
  *On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning.*  
  https://arxiv.org/abs/2009.06399

- **StyleGAN2-ADA**  
  Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila.  
  *Training Generative Adversarial Networks with Limited Data.*  
  https://arxiv.org/abs/2006.06676

- **Pivotal Tuning Inversion (PTI)**  
  Daniel Roich, Ron Mokady, Amit H. Bermano, and Daniel Cohen-Or.  
  *Pivotal Tuning for Latent-based Editing of Real Images.*  
  https://arxiv.org/abs/2106.05744

For the complete SPRUCE methodology, experiments, pretrained model links, and additional references, see the [main repository README](../README.md).

For inversion instructions, see [`../PTI/README.md`](../PTI/README.md).

---

## License

This project is released under the MIT License.

See [`../LICENSE`](../LICENSE) for details.
