# Counterfactual Generation

This folder implements part of the SPRUCE pipeline. Using the finetuned generators and the pivotal latent vectors 
acquired from the GAN inversion pipeline, we first compare the final reconstructed images with the original ones.We then compute
the counterfactual feature vectors of each image using the PIECE methodology and then generate the counterfactual images via
latent space optimization.


---

## 📂 Folder Structure

```

counterfactual\_generation/
├── configs/
│   └── config.yaml             # single YAML config for all scripts
├── scripts/
│   ├── batch_process_inversion.py  # saves the reconstructed images and compares them to the original ones
│   ├── compute_ideal_xp.py         # computes and saves the counterfactual feature vectors for each image
│   └── find_counterfactual.py      # latent‐space optimization → generates final counterfactual images
├── models/
│   ├── convnext.py      # ConvNeXt wrapper 
│   └── resnet50.py      # ResNet50 wrapper 
├── utils/
│   ├── io_utils.py              # image I/O, differencing, grid plotting
│   ├── piece_utils.py           # GMM fitting & PIECE core logic
│   ├── model_utils.py           # GAN & classifier loading functions
│   └── optim_utils.py           # latent‐vector optimization routines

├── dnnlib/                      # StyleGAN2-ADA core library
└── torch_utils/                 # PyTorch utility functions for GANs

````

---

## ⚙️ Configuration (`configs/config.yaml`)

All three scripts share one YAML file. Example structure:

```yaml
gan:
  original_weights:    "/path/to/original_G.pt"
  checkpoints_dir:     "/path/to/pti_generators/"
  pivot_latents_dir:   "/path/to/pivot_latents/"
  original_images_dir: "/path/to/original_images/"
  output_dir:          "/path/to/output_folder/"

classifier:
  arch:        "convnext"      # or "resnet50"
  weights:     "/path/to/classifier.pth"
  num_classes: 2
  normalized:  true

piece:
  counterfactual_class: 1
  activations:         "/path/to/activations.pkl"
  alpha:               0.5
  n_components:        5

opt:
  lr:            1e-3
  epochs:        200
  image_size:    256
  lambda_piece:  1.0
  lambda_perc:   0.1
  lambda_latent: 0.01
  lambda_image: 1.0
````

---

## 🚀 Scripts

### 1. `batch_process_inversion.py`

* **Purpose**:Using the pivot latent vectors and the fine‐tuned (PTI) StyleGAN generators acquired from the gan inversion pipeline
,we plot the final reconstructed image , the original one and their absolute differences so that we can assess the inversion qualitatively.
* **Input**: Finetuned GAN weights, pivot latents, original images (from `gan` section).
* **Output**:

  * Reconstructed images in `output_dir/inverted_images/`
  * Difference‐grids in `output_dir/inversion_comparisons/`

### 2. `compute_ideal_xp.py`

* **Purpose**: Compute “ideal” penultimate‐layer counterfactual feature vectors via PIECE for a target class.
* **Input**: Class activations pickle, PIECE hyperparameters (from `piece` section).
* **Output**:

  * One `*_cf_feature_vector.pickle` per image in `output_dir/feature_vectors/`
  * A file `features_changed.txt`, where we save the number of features changed for the construction of the counterfactual
  feature vector , along with the probability vector of the classifier for the original image.

### 3. `find_counterfactual.py`

* **Purpose**: Optimize StyleGAN latent codes to match PIECE feature vectors, yielding final counterfactual images.
* **Input**: PIECE counterfactual feature vectors, GAN & classifier settings (from `opt` & `gan` sections).
* **Output**:

  * Optimized counterfactual images in `output_dir/counterfactuals/`
  * Comparison grids in `output_dir/cf_comparisons/`
  * Metrics file `counterfactual_metrics.txt`

---

## 🛠️ Models & Utils

* **`models/`**

  * Wrap standard CNNs to expose logits + penultimate features.
* **`utils/`**

  * I/O, image differencing, GMM fitting, latent optimization helpers.
* **`dnnlib/` & `torch_utils/`**

  * Required dependencies for StyleGAN2-ADA GAN loading & sampling.

---

## 📖 Usage Examples

```bash
# 1. Inversion & PTI comparison
python -m scripts/batch_process_inversion --config configs/config.yaml

# 2. Compute PIECE feature vectors
python -m scripts/compute_ideal_xp       --config configs/config.yaml

# 3. Generate final counterfactual images
python -m scripts/find_counterfactual    --config configs/config.yaml
```

---

## 📄 License

This project is licensed under the MIT License.

