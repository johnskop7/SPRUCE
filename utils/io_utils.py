import torch
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lpips
import kornia
import kornia.morphology as km


def load_original_image(image_path, device, size=256):
    """Load and preprocess the original image from PNG format."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1] (StyleGAN format)
    ])

    image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image




def convert_tensor_to_png(image_tensor, filename='output_image.png', normalized=False,image_size=256):
  mean=[0.485, 0.456, 0.406]
  std=[0.229, 0.224, 0.225]
  resolution = image_size
  if normalized :
    tensor = image_tensor.detach().cpu().clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = tensor.clamp(0, 1).mul(255).byte()

    if tensor.shape[0] == 3:  # RGB
        array = tensor.numpy()
        array = np.transpose(array, (1, 2, 0))  # Convert from CHW to HWC
        image = Image.fromarray(array, 'RGB')
    elif tensor.shape[0] == 1:  # Grayscale
        array = tensor.squeeze(0).numpy()
        image = Image.fromarray(array, 'L')
    else:
        raise ValueError("Unsupported number of channels. Expected 1 (grayscale) or 3 (RGB).")

    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.LANCZOS)

    image.save(filename)
    print(f"Image saved to {filename}")
  else :
    tensor = image_tensor.mul(255).byte()

    if tensor.shape[0] == 3:  # RGB
        array = tensor.numpy()
        array = np.transpose(array, (1, 2, 0))
        image = Image.fromarray(array, 'RGB')
    elif tensor.shape[0] == 1:  # Grayscale
        array = tensor.squeeze(0).numpy()
        image = Image.fromarray(array, 'L')  # 'L' is for grayscale
    else:
        raise ValueError("Unsupported number of channels. Expected 1 (grayscale) or 3 (RGB).")

    image.save(filename)
  return image


def tensor_to_pil(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def reverse_normalization(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Reverse normalization for an image tensor normalized with ImageNet mean and std.
    Args:
        tensor (torch.Tensor): Normalized image tensor (C, H, W).
        mean (list): Mean used for normalization.
        std (list): Standard deviation used for normalization.
    Returns:
        np.ndarray: De-normalized image in [0, 255] range (H, W, C).
    """
    # Check that the tensor is 3D
    if len(tensor.shape) != 3 or tensor.shape[0] != 3:
        raise ValueError("Input tensor must be in [C, H, W] format with 3 channels.")

    # Reverse normalization: multiply by std and add mean
    tensor = tensor.clone()  # Avoid modifying the original tensor
    for c, (m, s) in enumerate(zip(mean, std)):
        tensor[c] = tensor[c] * s + m

    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to NumPy array and transpose to [H, W, C] for display
    image = tensor.permute(1, 2, 0).detach().cpu().numpy()  # Convert to HWC format
    image = (image * 255).astype(np.uint8)  # Scale to [0, 255]
    return image

def update(frame, ax, number_of_classes=2, normalized=False):
    """
    Updates the plot with the current frame and probabilities.
    Args:
        frame (tuple): Contains the image tensor and its associated probabilities.
        ax (matplotlib.axes.Axes): Axis to plot on.
        number_of_classes (int): Number of classes in the dataset (default: 2).
        normalized (bool): Whether the input images are normalized and need denormalization.
    """
    image, probs = frame

    # Denormalize the image if needed
    if normalized:
        image = reverse_normalization(image)  # Returns a NumPy array in [H, W, C]
    else:
        # Convert tensor to NumPy array in [H, W, C] format
        image = image.permute(1, 2, 0).detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)  # Scale to [0, 255] if not normalized

    ax.clear()
    ax.imshow(image)

    # Display probabilities
    if number_of_classes == 2:
        prob1, prob2 = probs[0][0].detach().cpu().numpy(), probs[0][1].detach().cpu().numpy()
        prob_text = f'Prob 1: {prob1:.2f}, Prob 2: {prob2:.2f}'
    elif number_of_classes == 4:
        prob1, prob2, prob3, prob4 = (probs[0][i].detach().cpu().numpy() for i in range(4))
        prob_text = f'Prob 1: {prob1:.2f}, Prob 2: {prob2:.2f}, Prob 3: {prob3:.2f}, Prob 4: {prob4:.2f}'

    ax.text(0.5, -0.1, prob_text, ha='center', va='top', transform=ax.transAxes, fontsize=10)
    ax.axis("off")  # Hide axes for a cleaner view
    return ax

def latent_vector_optimization(
    generator,
    w_plus_init,
    device,
    classifier,
    ideal_xp,
    original_image,
    epochs=100,
    learning_rate=0.01,
    lambda_piece=1.0,
    lambda_perc=0.5,
    lambda_latent=1e-2,
    lambda_image=0.1,
    image_size=256,
    normalized=False
):


    # 1. Convert the original image (NumPy) to a torch.Tensor in [-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    original_image = transform(original_image).unsqueeze(0).to(device)  # [B, C, H, W]

    # 2. Initialize w_e as a trainable copy of w_plus_init
    w_e = w_plus_init.clone().detach().to(device).requires_grad_(True)

    # 3. Set up an optimizer for the entire latent vector
    optimizer = optim.AdamW([w_e], lr=learning_rate, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Define loss functions
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    mse_loss = nn.MSELoss()

    images_for_gif = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Generate an image from the current latent vector
        image = generator.synthesis(w_e, noise_mode='const', force_fp32=True)

        # Calculate LPIPS loss
        lpips_loss = lpips_fn(image, original_image).mean()

        # Calculate normalized L1 difference
        image_diff_l1 = torch.norm(image - original_image, p=1)
        num_pixels = image.numel()
        l1_norm_normalized = image_diff_l1 / num_pixels

        # Shift image to [0,1] for classifier input if needed
        image = (image * 0.5 + 0.5).clamp(0, 1)

        # Preprocess if the classifier expects normalized input
        if normalized:
            image_preproc = preprocess_for_classifier(image)
        else:
            image_preproc = image

        # Forward pass through classifier
        logits, feats = classifier(image_preproc, return_features=True)

        # Feature alignment loss (PIECE)
        piece_loss = mse_loss(feats[0], ideal_xp)

        # Latent regularization (L1)
        latent_reg = torch.norm(w_e - w_plus_init.to(device), p=1)

        # Final combined loss
        loss = (
            lambda_piece * piece_loss +
            lambda_perc * lpips_loss +
            lambda_latent * latent_reg +
            lambda_image * image_diff_l1
        )
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([w_e], max_norm=1.0)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1).cpu()
            images_for_gif.append((image_preproc.squeeze(0).cpu(), probs))

        if epoch % 2 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] Loss={loss.item():.4f} | "
                f"Piece={piece_loss.item():.4f} | LPIPS={lpips_loss.item():.4f} | "
                f"L1={l1_norm_normalized.item():.4f}"
            )

    return w_e.detach(), images_for_gif


def preprocess_for_classifier(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)
    image_normalized = (image_tensor - mean) / std

    return image_normalized

def compute_class_probabilities(image, model, device):
    # Check if the tensor is 3D and convert to 4D if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    logits,features = model(image)
    probs = torch.softmax(logits, dim=1)
    _, predicted_labels = torch.max(probs, dim=1)
    return probs

def load_reconstructed_image(w_pivot,classifier,device,gan, normalized = False):
  #Generate the image and normalize it
  synthetic_image_initial = gan.synthesis(w_pivot,noise_mode = 'const',force_fp32=True)
  synthetic_image_initial = (synthetic_image_initial * 0.5 + 0.5).clamp(0, 1) #normalize the image from [-1,1] to [0,1]

  if normalized :
    # Preprocess the synthetic image for the classifier
    synthetic_image_initial = preprocess_for_classifier(synthetic_image_initial)
    synthetic_image_initial = synthetic_image_initial.to(device)

  probs = compute_class_probabilities(synthetic_image_initial,classifier,device)
  return synthetic_image_initial,probs[0].detach().cpu().numpy()
