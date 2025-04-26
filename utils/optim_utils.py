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
from utils.io_utils import preprocess_for_classifier

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
    lambda_piece = float(lambda_piece)
    lambda_perc = float(lambda_perc)
    lambda_latent = float(lambda_latent)
    lambda_image = float(lambda_image)

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
                piece_loss * lambda_piece +
                lpips_loss * lambda_perc +
                latent_reg * lambda_latent +
                image_diff_l1 * lambda_image
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
                f"Piece={piece_loss.item():.4f} | LPIPS={lpips_loss.item():.4f} | Latent_reg ={latent_reg.item():.4f} |"
                f"L1={l1_norm_normalized.item():.4f}"
            )

    return w_e.detach(), images_for_gif


def total_variation_isotropic(x):
    """Calculates the total isotropic variation of an image.

    Args:
        x: The input image tensor.

    Returns:
        The total variation of the image.
    """
    # Pad the input tensor with zeros along dimensions 2 and 3
    x = F.pad(x, (0, 1, 0, 1), mode='replicate')

    # Calculate the differences in pixel values along the x and y axes
    dx = x[:, :, 1:, :-1] - x[:, :, :-1, :-1]
    dy = x[:, :, :-1, 1:] - x[:, :, :-1, :-1]

    # Calculate the total variation using the padded differences
    tv = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8).mean()

    return tv


def total_variation_anisotropic(x):
    """
    Simple anisotropic TV for a 2D image (assuming shape [N, 1, H, W]).
    """
    diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return diff_h + diff_w


def plot_mask_and_generated_image(x_orig, x_cf, mask, epoch):
    """Utility function to plot original, counterfactual, and mask."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(x_orig[0].permute(1, 2, 0).detach().cpu().numpy())
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(x_cf[0].permute(1, 2, 0).detach().cpu().numpy())
    ax[1].set_title(f"Counterfactual Image (Epoch {epoch})")
    ax[1].axis("off")

    ax[2].imshow(mask[0, 0].detach().cpu().numpy(), cmap="jet")
    ax[2].set_title("Mask")
    ax[2].axis("off")

    plt.show()


def smooth_mask_with_morphology(m, kernel_size=3):
    """
    Apply morphological opening + closing to smooth a single-channel mask.
    m shape: [N, 1, H, W]
    kernel_size: int
    """
    # Make sure m is 4D: [N, C=1, H, W]
    assert m.ndim == 4, f"Expected [N,1,H,W], got shape {m.shape}"

    # Create a 2D kernel
    kernel = torch.ones((kernel_size, kernel_size), device=m.device)

    # Opening = Erosion then Dilation
    m_opened = km.opening(m, kernel)

    # Closing = Dilation then Erosion
    m_closed = km.closing(m_opened, kernel)

    return m_closed


def latent_vector_optimization_with_mask(
        learning_rate,
        epochs,
        generator,
        w_s,
        device,
        classifier,
        ideal_xp,
        lambda_mask_sparsity=1e-3,
        lambda_mask_tv=1e-4,
        lambda_piece=1.0,
        lambda_latent_reg=1e-4,
        lambda_perc=0.1,
        lambda_diff=0.1,
        normalized=False,
        morphological=False,
        gaussian=False
):
    with torch.no_grad():
        x_orig = generator.synthesis(w_s, noise_mode='const', force_fp32=True)
        x_orig = (x_orig * 0.5 + 0.5).clamp(0, 1)

    # Initialize
    w_e = w_s.clone().detach().requires_grad_(True)
    m = torch.zeros_like(x_orig[:, :1, :, :], requires_grad=True)

    optimizer = torch.optim.AdamW([w_e, m], betas=(0.9, 0.99), lr=learning_rate)

    mse_criterion = torch.nn.MSELoss()
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Generate from w_e
        x_gen = generator.synthesis(w_e, noise_mode='const', force_fp32=True)
        x_gen = (x_gen * 0.5 + 0.5).clamp(0, 1)

        m_clamped = torch.sigmoid(m)
        if morphological:
            m_clamped = smooth_mask_with_morphology(m_clamped, kernel_size=3)
        elif gaussian:
            m_clamped = kornia.filters.gaussian_blur2d(m_clamped, (3, 3), (1, 1))

        x_cf = (1 - m_clamped) * x_orig + m_clamped * x_gen

        # Preprocess for classifier
        classifier_input = x_cf
        if normalized:
            classifier_input = preprocess_for_classifier(classifier_input)

        logits, x_e = classifier(classifier_input, return_features=True)
        piece_loss = mse_criterion(x_e[0], ideal_xp)

        # Mask regularization
        mask_sparsity = m_clamped.abs().mean()
        mask_tv = total_variation_isotropic(m_clamped)

        # Latent regularization
        latent_reg = torch.norm(w_e - w_s, p=2)

        # Perceptual loss
        lpips_loss = lpips_fn(x_gen, x_orig).mean()

        diff_loss = ((x_gen - x_orig) * m_clamped).abs().mean()

        # Combine
        loss = (lambda_piece * piece_loss +
                lambda_mask_sparsity * mask_sparsity +
                lambda_mask_tv * mask_tv +
                lambda_latent_reg * latent_reg +
                lambda_perc * lpips_loss +
                lambda_diff * diff_loss
                )

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            probs = torch.softmax(logits, dim=1)
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.4f} "
                  f"| PIECE: {piece_loss.item():.4f} | L1(mask): {mask_sparsity.item():.4f} "
                  f"| TV(mask): {mask_tv.item():.4f} | LatentReg: {latent_reg.item():.4f} "
                  f"| LPIPS: {lpips_loss.item():.4f}")
            plot_mask_and_generated_image(x_orig, x_cf, m_clamped, epoch)
            print('The class probabilities of the masked counterfactual are:', probs[0].detach().cpu().numpy())
            print("---------------------------------------------------------------------------------")

    return w_e.detach(), m_clamped.detach(), x_orig.detach(), x_gen.detach()