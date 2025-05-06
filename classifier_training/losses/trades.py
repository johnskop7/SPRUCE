import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def trades_loss(
        model: nn.Module,
        x_natural: torch.Tensor,
        y: torch.Tensor,
        step_size: float,
        epsilon: float,
        perturb_steps: int,
        beta: float,
        distance: str = 'l_inf'
    ) -> torch.Tensor:
    """
    Compute the TRADES loss as described in: https://arxiv.org/abs/1901.08573

    Args:
        model: classifier model (outputs logits).
        x_natural: natural input images.
        y: ground-truth labels.
        step_size: step size for adversarial perturbation (alpha).
        epsilon: maximum perturbation magnitude.
        perturb_steps: number of PGD steps.
        beta: trade-off coefficient between natural and robust loss.
        distance: 'l_inf' or 'l2' norm for perturbations.

    Returns:
        Combined TRADES loss tensor.
    """
    # KL divergence for robust loss
    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    # Generate adversarial example start: random noise within epsilon ball
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_adv = model(x_adv)
                logits_nat = model(x_natural)
                loss_kl = criterion_kl(
                    F.log_softmax(logits_adv, dim=1),
                    F.softmax(logits_nat, dim=1)
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif distance == 'l2':
        delta = 0.001 * torch.randn_like(x_natural)
        delta = Variable(delta, requires_grad=True)
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta
            optimizer_delta.zero_grad()
            logits_adv = model(adv)
            logits_nat = model(x_natural)
            loss = -criterion_kl(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_nat, dim=1)
            )
            loss.backward()

            grad = delta.grad
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            normalized_grad = grad / (grad_norm + 1e-8)
            delta.data = delta + step_size * normalized_grad
            delta.data = torch.clamp(x_natural + delta.data, 0.0, 1.0) - x_natural
            delta.data = delta.renorm(p=2, dim=0, maxnorm=epsilon)

        x_adv = x_natural + delta.detach()
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")

    model.train()

    # Natural loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)

    # Robust loss (KL divergence)
    logits_adv = model(x_adv)
    loss_robust = criterion_kl(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits, dim=1)
    )

    return loss_natural + beta * loss_robust
