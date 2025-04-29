import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def trades_loss(model,
               x_natural,
               y,
               optimizer,
               step_size=0.02,
               epsilon=0.08,
               perturb_steps=10,
               beta=1.0,
               distance='l_inf'):
    """
    Compute the TRADES loss.

    Args:
        model: the classifier model.
        x_natural: input examples.
        y: labels.
        step_size: step size for PGD.
        epsilon: maximum perturbation.
        perturb_steps: number of PGD steps.
        beta: trade-off parameter between natural and robust loss.
        distance: distance metric ('l_inf' or 'l_2').

    Returns:
        loss: the computed TRADES loss.
    """
    # Define KL loss
    if distance == 'l_inf':
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
    elif distance == 'l2':
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError("Unsupported distance metric.")

    model.eval()
    batch_size = len(x_natural)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_adv,_ = model(x_adv)
                logits_nat,_ = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1),
                                       F.softmax(logits_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l2':
        delta = 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizer for delta
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # Optimize delta to maximize the KL divergence
            optimizer_delta.zero_grad()
            logits_adv,_ = model(adv)
            logits_nat,_ = model(x_natural)
            loss = -criterion_kl(F.log_softmax(logits_adv, dim=1),
                                 F.softmax(logits_nat, dim=1))
            loss.backward()

            # Normalize gradients
            grad = delta.grad
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1)
            grad = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
            delta.data = delta + step_size * grad.sign()
            delta.data = torch.clamp(x_natural + delta, 0.0, 1.0) - x_natural
            delta.data = delta.renorm(p=2, dim=0, maxnorm=epsilon)

        x_adv = x_natural + delta.detach()

    else:
        raise ValueError("Unsupported distance metric.")

    model.train()

    # Compute the TRADES loss
    logits,_ = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    logits_adv,_ = model(x_adv)
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1),
                               F.softmax(logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss
