import torch
import torch.nn.functional as F


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute KL divergence between two batches of probability distributions.

    Args:
        p (torch.Tensor): Probabilities, shape (batch_size, num_classes)
        q (torch.Tensor): Probabilities, shape (batch_size, num_classes)
        eps (float): Small value to avoid log(0)

    Returns:
        torch.Tensor: KL divergence for each batch item, shape (batch_size,)
    """
    is_batched = p.ndim == 2
    if not is_batched:
        p = p.view(1, -1)
        q = q.view(1, -1)

    kl_div = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=1)
    return kl_div if is_batched else kl_div.view(1)


def missingness_bias(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute the missingness bias between two batches of probability distributions.

    Args:
        p (torch.Tensor): Probabilities, shape (batch_size, num_classes)
        q (torch.Tensor): Probabilities, shape (batch_size, num_classes)
    """
    _, num_classes = p.shape
    assert num_classes == q.shape[1]
    p_dist = F.one_hot(p.argmax(dim=1), num_classes=num_classes).float().mean(dim=0)
    q_dist = F.one_hot(q.argmax(dim=1), num_classes=num_classes).float().mean(dim=0)
    return kl_divergence(p_dist, q_dist)


