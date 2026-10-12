"""Torch-only feedback and loss formulas used by the ErrorCompass manuscript.

The helpers contain no trainable parameters. Historical confusion, text anchors,
and teacher predictions are fixed targets; gradients only reach the student's
features or logits. Callers apply the configured loss weights and warm-up rules.
"""

import torch
from torch.nn import functional as F


@torch.no_grad()
def confusion_adjacency(counts):
    """Row-normalize all counts, remove self-edges, then symmetrize.

    Correct predictions remain in the row denominator. An unobserved row stays
    zero before symmetrization, without introducing artificial epsilon edges.
    """
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("counts must be a square class-by-class matrix")
    row_sum = counts.sum(dim=1, keepdim=True)
    denominator = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    adjacency = counts / denominator
    adjacency.fill_diagonal_(0)
    return 0.5 * (adjacency + adjacency.t())


@torch.no_grad()
def update_confusion_counts(counts, logits, labels, rho):
    """Return EMA hard-prediction counts with rows=true, columns=predicted.

    The update includes correct predictions and never changes ``counts`` in
    place. Unlike predicted probabilities, one argmax contributes one count.
    """
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("counts must be a square class-by-class matrix")
    n_classes = counts.shape[0]
    if logits.ndim != 2 or logits.shape[1] != n_classes:
        raise ValueError("logits must have one column per class")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must contain one class index per sample")
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    predictions = logits.argmax(dim=1).to(device=counts.device)
    labels = labels.to(device=counts.device, dtype=torch.long)
    indices = labels * n_classes + predictions
    increment = torch.bincount(indices, minlength=n_classes * n_classes)
    increment = increment.reshape_as(counts).to(dtype=counts.dtype)
    return rho * counts + (1 - rho) * increment


def sccm_loss(prompt_features, text_anchors):
    """Mean over classes of squared L2 distances before prototype fusion.

    Inputs are the already normalized prompt features and frozen text anchors
    with shape [classes, embedding dimension]. No embedding-dimension averaging
    is performed: each class contributes its full squared distance.
    """
    if prompt_features.ndim != 2 or prompt_features.shape != text_anchors.shape:
        raise ValueError("prompt features and anchors must share shape [C, d]")
    if prompt_features.shape[0] == 0:
        return prompt_features.sum() * 0
    return (prompt_features - text_anchors.detach()).square().sum(dim=1).mean()


def chc_loss(logits, labels, top_k, beta, confusion=None, eta=0.25):
    """Weighted hard-negative contrast in log space, retaining weight gradients.

    ``logits`` are the final classifier scores (the default temperature is one).
    Confusion is a fixed historical state; selected logits and their normalized
    weights both remain differentiable. Saturated top-k selects all negatives.
    """
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError('Expected logits [B,C] and labels [B].')
    k = min(int(top_k), logits.shape[1] - 1)
    if k < 1 or logits.shape[0] == 0:
        return logits.sum() * 0.0
    positive = logits.gather(1, labels[:, None])
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, labels[:, None], True)
    negative, indices = logits.masked_fill(mask, -torch.inf).topk(k, dim=1)
    log_weights = F.log_softmax(beta * negative, dim=1)
    if confusion is not None:
        if confusion.shape != (logits.shape[1], logits.shape[1]):
            raise ValueError('Confusion must have one row and column per class.')
        prior = confusion.detach().to(device=logits.device, dtype=logits.dtype)[labels[:, None], indices]
        log_weights = log_weights + torch.log1p(eta * prior)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
    denominator = torch.logsumexp(torch.cat((positive, negative + log_weights), dim=1), dim=1)
    return (denominator - positive[:, 0]).mean()


def cpmc_loss(prototypes, Aconf, top_k, threshold, margin_base, margin_scale):
    """Confusion-weighted cosine-margin hinge loss on class prototypes.

    Rank only genuine unordered class pairs (i < j), keep at most ``top_k``,
    then require confusion strictly above ``threshold``. The mean denominator
    includes every selected pair, including those already meeting the margin.
    There is no additional learned temperature multiplier in this formula.
    """
    if prototypes.ndim != 2:
        raise ValueError("prototypes must have shape [C, d]")
    n_classes = prototypes.shape[0]
    if Aconf.shape != (n_classes, n_classes):
        raise ValueError("Aconf must match the prototype class count")
    if n_classes < 2 or top_k <= 0:
        return prototypes.sum() * 0

    rows, columns = torch.triu_indices(n_classes, n_classes, offset=1,
                                      device=prototypes.device)
    pair_confusion = Aconf.detach().to(device=prototypes.device,
                                     dtype=prototypes.dtype)[rows, columns]
    selected_confusion, selected = pair_confusion.topk(min(top_k, rows.numel()))
    valid = selected_confusion > threshold
    if not valid.any():
        return prototypes.sum() * 0
    selected = selected[valid]
    weights = selected_confusion[valid]
    # A nonzero epsilon also handles a temporarily zero graph output in fp16.
    normalized = F.normalize(prototypes, dim=-1, eps=1e-6)
    cosine = (normalized[rows[selected]] * normalized[columns[selected]]).sum(-1)
    distances = 1 - cosine.clamp(-1, 1)
    margins = margin_base + margin_scale * weights
    return (weights * F.relu(margins - distances)).mean()


def kl_teacher_loss(student_logits, teacher_logits):
    """Batch-mean KL(q_teacher || p_student), with detached teacher logits."""
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must share shape [B, C]")
    if student_logits.shape[0] == 0:
        return student_logits.sum() * 0
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.log_softmax(teacher_logits.detach(), dim=1),
        reduction="batchmean",
        log_target=True,
    )
