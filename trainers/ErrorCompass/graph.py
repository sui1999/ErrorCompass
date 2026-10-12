"""Class-level SCGP graph operations from the ErrorCompass manuscript.

All graphs have one node per class. Semantic graph construction retains the
original cosine self-similarities; the propagation layer then adds exactly one
identity matrix, as specified by the manuscript's propagation equation.
"""

import math
import random

import torch
from torch import nn
from torch.nn import functional as F


def _validate_matrix(value, name, square=False):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or min(value.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if square and value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def semantic_adjacency(text_anchors, threshold=None):
    """Return row-normalized semantic adjacency for C-by-d text anchors.

    A single threshold is shared by all rows and sampled from U(0.5, 0.6)
    when omitted. An explicit threshold in [0, 1] enables deterministic
    evaluation/testing. Zero anchors produce zero rows without NaNs. No
    additional self-loop is inserted during semantic graph construction.
    """
    _validate_matrix(text_anchors, "text_anchors")
    if threshold is None:
        threshold = random.uniform(0.5, 0.6)
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")

    eps = torch.finfo(text_anchors.dtype).tiny
    anchors = F.normalize(text_anchors, p=2, dim=1, eps=eps)
    similarities = anchors @ anchors.T
    row_max = similarities.max(dim=1, keepdim=True).values
    retained = torch.where(
        similarities >= threshold * row_max,
        similarities,
        torch.zeros_like(similarities),
    )
    return retained / retained.sum(dim=1, keepdim=True).clamp_min(eps)


def context_adjacency(Asem, Aconf, semantic_weight, feedback_active):
    """Mix semantic adjacency and the shared, already prepared confusion graph.

    Aconf is used at its original scale: row-normalizing it again would erase
    the magnitude of errors relative to correct predictions. Before feedback
    activation (and during transfer), return Asem directly; Aconf may be None.
    """
    _validate_matrix(Asem, "Asem", square=True)
    weight = float(semantic_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("semantic_weight must be finite and in [0, 1]")
    if not feedback_active:
        return Asem
    _validate_matrix(Aconf, "Aconf", square=True)
    if Asem.shape != Aconf.shape:
        raise ValueError("Asem and Aconf must have the same class dimensions")
    if Asem.device != Aconf.device:
        raise ValueError("Asem and Aconf must be on the same device")
    return weight * Asem + (1.0 - weight) * Aconf


class GraphConvolution(nn.Module):
    """One class-level SCGP layer: LeakyReLU(D^-1/2 (A+I) D^-1/2 P W).

    The only learned parameter is ``propagation_weight`` (d by d). Stacking
    layers is the caller's responsibility; each call returns all C prototypes
    without adding prompt nodes, a bias, dropout, or an extra aggregation.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim
        self.propagation_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.hidden_dim)
        nn.init.uniform_(self.propagation_weight, -bound, bound)

    def forward(self, class_features, adjacency):
        _validate_matrix(class_features, "class_features")
        _validate_matrix(adjacency, "adjacency", square=True)
        if class_features.shape[1] != self.hidden_dim:
            raise ValueError("class_features width must equal hidden_dim")
        if adjacency.shape[0] != class_features.shape[0]:
            raise ValueError("adjacency must have one node per class prototype")
        if adjacency.device != class_features.device:
            raise ValueError("adjacency and class_features must be on the same device")
        if torch.any(adjacency < 0):
            raise ValueError("adjacency must be nonnegative")

        # Keep adjacency arithmetic in its native dtype, including under AMP.
        with_self = adjacency + torch.eye(
            adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
        )
        inverse_sqrt_degree = with_self.sum(dim=1).rsqrt()
        normalized = (
            inverse_sqrt_degree[:, None]
            * with_self
            * inverse_sqrt_degree[None, :]
        )
        transformed = class_features @ self.propagation_weight
        # Without autocast, mixed feature/graph dtypes also need compatible
        # operands. The conversion preserves autograd to a learned adjacency.
        propagated = normalized.to(dtype=transformed.dtype) @ transformed
        return F.leaky_relu(propagated, negative_slope=0.2)
