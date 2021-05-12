import math
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from kaggle_shopee.factories.config_factory import LossConfig


def log_t(u, t):
    """Compute log_t for `u'."""
    if t == 1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t == 1:
        return u.exp()
    else:
        return (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters):

    """Returns the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0

    for _ in range(num_iters):
        logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * logt_partition.pow(
            1.0 - t
        )

    logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = -log_t(1.0 / logt_partition, t) + mu

    return normalization_constants


def compute_normalization_binary_search(activations, t, num_iters):

    """Returns the normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu

    effective_dim = torch.sum(
        (normalized_activations > -1.0 / (1.0 - t)).to(torch.int32),
        dim=-1,
        keepdim=True,
    ).to(activations.dtype)

    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(
        shape_partition, dtype=activations.dtype, device=activations.device
    )
    upper = -log_t(1.0 / effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower) / 2.0
        sum_probs = torch.sum(
            exp_t(normalized_activations - logt_partition, t), dim=-1, keepdim=True
        )
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(
            lower * update + (1.0 - update) * logt_partition, shape_partition
        )
        upper = torch.reshape(
            upper * (1.0 - update) + update * logt_partition, shape_partition
        )

    logt_partition = (upper + lower) / 2.0
    return logt_partition + mu


class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """

    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(
                activations, t, num_iters
            )
        else:
            normalization_constants = compute_normalization_fixed_point(
                activations, t, num_iters
            )

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output

        return grad_input, None, None


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)


def tempered_sigmoid(activations, t, num_iters=5):
    """Tempered sigmoid function.
    Args:
      activations: Activations for the positive class for binary classification.
      t: Temperature tensor > 0.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    internal_activations = torch.stack(
        [activations, torch.zeros_like(activations)], dim=-1
    )
    internal_probabilities = tempered_softmax(internal_activations, t, num_iters)
    return internal_probabilities[..., 0]


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)


def bi_tempered_binary_logistic_loss(
    activations, labels, t1, t2, label_smoothing=0.0, num_iters=5, reduction="mean"
):

    """Bi-Tempered binary logistic loss.
    Args:
      activations: A tensor containing activations for class 1.
      labels: A tensor with shape as activations, containing probabilities for class 1
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing
      num_iters: Number of iterations to run the method.
    Returns:
      A loss tensor.
    """
    internal_activations = torch.stack(
        [activations, torch.zeros_like(activations)], dim=-1
    )
    internal_labels = torch.stack(
        [labels.to(activations.dtype), 1.0 - labels.to(activations.dtype)], dim=-1
    )
    return bi_tempered_logistic_loss(
        internal_activations,
        internal_labels,
        t1,
        t2,
        label_smoothing=label_smoothing,
        num_iters=num_iters,
        reduction=reduction,
    )


def bi_tempered_logistic_loss(
    activations,
    labels,
    t1,
    t2,
    label_smoothing=0.0,
    num_iters=5,
    reduction="mean",
    label_smoothing_p: float = 1.0,
):

    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot),
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """

    if len(labels.shape) < len(activations.shape):  # not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if (
        label_smoothing > 0
        and np.random.uniform(0.0, 1.0, size=1)[0] < label_smoothing_p
    ):
        num_classes = labels_onehot.shape[-1]
        labels_onehot = (
            1 - label_smoothing * num_classes / (num_classes - 1)
        ) * labels_onehot + label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = (
        labels_onehot * log_t(labels_onehot + 1e-10, t1)
        - labels_onehot * log_t(probabilities, t1)
        - labels_onehot.pow(2.0 - t1) / (2.0 - t1)
        + probabilities.pow(2.0 - t1) / (2.0 - t1)
    )
    loss_values = loss_values.sum(dim=-1)  # sum over classes

    if reduction == "none":
        return loss_values
    if reduction == "sum":
        return loss_values.sum()
    if reduction == "mean":
        return loss_values.mean()


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))


class LabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        n_classes: int = 5,
        smoothing: float = 0.0,
        smoothing_p: float = 1.0,
        dim: int = -1,
    ):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.smoothing_p = smoothing_p
        self.n_classes = n_classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            if np.random.uniform(0.0, 1.0, size=1)[0] < self.smoothing_p:
                true_dist.fill_(self.smoothing / (self.n_classes - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            else:
                true_dist.scatter_(1, target.data.unsqueeze(1), 1.0)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, n_dim, ls_=0.9):
        super().__init__()
        self.n_dim = n_dim
        self.ls_ = ls_

    def forward(self, x, target):
        target = F.one_hot(target, self.n_dim).float()
        target *= self.ls_
        target += (1 - self.ls_) / self.n_dim

        logprobs = F.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        logp = self.ce(pred, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class FocalCosineLoss(nn.Module):
    def __init__(
        self, alpha: int = 1, gamma: int = 2, xent: int = 1, reduction: str = "mean"
    ):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.y = torch.Tensor([1])
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        cosine_loss = F.cosine_embedding_loss(
            pred,
            F.one_hot(target, num_classes=pred.size(-1)),
            self.y,
            reduction=self.reduction,
        )
        cent_loss = F.cross_entropy(F.normalize(pred), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


class SymmetricCrossEntropy(nn.Module):
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        n_classes: int = 5,
        reduction: str = "mean",
    ):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_classes = n_classes
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        onehot_target = torch.eye(self.n_classes)[target]
        ce_loss = F.cross_entropy(pred, target, reduction=self.reduction)
        rce_loss = (-onehot_target * pred.softmax(1).clamp(1e-7, 1.0).log()).sum(1)
        if self.reduction == "mean":
            rce_loss = rce_loss.mean()
        elif self.reduction == "sum":
            rce_loss = rce_loss.sum()
        return self.alpha * ce_loss + self.beta * rce_loss


class BiTemperedLogisticLoss(nn.Module):
    def __init__(
        self,
        t1: float,
        t2: float,
        smoothing: float = 0.0,
        smoothing_p: float = 1.0,
        reduction: str = "none",
    ):
        super(BiTemperedLogisticLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.smoothing_p = smoothing_p
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss = bi_tempered_logistic_loss(
            pred,
            target,
            t1=self.t1,
            t2=self.t2,
            label_smoothing=self.smoothing,
            label_smoothing_p=self.smoothing_p,
            reduction=self.reduction,
        )
        loss = loss.mean()
        return loss


class TaylorSoftmax(nn.Module):
    def __init__(self, dim: int = 1, n: int = 2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.0
        for i in range(1, self.n + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdim=True)
        return out


class TaylorCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n: int = 2,
        ignore_index: int = -1,
        reduction: str = "mean",
        smoothing: float = 0.05,
    ):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing_loss = LabelSmoothingLoss(n_classes, smoothing)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        log_probs = self.taylor_softmax(pred).log()
        loss = self.label_smoothing_loss(log_probs, target)
        return loss


class LossFactory:
    @staticmethod
    def get_loss(config: LossConfig, **args):
        losses = {
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "CrossEntropyLossOneHot": CrossEntropyLossOneHot,
            "LabelSmoothingLoss": LabelSmoothingLoss,
            "CrossEntropyLossWithLabelSmoothing": CrossEntropyLossWithLabelSmoothing,
            "FocalLoss": FocalLoss,
            "FocalCosineLoss": FocalCosineLoss,
            "SymmetricCrossEntropy": SymmetricCrossEntropy,
            "BiTemperedLogisticLoss": BiTemperedLogisticLoss,
            "TaylorCrossEntropyLoss": TaylorCrossEntropyLoss,
            "KLDivLoss": nn.KLDivLoss,
        }
        assert config.name in losses, "loss_name not in {}".format(list(losses.keys()))
        return losses[config.name](**config.params, **args)
