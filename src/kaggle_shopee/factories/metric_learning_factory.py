import math
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from kaggle_shopee.factories.config_factory import MetricLearningConfig


class AdaCos(nn.Module):
    def __init__(
        self, in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi / 4
    ):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_zero = theta_zero
        self.s = math.log(out_features - 1) / math.cos(theta_zero)
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input).float()
        # normalize weights
        W = F.normalize(self.weight).float()
        # dot product
        logits = F.linear(x, W).float()
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7)).float()
        target_logits = torch.cos(theta + self.m).float()
        one_hot = torch.zeros_like(logits).float()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.where(
                one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits)
            )
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(B_avg) / torch.cos(
                torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med)
            )
        output *= self.s

        return output


class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.5,
        easy_margin: bool = False,
        smoothing: float = 0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.smoothing = smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight)).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=labels.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        if self.smoothing > 0:
            one_hot = (
                1 - self.smoothing
            ) * one_hot + self.smoothing / self.out_features

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArcAdaptiveMarginProduct(nn.Module):
    """Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        train_df: pd.DataFrame,
        a: float = 0.45,
        b: float = 0.05,
        n_sqrt: int = 2,
        s: float = 30.0,
        easy_margin: bool = False,
        smoothing: float = 0.0,
    ):
        super(ArcAdaptiveMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margins = ArcAdaptiveMarginProduct._get_margins(train_df, a, b, n_sqrt)
        self.s = s
        self.smoothing = smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.easy_margin = easy_margin
        nn.init.xavier_uniform_(self.weight)

    @staticmethod
    def _get_margins(
        train_df: pd.DataFrame, a: float, b: float, n_sqrt: int
    ) -> Dict[int, float]:
        tmp = train_df["label_group_le"].value_counts().to_dict()

        for _ in range(n_sqrt):
            tmp = {k: np.sqrt(v) for k, v in tmp.items()}
        tmp = {k: 1 / v for k, v in tmp.items()}

        _min = np.min(list(tmp.values()))
        _max = np.max(list(tmp.values()))

        tmp = {k: (v - _min) / (_max - _min) for k, v in tmp.items()}
        tmp = {k: a * v + b for k, v in tmp.items()}
        return tmp

    def forward(self, inputs, labels):
        ms = np.array([self.margins[_label] for _label in labels.cpu().numpy()])
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight)).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=labels.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        if self.smoothing > 0:
            one_hot = (
                1 - self.smoothing
            ) * one_hot + self.smoothing / self.out_features

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """

    def __init__(self, in_features: int, out_features: int, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(
            self.LambdaMin,
            self.base * (1 + self.gamma * self.iter) ** (-1 * self.power),
        )

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight)).float()
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", m="
            + str(self.m)
            + ")"
        )


class ArcMarginProduct2(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
      in_features: size of each input sample
      out_features: size of each output sample
      s: norm of input feature
      m: margin
      cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.5,
        easy_margin: bool = False,
        smoothing: float = 0.0,
    ):
        super(ArcMarginProduct2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.smoothing = smoothing  # label smoothing
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.W))
        if label is None:
            return cosine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine.float() > 0, phi, cosine.float())
        else:
            phi = torch.where(cosine.float() > self.th, phi, cosine.float() - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.smoothing > 0:
            one_hot = (
                1 - self.smoothing
            ) * one_hot + self.smoothing / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArcMarginProductSubCenter(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        m: float = 0.5,
        s: float = 30.0,
        k=3,
        easy_margin: bool = False,
        smoothing: float = 0.0,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.m = m
        self.s = s
        self.k = k
        self.smoothing = smoothing
        self.out_features = out_features
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features, labels):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        cosine = cosine.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        if self.smoothing > 0:
            one_hot = (
                1 - self.smoothing
            ) * one_hot + self.smoothing / self.out_features

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return cosine


class AddMarginProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(
        self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.4
    ):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, labels):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight)).float()
        phi = cosine - self.m

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", s="
            + str(self.s)
            + ", m="
            + str(self.m)
            + ")"
        )


class ArcMarginProductSubCenter2(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


products = {
    "AdaCos": AdaCos,
    "ArcMarginProduct": ArcMarginProduct,
    "ArcMarginProduct2": ArcMarginProduct2,
    "ArcAdaptiveMarginProduct": ArcAdaptiveMarginProduct,
    "AddMarginProduct": AddMarginProduct,
    "ArcMarginProductSubCenter": ArcMarginProductSubCenter,
    "ArcMarginProductSubCenter2": ArcMarginProductSubCenter2,
    "SphereProduct": SphereProduct,
}


class MetricLearningFactory:
    @staticmethod
    def get_metric_learning_product(
        met_config: MetricLearningConfig, in_features: int, out_features: int, **args
    ):
        if met_config.name not in products.keys():
            raise ValueError(
                "met_config.name not in the keys: {}".format(list(products.keys()))
            )
        return products[met_config.name](
            in_features=in_features,
            out_features=out_features,
            **met_config.params,
            **args
        )


class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """

    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(
                valid_positive_dist, dim=1, keepdim=True
            )

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                1.0 - mask_anchor_negative
            )
            hardest_negative_dist, _ = torch.min(
                anchor_negative_dist, dim=1, keepdim=True
            )

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + 0.1)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1

    return mask


def _get_triplet_mask(labels):
    """return a 3d mask where mask[a, p, n] is true if the triplet (a, p, n) is valid.
    a triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)

    mask = distinct_indices * valid_labels  # Combine the two masks

    return mask
