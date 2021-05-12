from typing import List, Tuple

import numpy as np
from kaggle_shopee.utils.test_util import TestUtil


class MetricUtil:
    @staticmethod
    def f1_scores(y_true: List[List[str]], y_pred: List[List[str]]) -> List[float]:
        TestUtil.assert_any(len(y_true), len(y_pred))
        scores: List[float] = []
        for i in range(len(y_true)):
            intersect_n = len(np.intersect1d(y_true[i], y_pred[i]))
            score = 2 * intersect_n / (len(y_true[i]) + len(y_pred[i]))
            scores.append(score)
        return scores

    @staticmethod
    def precision_recall(
        y_true: List[List[str]], y_pred: List[List[str]]
    ) -> Tuple[List[float], List[float]]:
        TestUtil.assert_any(len(y_true), len(y_pred))
        precisions: List[float] = []
        recalls: List[float] = []
        for i in range(len(y_true)):
            precisions.append(np.mean([t in y_true[i] for t in y_pred[i]]))
            recalls.append(np.mean([t in y_pred[i] for t in y_true[i]]))
        return precisions, recalls
