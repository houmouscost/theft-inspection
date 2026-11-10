"""Detektor gradient boosting + kalibrasi Platt (monoton ketat -> peringkat terjaga)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .features import columns


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Detector:
    def __init__(self, name: str, use_level: bool, use_missing: bool, params: dict, seed: int = 0,
                 use_change: bool = False):
        self.name = name
        self.cols = columns(use_level, use_missing, use_change)
        self.params = params
        self.seed = seed
        self.clf = None
        self.platt = None

    def fit(self, F: np.ndarray, y: np.ndarray, F_cal: np.ndarray, y_cal: np.ndarray):
        p = self.params
        self.clf = HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 300), learning_rate=p.get("learning_rate", 0.08),
            max_leaf_nodes=p.get("max_leaf_nodes", 31), l2_regularization=p.get("l2", 1.0),
            class_weight="balanced", early_stopping=False, random_state=self.seed)
        self.clf.fit(F[:, self.cols], y)
        raw = self.clf.predict_proba(F_cal[:, self.cols])[:, 1]
        self.platt = LogisticRegression(C=1e6, max_iter=1000)
        if len(np.unique(y_cal)) == 2:
            self.platt.fit(_logit(raw).reshape(-1, 1), y_cal)
        else:  # degenerate: identitas
            self.platt = None
        return self

    uses_series = False

    def score_rows(self, F_rows: np.ndarray, X_rows=None) -> np.ndarray:
        """Antarmuka seragam dengan detektor deret (deep.SeriesDetector): GBM memakai fitur."""
        return self.score(F_rows)

    def score(self, F: np.ndarray) -> np.ndarray:
        raw = self.clf.predict_proba(F[:, self.cols])[:, 1]
        if self.platt is None:
            return raw
        return self.platt.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
