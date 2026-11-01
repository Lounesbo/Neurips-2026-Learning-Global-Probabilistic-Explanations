# -----------------------------------------------------------------------------
#
# Datasets
#
# Every attribute becomes one binary feature in Omega_i = {0, 1}, so that c = 2
# and d is the number of retained attributes. The binary feature keeps the name
# of a readable predicate:
#
#   numerical attribute with more than two values  ->  "age>37"  (median split)
#   any attribute with exactly two values          ->  "sex=Male" (one of them)
#   categorical attribute with more values          ->  "workclass=Private" (mode)
#
# As in AIJ, rows with a missing target are dropped, then every attribute
# holding a missing value is dropped: no imputation, so the rule fits in one
# sentence. No feature selection either, which would bring in a second model.
#
# D is the product of the empirical marginals of the whole dataset, smoothed by
# Laplace (counts + alpha) so that it has full support. The classifier is
# trained on a stratified split and its test accuracy is reported.
#
# -----------------------------------------------------------------------------

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from distribution import ProductDistribution
from utils import Console


class Dataset:
    OPENML = {
        13: "Breast Cancer",
        15: "Breast Cancer (WI)",
        24: "Mushroom",
        27: "Colic",
        29: "Credit Approval",
        31: "Credit-G",
        37: "Pima Indians Diabetes",
        44: "Spambase",
        50: "Tic-Tac-Toe",
        55: "Hepatitis",
        59: "Ionosphere",
        179: "Adult",
        335: "Monks 3",
        1116: "Musk",
        1461: "Bank Marketing",
        1480: "Indian Liver Patients",
        1486: "Nomao",
        40536: "Speed Dating",
        40683: "Postoperative",
        40981: "Australian",
        42192: "COMPAS",
        43672: "Heart Disease",
    }

    # Only the ARFF marker '?' and the empty string mean "missing". Words such
    # as 'none' or 'na' are genuine category values in several datasets (e.g.
    # other_parties = none in Credit-G), and must not be read as missing.
    MISSING = ['?', '']

    def __init__(self, id: int = -1, name: str = "", seed: int = 48, test_size: float = 0.2,
                 alpha: float = 1.0, device: str = "cpu", console: Optional[Console] = None):
        self.id = id
        self.name = name or self.OPENML.get(id, f"Dataset {id}")
        self.seed = seed
        self.test_size = test_size
        self.alpha = alpha
        self.device = device
        self.console = console or Console(verbose=False)

        self.names: List[str] = []
        self.X = np.zeros((0, 0), dtype=np.uint8)
        self.Y = np.zeros(0, dtype=np.int8)
        self.X_train = self.X_test = self.X
        self.Y_train = self.Y_test = self.Y
        self.distribution: Optional[ProductDistribution] = None

        self.n_raw_attributes = 0
        self.n_missing_values = 0
        self.n_dropped_incomplete = 0
        self.n_dropped_constant = 0
        self.n_categorical = 0
        self.n_numerical = 0

    @property
    def d(self) -> int:
        return self.X.shape[1]

    @property
    def n_instances(self) -> int:
        return self.X.shape[0]

    def summary(self) -> dict:
        return {
            "dataset": self.name, "openml_id": self.id, "d": self.d, "n_instances": self.n_instances,
            "raw_attributes": self.n_raw_attributes, "missing_values": self.n_missing_values,
            "categorical": self.n_categorical,
            "numerical": self.n_numerical, "dropped_incomplete": self.n_dropped_incomplete,
            "dropped_constant": self.n_dropped_constant,
        }

    # -- Pipeline --------------------------------------------------------------

    def build(self) -> "Dataset":
        from sklearn.datasets import fetch_openml
        self.console.log("Loading dataset", self.name)
        # Always the pure Python ARFF parser. The pandas parser of scikit-learn
        # strips quotes from category labels with a regular expression, which
        # fails when recent pandas infers integer labels for nominal attributes
        # such as {0, 1}; on failure, scikit-learn then deletes its cached file
        # to download it again, which Windows refuses while another process
        # (the other GPU shard) is reading it. liac-arff has neither problem,
        # and several processes can read the same cached file at once.
        X, y = fetch_openml(data_id=self.id, parser="liac-arff", as_frame=True, return_X_y=True)
        return self.build_from_frame(X, y)

    def build_from_frame(self, raw_X: pd.DataFrame, raw_y: pd.Series) -> "Dataset":
        raw_X, raw_y = self._clean(raw_X, raw_y)
        self._binarize(raw_X)
        if self.d == 0:
            raise ValueError(f"{self.name}: no attribute left after preprocessing "
                             f"({self.n_dropped_incomplete} incomplete, {self.n_dropped_constant} constant)")
        self._encode_labels(raw_y)
        self._split()
        self.distribution = ProductDistribution.from_data(
            self.X, cards=[2] * self.d, alpha=self.alpha, names=self.names, device=self.device
        )
        self.console.log("Binary dimension (d)", self.d)
        return self

    def _clean(self, raw_X: pd.DataFrame, raw_y: pd.Series):
        self.n_raw_attributes = raw_X.shape[1]
        raw_X = raw_X.replace(self.MISSING, np.nan)
        keep = raw_y.notna()
        raw_X, raw_y = raw_X.loc[keep], raw_y.loc[keep]
        self.n_missing_values = int(raw_X.isna().values.sum())
        incomplete = raw_X.columns[raw_X.isna().any()]
        self.n_dropped_incomplete = len(incomplete)
        return raw_X.drop(columns=incomplete), raw_y

    def _binarize(self, raw_X: pd.DataFrame) -> None:
        columns, names = [], []
        self.n_categorical = self.n_numerical = self.n_dropped_constant = 0
        for col in raw_X.columns:
            series = raw_X[col]
            values = series.dropna().unique()
            if len(values) < 2:
                self.n_dropped_constant += 1
                continue
            numeric = pd.api.types.is_numeric_dtype(series)
            if len(values) == 2:
                # One of the two values, the larger for numbers, the mode otherwise.
                target = max(values) if numeric else series.mode().iloc[0]
                bit, name = (series == target), f"{col}={target}"
            elif numeric:
                median = float(series.median())
                bit, name = (series > median), f"{col}>{median:g}"
                if bit.nunique() < 2:  # mass at the maximum: split below it instead
                    bit, name = (series >= median), f"{col}>={median:g}"
            else:
                mode = series.astype(str).mode().iloc[0]
                bit, name = (series.astype(str) == mode), f"{col}={mode}"
            if bit.nunique() < 2:
                self.n_dropped_constant += 1
                continue
            self.n_numerical += int(numeric)
            self.n_categorical += int(not numeric)
            columns.append(bit.to_numpy(dtype=np.uint8))
            names.append(name)
        self.X = np.stack(columns, axis=1) if columns else np.zeros((len(raw_X), 0), dtype=np.uint8)
        self.names = names

    def _encode_labels(self, raw_y: pd.Series) -> None:
        if raw_y.nunique() != 2:
            raise ValueError(f"{self.name} is not a binary classification task")
        majority = raw_y.value_counts().idxmax()
        self.Y = np.where(raw_y.to_numpy() == majority, 1, -1).astype(np.int8)

    def _split(self) -> None:
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=self.test_size, random_state=self.seed, stratify=self.Y
        )
