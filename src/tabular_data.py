"""Utility helpers for preparing tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TabularMetadata:
    """Stores information about the processed tabular dataset."""

    continuous_columns: Sequence[str]
    categorical_columns: Sequence[str]
    categorical_cardinalities: Sequence[int]


class TabularPreprocessor:
    """Normalize numeric columns and encode categoricals."""

    def __init__(self, continuous_columns: Sequence[str], categorical_columns: Sequence[str]):
        self.continuous_columns = list(continuous_columns)
        self.categorical_columns = list(categorical_columns)
        self._continuous_stats: Dict[str, Tuple[float, float]] = {}
        self._categorical_maps: Dict[str, Dict[str, int]] = {}

    @staticmethod
    def _stable_unique(values: Iterable[str]) -> List[str]:
        seen: Dict[str, None] = {}
        for value in values:
            if value not in seen:
                seen[value] = None
        return list(seen.keys())

    def fit(self, frame: pd.DataFrame) -> None:
        """Collect statistics required for transformation."""

        for column in self.continuous_columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            mean = float(series.mean()) if not series.empty else 0.0
            std = float(series.std()) if not series.empty else 0.0
            if np.isnan(mean):
                mean = 0.0
            if np.isnan(std) or std == 0.0:
                std = 1.0
            self._continuous_stats[column] = (mean, std)

        for column in self.categorical_columns:
            series = frame[column].astype(str).fillna("__missing__")
            categories = self._stable_unique(series.tolist())
            if "__unknown__" in categories:
                categories.remove("__unknown__")
            categories.append("__unknown__")
            mapping = {category: idx for idx, category in enumerate(categories)}
            self._categorical_maps[column] = mapping

    def transform(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Apply normalization/encoding to a dataframe."""

        cont_arrays: List[np.ndarray] = []
        for column in self.continuous_columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            mean, std = self._continuous_stats[column]
            series = series.fillna(mean)
            normalized = (series - mean) / std
            cont_arrays.append(normalized.to_numpy(dtype=np.float32))

        if cont_arrays:
            cont_array = np.stack(cont_arrays, axis=1)
        else:
            cont_array = np.empty((len(frame), 0), dtype=np.float32)

        cat_arrays: List[np.ndarray] = []
        for column in self.categorical_columns:
            series = frame[column].astype(str).fillna("__missing__")
            mapping = self._categorical_maps[column]
            unknown_index = mapping["__unknown__"]
            encoded = np.array([mapping.get(value, unknown_index) for value in series], dtype=np.int64)
            cat_arrays.append(encoded)

        if cat_arrays:
            cat_array = np.stack(cat_arrays, axis=1)
        else:
            cat_array = np.empty((len(frame), 0), dtype=np.int64)

        return cont_array, cat_array

    @property
    def categorical_cardinalities(self) -> List[int]:
        return [len(self._categorical_maps[column]) for column in self.categorical_columns]


class TabularDataset(Dataset):
    """A simple ``Dataset`` wrapper around processed arrays."""

    def __init__(self, continuous: np.ndarray, categorical: np.ndarray, targets: np.ndarray):
        if continuous.ndim != 2:
            raise ValueError("continuous must be a 2D array")
        if categorical.ndim != 2:
            raise ValueError("categorical must be a 2D array")
        if continuous.shape[0] != categorical.shape[0] or continuous.shape[0] != len(targets):
            raise ValueError("continuous, categorical and targets must contain the same number of samples")

        self.continuous = torch.as_tensor(continuous, dtype=torch.float32)
        self.categorical = torch.as_tensor(categorical, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.long)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.targets.size(0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.continuous[index], self.categorical[index], self.targets[index]


def merge_tables(file_paths: Sequence[str], label_column: str, target_column: str) -> pd.DataFrame:
    """Merge multiple CSV tables into a single dataframe."""

    if not file_paths:
        raise ValueError("Expected at least one CSV file")

    dataframes = [pd.read_csv(path) for path in file_paths]
    base = dataframes[0]
    merge_keys = [label_column, target_column]
    for frame in dataframes[1:]:
        base = base.merge(frame, on=merge_keys, how="inner", suffixes=("", "_dup"))
        duplicated = [column for column in base.columns if column.endswith("_dup")]
        if duplicated:
            base = base.drop(columns=duplicated)
    return base.reset_index(drop=True)


def train_val_split(frame: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(frame))
    rng.shuffle(indices)
    cutoff = int(len(indices) * (1.0 - val_ratio))
    cutoff = max(1, min(cutoff, len(indices) - 1))
    train_indices = indices[:cutoff]
    val_indices = indices[cutoff:]
    return frame.iloc[train_indices].reset_index(drop=True), frame.iloc[val_indices].reset_index(drop=True)


def build_datasets(
    frame: pd.DataFrame,
    label_column: str,
    target_column: str,
    val_ratio: float,
    seed: int,
) -> Tuple[TabularDataset, TabularDataset, TabularMetadata]:
    """Split a dataframe and return datasets plus metadata."""

    feature_columns = [
        column for column in frame.columns if column not in {label_column, target_column}
    ]
    continuous_columns: List[str] = []
    categorical_columns: List[str] = []
    for column in feature_columns:
        if column == target_column:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            continuous_columns.append(column)
        else:
            categorical_columns.append(column)

    train_frame, val_frame = train_val_split(frame, val_ratio, seed)

    preprocessor = TabularPreprocessor(continuous_columns, categorical_columns)
    preprocessor.fit(train_frame)

    train_cont, train_cat = preprocessor.transform(train_frame)
    val_cont, val_cat = preprocessor.transform(val_frame)

    train_targets = train_frame[target_column].to_numpy(dtype=np.int64)
    val_targets = val_frame[target_column].to_numpy(dtype=np.int64)

    metadata = TabularMetadata(
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        categorical_cardinalities=preprocessor.categorical_cardinalities,
    )

    return (
        TabularDataset(train_cont, train_cat, train_targets),
        TabularDataset(val_cont, val_cat, val_targets),
        metadata,
    )
