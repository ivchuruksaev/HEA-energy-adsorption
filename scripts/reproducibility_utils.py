import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def parse_composition_elements(composition):
    elements = []
    current = ""

    for char in str(composition):
        if char.isupper() and current:
            elements.append(current)
            current = char
        else:
            current += char

    if current:
        elements.append(current)

    return elements


def build_binary_composition_matrix(compositions, elements=None):
    parsed = [parse_composition_elements(item) for item in compositions]

    if elements is None:
        elements = sorted({element for row in parsed for element in row})

    rows = []

    for composition, row_elements in zip(compositions, parsed):
        values = {"composition": composition}
        element_set = set(row_elements)

        for element in elements:
            values[f"x_{element}"] = int(element in element_set)

        rows.append(values)

    return pd.DataFrame(rows), elements


def build_lq_feature_matrix(df, element_columns=None):
    if element_columns is None:
        element_columns = [column for column in df.columns if column.startswith("x_")]

    X_linear = df[element_columns].to_numpy(dtype=float)
    pair_features = []
    pair_names = []

    for i, left in enumerate(element_columns):
        for j in range(i + 1, len(element_columns)):
            right = element_columns[j]
            pair_features.append((X_linear[:, i] * X_linear[:, j]).reshape(-1, 1))
            pair_names.append(f"{left}:{right}")

    if pair_features:
        X_pairs = np.hstack(pair_features)
        X = np.hstack([X_linear, X_pairs])
        feature_names = element_columns + pair_names
    else:
        X = X_linear
        feature_names = element_columns

    return X, feature_names


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
