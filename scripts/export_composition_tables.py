import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproducibility_utils import build_binary_composition_matrix


def read_energy_file(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def values_from_keys(data, keys):
    for key in keys:
        value = data.get(key)

        if isinstance(value, list) and len(value) > 0:
            return [float(item) for item in value]

        if isinstance(value, (int, float)):
            return [float(value)]

    return []


def summarize_composition(path):
    composition = path.name
    s_values = []
    h_values = []
    n_runs = 0

    for json_path in sorted(path.glob("run_*/data.json")):
        data = read_energy_file(json_path)
        s_values.extend(values_from_keys(data, ["energies_S_ads_raw", "E_ads_S", "E_S"]))
        h_values.extend(values_from_keys(data, ["energies_H_ads_raw", "E_ads_H", "E_H"]))
        n_runs += 1

    if not s_values or not h_values:
        return None

    e_s_mean = float(np.mean(s_values))
    e_h_mean = float(np.mean(h_values))
    d_value = e_s_mean + 2.0 * e_h_mean

    return {
        "composition": composition,
        "n_runs": n_runs,
        "E_ads_S_mean": e_s_mean,
        "E_ads_S_std": float(np.std(s_values)),
        "E_ads_H_mean": e_h_mean,
        "E_ads_H_std": float(np.std(h_values)),
        "D": d_value,
    }


def export_tables(base_path, out_dir):
    base_path = Path(base_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for composition_path in sorted(base_path.iterdir()):
        if composition_path.is_dir():
            record = summarize_composition(composition_path)

            if record is not None:
                records.append(record)

    if not records:
        raise RuntimeError("No valid composition records were found")

    dft_df = pd.DataFrame(records).sort_values("composition").reset_index(drop=True)
    binary_df, _ = build_binary_composition_matrix(dft_df["composition"])
    descriptor_df = dft_df[["composition", "D", "E_ads_S_mean", "E_ads_H_mean"]].merge(binary_df, on="composition")

    dft_path = out_dir / "dft_calculated_alloys.csv"
    descriptor_path = out_dir / "descriptor_table.csv"

    dft_df.to_csv(dft_path, index=False)
    descriptor_df.to_csv(descriptor_path, index=False)

    return dft_path, descriptor_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    dft_path, descriptor_path = export_tables(args.base_path, args.out_dir)

    print(dft_path)
    print(descriptor_path)


if __name__ == "__main__":
    main()
