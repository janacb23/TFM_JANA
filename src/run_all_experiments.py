"""
run_all_experiments.py

Lanza la matriz completa de experimentos definida en config.yaml:
    arquitecturas x task_modes x n_runs

Al final genera una tabla CSV comparativa (binario vs multiclase, por arquitectura) con media +- desviación

Uso:
    python run_all_experiments.py --config ../configs/config.yaml
"""

import os
import json
import argparse
import subprocess
import yaml
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    architectures = cfg["experiment"]["architectures"]
    task_modes = cfg["experiment"]["task_modes"]
    results_dir = cfg["paths"]["results_dir"]

    for arch in architectures:
        for task_mode in task_modes:
            print(f"\n{'='*60}")
            print(f"Lanzando: arch={arch} | task_mode={task_mode}")
            print(f"{'='*60}")
            subprocess.run([
                "python", "train_cnn.py",
                "--config", args.config,
                "--architecture", arch,
                "--task_mode", task_mode,
            ], check=True)

    # --- Construir tabla comparativa ---
    rows = []
    for arch in architectures:
        for task_mode in task_modes:
            summary_path = os.path.join(results_dir, f"{arch}_{task_mode}_summary.json")
            if not os.path.exists(summary_path):
                continue
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            row = {"architecture": arch, "task_mode": task_mode}
            for metric_name, vals in summary.items():
                row[f"{metric_name}_mean"] = round(vals["mean"] * 100, 2)
                row[f"{metric_name}_std"] = round(vals["std"] * 100, 2)
            row["n_runs"] = summary[list(summary.keys())[0]]["n_runs"]
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, "comparativa_binario_vs_multiclase.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nTabla comparativa guardada en: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
