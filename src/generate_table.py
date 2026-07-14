import os
import json
import yaml
import pandas as pd

with open("../configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

architectures = cfg["experiment"]["architectures"]
task_modes = cfg["experiment"]["task_modes"]
results_dir = cfg["paths"]["results_dir"]

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