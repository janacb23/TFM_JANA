"""
evaluate_cnn.py

Evalúa los modelos CNN ya entrenados sobre el conjunto de test
Carga los checkpoints y calcula las métricas finales (AUC, sensibilidad, 
especificidad, F1, accuracy) con media +- desviación entre los 3 runs.

Uso:
    python evaluate_cnn.py --config ../configs/config.yaml
    python evaluate_cnn.py --config ../configs/config.yaml --architecture efficientnet_b0 --task_mode binary
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from dataset import RetinopathyDataset
from models import build_model
from metrics import compute_binary_metrics, compute_multiclass_metrics, aggregate_runs


@torch.no_grad()
def evaluate_on_test(model, loader, device, task_mode: str) -> dict:
    model.eval()
    all_labels, all_preds, all_probas = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        if task_mode == "binary":
            all_probas.extend(probs[:, 1])
        else:
            all_probas.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probas)

    if task_mode == "binary":
        return compute_binary_metrics(y_true, y_pred, y_proba)
    else:
        return compute_multiclass_metrics(y_true, y_pred, y_proba)


def evaluate_architecture(cfg, architecture, task_mode, device):
    data_cfg = cfg["data"]
    test_dir = os.path.join(data_cfg["data_root"], data_cfg["test_dir"])

    test_ds = RetinopathyDataset(
        test_dir, task_mode=task_mode,
        image_size=data_cfg["image_size"],
        use_clahe=cfg["augmentation"]["clahe"],
        augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=data_cfg["batch_size"],
        shuffle=False, num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    num_classes = test_ds.get_num_classes()
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    n_runs = cfg["experiment"]["n_runs"]

    run_metrics = []
    for run_idx in range(n_runs):
        ckpt_path = os.path.join(ckpt_dir, f"{architecture}_{task_mode}_run{run_idx}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [AVISO] No se encontró checkpoint: {ckpt_path}")
            continue

        model = build_model(architecture, num_classes).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        metrics = evaluate_on_test(model, test_loader, device, task_mode)
        print(f"  Run {run_idx}: " + " | ".join([f"{k}: {v*100:.2f}%" for k, v in metrics.items()]))
        run_metrics.append(metrics)

    if not run_metrics:
        print(f"  No se encontraron checkpoints para {architecture} {task_mode}")
        return None

    return aggregate_runs(run_metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    parser.add_argument("--architecture", type=str, default=None,
                        choices=["efficientnet_b0", "mobilenet_v3_large"])
    parser.add_argument("--task_mode", type=str, default=None,
                        choices=["binary", "multiclass"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Si se especifica arquitectura y modo, evalúa solo esa combinación
    if args.architecture and args.task_mode:
        combos = [(args.architecture, args.task_mode)]
    else:
        # Evalúa todas las combinaciones
        combos = [
            (arch, mode)
            for arch in cfg["experiment"]["architectures"]
            for mode in cfg["experiment"]["task_modes"]
        ]

    all_summaries = {}
    for architecture, task_mode in combos:
        print(f"\n=== {architecture} | {task_mode} ===")
        summary = evaluate_architecture(cfg, architecture, task_mode, device)
        if summary is None:
            continue

        all_summaries[f"{architecture}_{task_mode}"] = summary

        # Guardar JSON individual
        out_path = os.path.join(results_dir, f"{architecture}_{task_mode}_TEST_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Guardado en: {out_path}")

        # Imprimir resumen
        print("  Resumen (media ± std):")
        for metric, vals in summary.items():
            print(f"    {metric}: {vals['mean']*100:.2f}% ± {vals['std']*100:.2f}%")

    # Guardar resumen global
    global_path = os.path.join(results_dir, "TEST_all_cnn_summary.json")
    with open(global_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Evaluación en test completada. Resultados en: {results_dir}")


if __name__ == "__main__":
    main()
