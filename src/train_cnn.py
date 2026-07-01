"""
train_cnn.py

Entrena EfficientNet-B0 / MobileNetV3-Large en modo binario y multiclase,
repitiendo el experimento n_runs veces con distintas seeds para reportar media +- desviación estándar 

Uso:
    python train_cnn.py --config ../configs/config.yaml --architecture efficientnet_b0 --task_mode binary
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from dataset import RetinopathyDataset
from models import build_model
from metrics import compute_binary_metrics, compute_multiclass_metrics, aggregate_runs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, task_mode: str):
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
            all_probas.extend(probs[:, 1])  # probabilidad de la clase positiva
        else:
            all_probas.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probas)

    if task_mode == "binary":
        return compute_binary_metrics(y_true, y_pred, y_proba)
    else:
        return compute_multiclass_metrics(y_true, y_pred, y_proba)


def run_single_experiment(cfg, architecture, task_mode, seed, device):
    set_seed(seed)

    data_cfg = cfg["data"]
    train_dir = os.path.join(data_cfg["data_root"], data_cfg["train_dir"])
    val_dir = os.path.join(data_cfg["data_root"], data_cfg["val_dir"])

    train_ds = RetinopathyDataset(
        train_dir, task_mode=task_mode, image_size=data_cfg["image_size"],
        use_clahe=cfg["augmentation"]["clahe"], augment=True,
    )
    val_ds = RetinopathyDataset(
        val_dir, task_mode=task_mode, image_size=data_cfg["image_size"],
        use_clahe=cfg["augmentation"]["clahe"], augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"],
                               shuffle=True, num_workers=data_cfg["num_workers"],
                               pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=data_cfg["batch_size"],
                             shuffle=False, num_workers=data_cfg["num_workers"],
                             pin_memory=True)

    num_classes = train_ds.get_num_classes()
    model = build_model(architecture, num_classes).to(device)

    train_cfg = cfg["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                                   weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["mixed_precision"])

    best_metric = -1.0
    best_state = None
    patience_counter = 0
    monitor_key = "auc" if task_mode == "binary" else "auc_macro"

    for epoch in range(train_cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                      device, scaler, train_cfg["mixed_precision"])
        val_metrics = evaluate(model, val_loader, device, task_mode)
        scheduler.step()

        print(f"  [seed={seed} arch={architecture} mode={task_mode}] "
              f"epoch {epoch+1}/{train_cfg['epochs']} "
              f"train_loss={train_loss:.4f} val_{monitor_key}={val_metrics[monitor_key]:.4f}")

        if val_metrics[monitor_key] > best_metric:
            best_metric = val_metrics[monitor_key]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping_patience"]:
                print(f"  Early stopping en epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    final_val_metrics = evaluate(model, val_loader, device, task_mode)
    return model, final_val_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    parser.add_argument("--architecture", type=str, required=True,
                         choices=["efficientnet_b0", "mobilenet_v3_large"])
    parser.add_argument("--task_mode", type=str, required=True,
                         choices=["binary", "multiclass"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    n_runs = cfg["experiment"]["n_runs"]
    base_seed = cfg["experiment"]["base_seed"]

    run_metrics = []
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        print(f"\n=== Run {run_idx+1}/{n_runs} (seed={seed}) ===")
        model, metrics = run_single_experiment(cfg, args.architecture, args.task_mode, seed, device)
        run_metrics.append(metrics)

        ckpt_dir = cfg["paths"]["checkpoints_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            ckpt_dir, f"{args.architecture}_{args.task_mode}_run{run_idx}.pt"
        )
        torch.save(model.state_dict(), ckpt_path)

    summary = aggregate_runs(run_metrics)

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{args.architecture}_{args.task_mode}_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResumen guardado en {out_path}")
    print(json.dumps(
        {k: f"{v['mean']*100:.2f}% ± {v['std']*100:.2f}%" for k, v in summary.items()},
        indent=2, ensure_ascii=False
    ))


if __name__ == "__main__":
    main()
