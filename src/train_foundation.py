"""
train_foundation.py

Pipeline de modelos fundacionales para detección de retinopatía diabética
Implementa 4 configuraciones:
    1. CLIP zero-shot: clasificación por similitud con prompts de texto, sin ejemplos etiquetados
    2. CLIP few-shot: embeddings CLIP + clasificador ligero (Logistic Regression)
    3. DINOv2 zero-shot: clasificación por similitud de embeddings con k-NN (1-shot, 5-shot)
    4. DINOv2 linear probe: embeddings DINOv2 + clasificador ligero

Uso:
    python train_foundation.py --config ../configs/config.yaml --mode all
    python train_foundation.py --config ../configs/config.yaml --mode clip_zeroshot
    python train_foundation.py --config ../configs/config.yaml --mode clip_fewshot
    python train_foundation.py --config ../configs/config.yaml --mode dino_zeroshot
    python train_foundation.py --config ../configs/config.yaml --mode dino_linearprobe
"""

import os
import json
import argparse
import random
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, confusion_matrix
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel


# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_images_from_folder(root_dir: str, task_mode: str = "binary",
                             max_per_class: int = None):
    """
    Carga imágenes desde carpetas 0-4 (ImageFolder style)
    Devuelve lista de (PIL.Image, label)
    max_per_class: limita imágenes por clase (útil para few-shot)
    """
    samples = []
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        try:
            raw_label = int(class_name)
        except ValueError:
            continue
        label = 0 if (task_mode == "binary" and raw_label == 0) else (
            1 if task_mode == "binary" else raw_label
        )
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if max_per_class:
            files = files[:max_per_class]
        for fname in files:
            samples.append((os.path.join(class_dir, fname), label))
    return samples


def compute_binary_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def format_metrics(metrics: dict) -> str:
    return " | ".join([f"{k}: {v*100:.2f}%" for k, v in metrics.items()])


def save_results(results: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en: {out_path}")


# ─────────────────────────────────────────────
# 1. CLIP Zero-Shot
# ─────────────────────────────────────────────

CLIP_PROMPTS = {
    "positive": [
        "a retinal fundus image showing signs of diabetic retinopathy",
        "a fundus photograph with microaneurysms and hemorrhages",
        "a retina image with diabetic retinopathy lesions",
        "an eye fundus image with retinal damage from diabetes",
    ],
    "negative": [
        "a healthy retinal fundus image with no diabetic retinopathy",
        "a normal fundus photograph with no lesions",
        "a healthy retina image without diabetic retinopathy",
        "a normal eye fundus image without any pathology",
    ],
}


def clip_zeroshot(cfg: dict, device: torch.device) -> dict:
    print("\n=== CLIP Zero-Shot ===")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    data_root = cfg["data"]["data_root"]
    test_dir = os.path.join(data_root, cfg["data"]["test_dir"])
    samples = load_images_from_folder(test_dir, task_mode="binary")
    print(f"  Test samples: {len(samples)}")

    # Tokenizar prompts una vez
    all_prompts = CLIP_PROMPTS["positive"] + CLIP_PROMPTS["negative"]
    text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        if hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)        
    n_pos = len(CLIP_PROMPTS["positive"])
    pos_features = text_features[:n_pos].mean(dim=0, keepdim=True)
    neg_features = text_features[n_pos:].mean(dim=0, keepdim=True)

    y_true, y_pred, y_proba = [], [], []

    batch_size = 32
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p, _ in batch]
        labels = [l for _, l in batch]

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            img_features = model.get_image_features(**inputs)
            if hasattr(img_features, 'pooler_output'):
                img_features = img_features.pooler_output
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        sim_pos = (img_features @ pos_features.T).squeeze().cpu().numpy()
        sim_neg = (img_features @ neg_features.T).squeeze().cpu().numpy()

        if sim_pos.ndim == 0:
            sim_pos = np.array([float(sim_pos)])
            sim_neg = np.array([float(sim_neg)])

        proba = np.exp(sim_pos) / (np.exp(sim_pos) + np.exp(sim_neg))
        preds = (proba > 0.5).astype(int)

        y_true.extend(labels)
        y_pred.extend(preds.tolist())
        y_proba.extend(proba.tolist())

    metrics = compute_binary_metrics(np.array(y_true), np.array(y_pred), np.array(y_proba))
    print(f"  {format_metrics(metrics)}")
    return metrics


# ─────────────────────────────────────────────
# 2. CLIP Few-Shot
# ─────────────────────────────────────────────

def extract_clip_embeddings(samples, processor, model, device, batch_size=32):
    embeddings, labels = [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p, _ in batch]
        batch_labels = [l for _, l in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if hasattr(feats, 'pooler_output'):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
        labels.extend(batch_labels)
    return np.vstack(embeddings), np.array(labels)


def clip_fewshot(cfg: dict, device: torch.device,
                 n_runs: int = 3, base_seed: int = 42) -> dict:
    print("\n=== CLIP Few-Shot (linear probe) ===")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    data_root = cfg["data"]["data_root"]
    train_dir = os.path.join(data_root, cfg["data"]["train_dir"])
    test_dir = os.path.join(data_root, cfg["data"]["test_dir"])

    print("  Extrayendo embeddings de train...")
    train_samples = load_images_from_folder(train_dir, task_mode="binary")
    X_train, y_train = extract_clip_embeddings(train_samples, processor, model, device)

    print("  Extrayendo embeddings de test...")
    test_samples = load_images_from_folder(test_dir, task_mode="binary")
    X_test, y_test = extract_clip_embeddings(test_samples, processor, model, device)

    run_metrics = []
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        set_seed(seed)
        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        metrics = compute_binary_metrics(y_test, y_pred, y_proba)
        print(f"  Run {run_idx+1}: {format_metrics(metrics)}")
        run_metrics.append(metrics)

    summary = _aggregate(run_metrics)
    return summary


# ─────────────────────────────────────────────
# 3. DINOv2 Zero-Shot (k-NN con pocos ejemplos)
# ─────────────────────────────────────────────

def extract_dino_embeddings(samples, processor, model, device, batch_size=32):
    embeddings, labels = [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p, _ in batch]
        batch_labels = [l for _, l in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :]  # CLS token
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
        labels.extend(batch_labels)
    return np.vstack(embeddings), np.array(labels)


def dino_zeroshot(cfg: dict, device: torch.device,
                  k_shots: list = [1, 5], n_runs: int = 3,
                  base_seed: int = 42) -> dict:
    print("\n=== DINOv2 Zero-Shot (k-NN few-shot) ===")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    data_root = cfg["data"]["data_root"]
    train_dir = os.path.join(data_root, cfg["data"]["train_dir"])
    test_dir = os.path.join(data_root, cfg["data"]["test_dir"])

    print("  Extrayendo embeddings de test...")
    test_samples = load_images_from_folder(test_dir, task_mode="binary")
    X_test, y_test = extract_dino_embeddings(test_samples, processor, model, device)

    all_results = {}
    for k in k_shots:
        print(f"\n  -- {k}-shot --")
        run_metrics = []
        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            set_seed(seed)

            # Seleccionar k ejemplos por clase del train
            support_samples = []
            train_samples_all = load_images_from_folder(train_dir, task_mode="binary")
            class_0 = [s for s in train_samples_all if s[1] == 0]
            class_1 = [s for s in train_samples_all if s[1] == 1]
            random.shuffle(class_0)
            random.shuffle(class_1)
            support_samples = class_0[:k] + class_1[:k]

            X_support, y_support = extract_dino_embeddings(
                support_samples, processor, model, device
            )

            # k-NN con similitud coseno
            sims = X_test @ X_support.T  # (n_test, 2k)
            top_k_idx = np.argsort(-sims, axis=1)[:, :max(1, k)]
            y_pred = []
            y_proba = []
            for i in range(len(X_test)):
                neighbor_labels = y_support[top_k_idx[i]]
                neighbor_sims = sims[i, top_k_idx[i]]
                # Voto ponderado por similitud
                score_1 = neighbor_sims[neighbor_labels == 1].sum()
                score_0 = neighbor_sims[neighbor_labels == 0].sum()
                total = score_0 + score_1 + 1e-8
                prob_1 = score_1 / total
                y_proba.append(prob_1)
                y_pred.append(1 if prob_1 > 0.5 else 0)

            metrics = compute_binary_metrics(y_test, np.array(y_pred), np.array(y_proba))
            print(f"    Run {run_idx+1}: {format_metrics(metrics)}")
            run_metrics.append(metrics)

        all_results[f"{k}shot"] = _aggregate(run_metrics)

    return all_results


# ─────────────────────────────────────────────
# 4. DINOv2 Linear Probe
# ─────────────────────────────────────────────

def dino_linearprobe(cfg: dict, device: torch.device,
                     n_runs: int = 3, base_seed: int = 42) -> dict:
    print("\n=== DINOv2 Linear Probe ===")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    data_root = cfg["data"]["data_root"]
    train_dir = os.path.join(data_root, cfg["data"]["train_dir"])
    test_dir = os.path.join(data_root, cfg["data"]["test_dir"])

    print("  Extrayendo embeddings de train...")
    train_samples = load_images_from_folder(train_dir, task_mode="binary")
    X_train, y_train = extract_dino_embeddings(train_samples, processor, model, device)

    print("  Extrayendo embeddings de test...")
    test_samples = load_images_from_folder(test_dir, task_mode="binary")
    X_test, y_test = extract_dino_embeddings(test_samples, processor, model, device)

    run_metrics = []
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        set_seed(seed)
        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        metrics = compute_binary_metrics(y_test, y_pred, y_proba)
        print(f"  Run {run_idx+1}: {format_metrics(metrics)}")
        run_metrics.append(metrics)

    return _aggregate(run_metrics)


# ─────────────────────────────────────────────
# Runs
# ─────────────────────────────────────────────

def _aggregate(list_of_metric_dicts: list) -> dict:
    keys = list_of_metric_dicts[0].keys()
    summary = {}
    for k in keys:
        values = np.array([m[k] for m in list_of_metric_dicts])
        summary[k] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n_runs": len(values),
            "values": values.tolist(),
        }
    return summary


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "clip_zeroshot", "clip_fewshot",
                                 "dino_zeroshot", "dino_linearprobe"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    n_runs = cfg["experiment"]["n_runs"]
    base_seed = cfg["experiment"]["base_seed"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    if args.mode in ("all", "clip_zeroshot"):
        metrics = clip_zeroshot(cfg, device)
        save_results(metrics, os.path.join(results_dir, "clip_zeroshot_summary.json"))

    if args.mode in ("all", "clip_fewshot"):
        summary = clip_fewshot(cfg, device, n_runs=n_runs, base_seed=base_seed)
        save_results(summary, os.path.join(results_dir, "clip_fewshot_summary.json"))

    if args.mode in ("all", "dino_zeroshot"):
        results = dino_zeroshot(cfg, device, k_shots=[1, 5],
                                n_runs=n_runs, base_seed=base_seed)
        save_results(results, os.path.join(results_dir, "dino_zeroshot_summary.json"))

    if args.mode in ("all", "dino_linearprobe"):
        summary = dino_linearprobe(cfg, device, n_runs=n_runs, base_seed=base_seed)
        save_results(summary, os.path.join(results_dir, "dino_linearprobe_summary.json"))

    print("\n✓ Pipeline de modelos fundacionales completado.")
    print(f"Resultados en: {results_dir}")


if __name__ == "__main__":
    main()
