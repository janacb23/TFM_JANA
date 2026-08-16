"""
evaluate_external_messidor.py

Evaluación externa de todos los modelos sobre Messidor-2
Messidor-2 tiene CSV con columnas: id_code, diagnosis (0-2), adjudicated_gradable
Se filtran solo las imágenes gradables (adjudicated_gradable == 1).
Binarización: 0 → sin RD, 1-2 → con RD.

Uso:
    python evaluate_external_messidor.py --config ../configs/config.yaml \
        --messidor_csv C:/ruta/messidor_data.csv \
        --messidor_images C:/ruta/messidor-2/preprocess
"""

import os
import json
import argparse
import numpy as np
import torch
import yaml
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model
from dataset import apply_clahe
from train_foundation import load_images_from_folder as load_folder


# ─────────────────────────────────────────────
# Dataset Messidor-2
# ─────────────────────────────────────────────

class MessidorDataset(Dataset):
    """
    Carga Messidor-2 desde CSV + carpeta de imágenes.
    Filtra imágenes no gradables y binariza: 0 → sin RD, 1-2 → con RD.
    """
    def __init__(self, csv_path: str, images_dir: str,
                 image_size: int = 380, use_clahe: bool = True):
        from torchvision import transforms
        df = pd.read_csv(csv_path)
        # Filtrar solo imágenes gradables
        df = df[df["adjudicated_gradable"] == 1].reset_index(drop=True)
        print(f"  Messidor-2: {len(df)} imágenes gradables de {len(pd.read_csv(csv_path))} totales")
        self.df = df
        self.images_dir = images_dir
        self.use_clahe = use_clahe

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["id_code"])
        img = Image.open(img_path).convert("RGB")

        if self.use_clahe:
            img_np = np.array(img)
            img_np = apply_clahe(img_np)
            img = Image.fromarray(img_np)

        img = self.transform(img)
        label = 0 if row["diagnosis"] == 0 else 1
        return img, torch.tensor(label, dtype=torch.long)


def load_messidor_pil(csv_path, images_dir):
    df = pd.read_csv(csv_path)
    df = df[df["adjudicated_gradable"] == 1].reset_index(drop=True)
    samples = []
    for _, row in df.iterrows():
        label = 0 if row["diagnosis"] == 0 else 1
        path = os.path.join(images_dir, row["id_code"])
        samples.append((path, label))
    return samples


# ─────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def aggregate(list_of_dicts):
    keys = list_of_dicts[0].keys()
    summary = {}
    for k in keys:
        values = np.array([m[k] for m in list_of_dicts])
        summary[k] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n_runs": len(values),
            "values": values.tolist(),
        }
    return summary


# ─────────────────────────────────────────────
# CNNs
# ─────────────────────────────────────────────

@torch.no_grad()
def eval_cnn_on_loader(model, loader, device):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(probs.argmax(axis=1).tolist())
        y_proba.extend(probs[:, 1].tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_proba)


def evaluate_cnns(cfg, csv_path, images_dir, device):
    print("\n=== CNNs supervisadas (Messidor-2 externo) ===")
    ds = MessidorDataset(csv_path, images_dir,
                         image_size=cfg["data"]["image_size"],
                         use_clahe=cfg["augmentation"]["clahe"])
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"],
                        shuffle=False, num_workers=cfg["data"]["num_workers"],
                        pin_memory=True)

    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    n_runs = cfg["experiment"]["n_runs"]
    results = {}

    for arch in ["efficientnet_b0", "mobilenet_v3_large"]:
        print(f"\n  {arch}:")
        run_metrics = []
        for run_idx in range(n_runs):
            ckpt_path = os.path.join(ckpt_dir, f"{arch}_binary_run{run_idx}.pt")
            if not os.path.exists(ckpt_path):
                print(f"    [AVISO] No encontrado: {ckpt_path}")
                continue
            model = build_model(arch, num_classes=2).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            y_true, y_pred, y_proba = eval_cnn_on_loader(model, loader, device)
            metrics = compute_metrics(y_true, y_pred, y_proba)
            print(f"    Run {run_idx}: AUC={metrics['auc']*100:.2f}%  "
                  f"Sens={metrics['sensitivity']*100:.2f}%  "
                  f"Spec={metrics['specificity']*100:.2f}%  "
                  f"F1={metrics['f1']*100:.2f}%")
            run_metrics.append(metrics)
        if run_metrics:
            results[arch] = aggregate(run_metrics)
            print(f"    Resumen AUC: {results[arch]['auc']['mean']*100:.2f}% "
                  f"± {results[arch]['auc']['std']*100:.2f}%")

    return results


# ─────────────────────────────────────────────
# Embeddings helpers
# ─────────────────────────────────────────────

def extract_clip_emb(samples, processor, model, device, batch_size=32):
    embeddings, labels = [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p, _ in batch]
        batch_labels = [l for _, l in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
        labels.extend(batch_labels)
    return np.vstack(embeddings), np.array(labels)


def extract_dino_emb(samples, processor, model, device, batch_size=32):
    embeddings, labels = [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p, _ in batch]
        batch_labels = [l for _, l in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
        labels.extend(batch_labels)
    return np.vstack(embeddings), np.array(labels)


# ─────────────────────────────────────────────
# CLIP zero-shot
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


def evaluate_clip_zeroshot(csv_path, images_dir, device):
    print("\n=== CLIP Zero-Shot (Messidor-2) ===")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    all_prompts = CLIP_PROMPTS["positive"] + CLIP_PROMPTS["negative"]
    text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    n_pos = len(CLIP_PROMPTS["positive"])
    pos_features = text_features[:n_pos].mean(dim=0, keepdim=True)
    neg_features = text_features[n_pos:].mean(dim=0, keepdim=True)

    samples = load_messidor_pil(csv_path, images_dir)
    y_true, y_pred, y_proba = [], [], []

    for i in range(0, len(samples), 32):
        batch = samples[i:i+32]
        images = [Image.open(p).convert("RGB") for p, _ in batch]
        labels = [l for _, l in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            img_features = model.get_image_features(**inputs)
            if hasattr(img_features, "pooler_output"):
                img_features = img_features.pooler_output
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        sim_pos = (img_features @ pos_features.T).squeeze().cpu().numpy()
        sim_neg = (img_features @ neg_features.T).squeeze().cpu().numpy()
        if np.ndim(sim_pos) == 0:
            sim_pos = np.array([float(sim_pos)])
            sim_neg = np.array([float(sim_neg)])

        proba = np.exp(sim_pos) / (np.exp(sim_pos) + np.exp(sim_neg))
        y_true.extend(labels)
        y_pred.extend((proba > 0.5).astype(int).tolist())
        y_proba.extend(proba.tolist())

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_proba))
    print(f"  AUC={metrics['auc']*100:.2f}%  Sens={metrics['sensitivity']*100:.2f}%  "
          f"Spec={metrics['specificity']*100:.2f}%  F1={metrics['f1']*100:.2f}%")
    return metrics


# ─────────────────────────────────────────────
# CLIP linear probe
# ─────────────────────────────────────────────

def evaluate_clip_linearprobe(cfg, csv_path, images_dir, device):
    print("\n=== CLIP Linear Probe (Messidor-2) ===")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    train_dir = os.path.join(cfg["data"]["data_root"], cfg["data"]["train_dir"])
    print("  Extrayendo embeddings del train (dr_unified_v2)...")
    train_samples = load_folder(train_dir, task_mode="binary")
    X_train, y_train = extract_clip_emb(train_samples, processor, model, device)

    print("  Extrayendo embeddings de Messidor-2...")
    test_samples = load_messidor_pil(csv_path, images_dir)
    X_test, y_test = extract_clip_emb(test_samples, processor, model, device)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print(f"  AUC={metrics['auc']*100:.2f}%  Sens={metrics['sensitivity']*100:.2f}%  "
          f"Spec={metrics['specificity']*100:.2f}%  F1={metrics['f1']*100:.2f}%")
    return metrics


# ─────────────────────────────────────────────
# DINOv2 linear probe
# ─────────────────────────────────────────────

def evaluate_dino_linearprobe(cfg, csv_path, images_dir, device):
    print("\n=== DINOv2 Linear Probe (Messidor-2) ===")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    train_dir = os.path.join(cfg["data"]["data_root"], cfg["data"]["train_dir"])
    print("  Extrayendo embeddings del train (dr_unified_v2)...")
    train_samples = load_folder(train_dir, task_mode="binary")
    X_train, y_train = extract_dino_emb(train_samples, processor, model, device)

    print("  Extrayendo embeddings de Messidor-2...")
    test_samples = load_messidor_pil(csv_path, images_dir)
    X_test, y_test = extract_dino_emb(test_samples, processor, model, device)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print(f"  AUC={metrics['auc']*100:.2f}%  Sens={metrics['sensitivity']*100:.2f}%  "
          f"Spec={metrics['specificity']*100:.2f}%  F1={metrics['f1']*100:.2f}%")
    return metrics


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    parser.add_argument("--messidor_csv", type=str, required=True,
                        help="Ruta al archivo messidor_data.csv")
    parser.add_argument("--messidor_images", type=str, required=True,
                        help="Ruta a la carpeta con las imágenes de Messidor-2")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "cnn", "clip_zeroshot",
                                 "clip_linearprobe", "dino_linearprobe"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    print(f"Dataset externo: Messidor-2")

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    if args.mode in ("all", "cnn"):
        results = evaluate_cnns(cfg, args.messidor_csv, args.messidor_images, device)
        all_results["cnn"] = results
        out = os.path.join(results_dir, "Messidor_cnn_summary.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nCNN guardado en: {out}")

    if args.mode in ("all", "clip_zeroshot"):
        metrics = evaluate_clip_zeroshot(args.messidor_csv, args.messidor_images, device)
        all_results["clip_zeroshot"] = metrics
        out = os.path.join(results_dir, "Messidor_clip_zeroshot_summary.json")
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)

    if args.mode in ("all", "clip_linearprobe"):
        metrics = evaluate_clip_linearprobe(cfg, args.messidor_csv, args.messidor_images, device)
        all_results["clip_linearprobe"] = metrics
        out = os.path.join(results_dir, "Messidor_clip_linearprobe_summary.json")
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)

    if args.mode in ("all", "dino_linearprobe"):
        metrics = evaluate_dino_linearprobe(cfg, args.messidor_csv, args.messidor_images, device)
        all_results["dino_linearprobe"] = metrics
        out = os.path.join(results_dir, "Messidor_dino_linearprobe_summary.json")
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)

    global_out = os.path.join(results_dir, "Messidor_all_summary.json")
    with open(global_out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Evaluación externa Messidor-2 completada. Resultados en: {results_dir}")


if __name__ == "__main__":
    main()
