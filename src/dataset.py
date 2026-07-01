"""
dataset.py

Carga el dataset dr_unified_v2 (estructura train/val/test con carpetas 0-4 por severidad)
2 modos:
    - "binary": 0 -> sin RD (clase 0), 1-4 -> con RD (clase 1)
    - "multiclass": se mantienen las 5 clases originales (0-4)

El dataset ya viene con split train/val/test (EyePACS+APTOS+Messidor)
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Aplica CLAHE sobre el espacio LAB de una imagen RGB
    Mejora el contraste local, para resaltar microaneurismas o hemorragias
    en fondos de ojo con bajo contraste (H4 del TFM)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class RetinopathyDataset(Dataset):
    """
    Espera una carpeta raíz con subcarpetas '0'..'4', cada una con las imágenes de esa clase 
    """

    def __init__(self, root_dir: str, task_mode: str = "binary", image_size: int = 380, use_clahe: bool = True, augment: bool = False):
        assert task_mode in ("binary", "multiclass")
        self.root_dir = root_dir
        self.task_mode = task_mode
        self.use_clahe = use_clahe
        self.image_size = image_size

        self.samples = []  # lista de (filepath, raw_label_int)
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            try:
                raw_label = int(class_name)
            except ValueError:
                continue  # ignora carpetas que no sean '0'..'4'
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), raw_label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No se encontraron imágenes en {root_dir}"
            )

        self.transform = self._build_transform(augment)

    def _build_transform(self, augment: bool):
        ops = []
        if augment:
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        ops += [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ]
        return transforms.Compose(ops)

    def _map_label(self, raw_label: int) -> int:
        if self.task_mode == "binary":
            return 0 if raw_label == 0 else 1
        return raw_label  # multiclass: 0..4 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, raw_label = self.samples[idx]
        img = Image.open(filepath).convert("RGB")

        if self.use_clahe:
            img_np = np.array(img)
            img_np = apply_clahe(img_np)
            img = Image.fromarray(img_np)

        img = self.transform(img)
        label = self._map_label(raw_label)
        return img, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self) -> int:
        return 2 if self.task_mode == "binary" else 5

    def get_class_distribution(self) -> dict:
        dist = {}
        for _, raw_label in self.samples:
            label = self._map_label(raw_label)
            dist[label] = dist.get(label, 0) + 1
        return dict(sorted(dist.items()))
