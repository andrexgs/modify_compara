#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_manager.py
===============
Responsável por carregar, transformar e dividir os dados para:
  - Pipeline principal (Active Learning): get_al_data()
  - Pipeline Siamese: get_siamese_data()

As transformações e o split treino/validação são controlados por
hyperparameters.py — não mexa aqui para mudar augmentation ou tamanho de imagem.
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

from hyperparameters import DATA_HYPERPARAMETERS, DATA_AUGMENTATION, ACTIVE_LEARNING, SIAMESE_DATA_HYPERPARAMETERS


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms():
    """
    Retorna um par (transform_treino, transform_val).
    
    - transform_treino: inclui augmentation (flip, rotação, jitter)
    - transform_val:    apenas resize + normalize (sem augmentation)
    """
    # Normalização padrão ImageNet (usada por todas as redes pré-treinadas)
    _mean = [0.485, 0.456, 0.406]
    _std  = [0.229, 0.224, 0.225]
    _size = DATA_HYPERPARAMETERS["IMAGE_SIZE"]

    aug = DATA_HYPERPARAMETERS["USE_DATA_AUGMENTATION"]

    train_list = [
        transforms.Resize((_size, _size)),
    ]
    if aug:
        train_list += [
            transforms.RandomHorizontalFlip(p=DATA_AUGMENTATION["HORIZONTAL_FLIP"]),
            transforms.RandomVerticalFlip(p=DATA_AUGMENTATION["VERTICAL_FLIP"]),
            transforms.RandomRotation(degrees=DATA_AUGMENTATION["ROTATION"]),
        ]
        if DATA_AUGMENTATION["COLOR_JITTER"] > 0:
            train_list.append(
                transforms.ColorJitter(brightness=DATA_AUGMENTATION["COLOR_JITTER"])
            )
    train_list += [
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ]

    val_list = [
        transforms.Resize((_size, _size)),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ]

    return transforms.Compose(train_list), transforms.Compose(val_list)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL — Active Learning
# ─────────────────────────────────────────────────────────────────────────────

def get_al_data():
    """
    Prepara os dados para o loop de Active Learning.

    Retorna
    -------
    train_aug    : Dataset completo de treino com augmentation (para treinar)
    train_clean  : Dataset completo de treino sem augmentation (para inferência do oráculo)
    labeled_idx  : Índices inicialmente rotulados (pool rotulado)
    unlabeled_idx: Índices restantes (pool não-rotulado)
    val_loader   : DataLoader de validação (fixo em todos os ciclos)
    test_loader  : DataLoader de teste (fixo)
    """
    train_dir = DATA_HYPERPARAMETERS["TRAIN_DATA_DIR"]
    test_dir  = DATA_HYPERPARAMETERS["TEST_DATA_DIR"]

    trans_train, trans_val = get_transforms()

    # Dois datasets apontando para os mesmos arquivos, mas com transforms diferentes
    train_aug   = datasets.ImageFolder(train_dir, transform=trans_train)
    train_clean = datasets.ImageFolder(train_dir, transform=trans_val)
    test_ds     = datasets.ImageFolder(test_dir,  transform=trans_val)

    # Divisão treino / validação (80/20 por padrão)
    n_total = len(train_aug)
    n_val   = int(n_total * DATA_HYPERPARAMETERS["VAL_SPLIT"])
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)   # seed fixa → reprodutibilidade
    train_subset, val_subset = random_split(train_aug, [n_train, n_val], generator=generator)
    train_indices = train_subset.indices

    # Inicialização do pool rotulado (seleção aleatória com seed fixa)
    np.random.seed(42)
    initial_size  = min(len(train_indices), ACTIVE_LEARNING["INITIAL_LABELED"])
    shuffled      = np.random.permutation(train_indices)
    labeled_idx   = list(shuffled[:initial_size])
    unlabeled_idx = list(shuffled[initial_size:])

    bs      = DATA_HYPERPARAMETERS["BATCH_SIZE"]
    workers = DATA_HYPERPARAMETERS["NUM_WORKERS"]

    val_loader  = DataLoader(val_subset, batch_size=bs, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds,    batch_size=bs, shuffle=False, num_workers=workers)

    return train_aug, train_clean, labeled_idx, unlabeled_idx, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE SIAMESE
# ─────────────────────────────────────────────────────────────────────────────

class SiamesePairDataset(torch.utils.data.Dataset):
    """
    Gera pares (img_a, img_b, label) onde:
      label = 1 → mesma classe (par positivo)
      label = 0 → classes diferentes (par negativo)
    """

    def __init__(self, base_dataset, samples_per_class: int = 30):
        self.dataset   = base_dataset
        self.classes   = base_dataset.classes
        self.n_classes = len(self.classes)

        # Agrupa índices por classe
        self._idx_by_class: dict[int, list[int]] = {c: [] for c in range(self.n_classes)}
        for i, (_, label) in enumerate(base_dataset.samples):
            self._idx_by_class[label].append(i)

        self.samples_per_class = samples_per_class
        self._pairs = self._build_pairs()

    def _build_pairs(self):
        pairs = []
        for cls in range(self.n_classes):
            same_idxs  = self._idx_by_class[cls]
            other_idxs = [i for c, idxs in self._idx_by_class.items()
                          if c != cls for i in idxs]

            # Pares positivos
            for _ in range(self.samples_per_class):
                a, b = np.random.choice(same_idxs, 2, replace=len(same_idxs) < 2)
                pairs.append((int(a), int(b), 1))

            # Pares negativos
            for _ in range(self.samples_per_class):
                a = int(np.random.choice(same_idxs))
                b = int(np.random.choice(other_idxs))
                pairs.append((a, b, 0))

        return pairs

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        a_idx, b_idx, label = self._pairs[idx]
        img_a, _ = self.dataset[a_idx]
        img_b, _ = self.dataset[b_idx]
        return img_a, img_b, torch.tensor(label, dtype=torch.float32)


def get_siamese_data():
    """
    Retorna DataLoaders para o pipeline Siamese:
      train_loader_pairs : pares (img_a, img_b, label) para treino de reconhecimento
      train_loader_cls   : imagens individuais (img, label) para treino de classificação
      val_loader         : validação
      test_loader        : teste

    Configuração em SIAMESE_DATA_HYPERPARAMETERS (hyperparameters.py).
    """
    train_dir = SIAMESE_DATA_HYPERPARAMETERS["TRAIN_DATA_DIR"]
    test_dir  = SIAMESE_DATA_HYPERPARAMETERS["TEST_DATA_DIR"]

    _, trans_val = get_transforms()
    trans_train, _ = get_transforms()

    full_ds = datasets.ImageFolder(train_dir, transform=trans_train)
    test_ds = datasets.ImageFolder(test_dir,  transform=trans_val)

    n_total = len(full_ds)
    n_val   = int(n_total * SIAMESE_DATA_HYPERPARAMETERS["VAL_SPLIT"])
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=generator)

    # Dataset de pares para o cabeçalho de reconhecimento
    train_base = datasets.ImageFolder(train_dir, transform=trans_train)
    pair_ds    = SiamesePairDataset(
        train_base,
        samples_per_class=SIAMESE_DATA_HYPERPARAMETERS.get("CLASS_SAMPLE_SIZE", 30),
    )

    bs_rec = SIAMESE_DATA_HYPERPARAMETERS["BATCH_SIZE_REC"]
    bs_cls = SIAMESE_DATA_HYPERPARAMETERS["BATCH_SIZE_CLS"]
    workers = DATA_HYPERPARAMETERS["NUM_WORKERS"]

    train_loader_pairs = DataLoader(pair_ds,   batch_size=bs_rec, shuffle=True,  num_workers=workers)
    train_loader_cls   = DataLoader(train_sub, batch_size=bs_cls, shuffle=True,  num_workers=workers)
    val_loader         = DataLoader(val_sub,   batch_size=bs_cls, shuffle=False, num_workers=workers)
    test_loader        = DataLoader(test_ds,   batch_size=bs_cls, shuffle=False, num_workers=workers)

    return train_loader_pairs, train_loader_cls, val_loader, test_loader
