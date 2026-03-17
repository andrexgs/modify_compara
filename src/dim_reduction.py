#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dim_reduction.py
================
Extração de features e redução de dimensionalidade (t-SNE / PCA)
para visualização do espaço de representação do modelo.

Funções principais:
  extract_features()              — hook no penúltimo layer para obter embeddings
  plot_dimensionality_reduction() — scatter plot colorido por classe
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def extract_features(model, dataloader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrai os vetores de features do penúltimo layer do modelo.

    Estratégia de hook (em ordem de prioridade):
      1. model.avgpool   (ResNets, ConvNeXt)
      2. model.global_pool (redes timm)
      3. model.head       (ViTs e variantes)
      4. Fallback: usa diretamente a saída do modelo (logits)

    Retorna
    -------
    features : np.ndarray de shape (N, D)
    labels   : np.ndarray de shape (N,)
    """
    print("  Extraindo features para redução de dimensionalidade...")
    model.eval()

    activation: dict = {}

    def _make_hook(name: str):
        def _hook(m, inp, out):
            activation[name] = out.detach()
        return _hook

    handle = None
    for attr_name in ("avgpool", "global_pool", "head"):
        layer = getattr(model, attr_name, None)
        if isinstance(layer, nn.Module):
            handle = layer.register_forward_hook(_make_hook("feats"))
            break

    features_list: list[np.ndarray] = []
    labels_list:   list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) < 2:
                continue
            inputs, targets = batch[0].to(device), batch[1]

            model(inputs)

            feat = activation["feats"].flatten(1).cpu().numpy() \
                   if "feats" in activation \
                   else model(inputs).cpu().numpy()

            features_list.append(feat)
            labels_list.append(targets.numpy())
            activation.clear()   # limpa para o próximo batch

    if handle:
        handle.remove()

    if not features_list:
        return np.array([]), np.array([])

    return np.concatenate(features_list), np.concatenate(labels_list)


def plot_dimensionality_reduction(
    features: np.ndarray,
    labels:   np.ndarray,
    class_names: list[str],
    run_name: str,
    method: str = "t-SNE",
):
    """
    Cria um scatter plot 2D das features usando t-SNE ou PCA.
    Salva o resultado em ../results/dr/<run_name>_<method>.png.

    Parâmetros
    ----------
    features    : embeddings de shape (N, D)
    labels      : classes de cada amostra, shape (N,)
    class_names : lista de nomes das classes
    run_name    : identificador do experimento (usado no nome do arquivo)
    method      : "t-SNE" (padrão) ou "PCA"
    """
    if len(features) == 0:
        return

    if method == "t-SNE":
        perplexity = min(30, len(features) - 1) if len(features) > 1 else 1
        reducer    = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        reducer = PCA(n_components=2)

    try:
        proj = reducer.fit_transform(features)
    except Exception as e:
        print(f"  [DR] Redução de dimensionalidade falhou: {e}")
        return

    safe_labels = [
        class_names[int(l)] if int(l) < len(class_names) else f"C{l}"
        for l in labels
    ]
    df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "classe": safe_labels})

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="x", y="y", hue="classe", palette="tab10", alpha=0.8, s=60)
    plt.title(f"{method} — {run_name}")
    plt.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc="upper left")

    save_path = f"../results/dr/{run_name}_{method}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  [DR] Salvo em {save_path}")
