#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analysis.py
===========
Análise estatística rápida dos resultados (Python).
Para análise completa com ANOVA e boxplots, use o script R: graphics.R

Como usar:
  cd src
  python analysis.py
"""

import os
import numpy as np
if not hasattr(np, "float"):
    np.float = float

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RESULTS_PATH = "../results_dl/results.csv"
COLS         = ["run", "learning_rate", "architecture", "optimizer", "precision", "recall", "f1"]


def run_stats():
    if not os.path.exists(RESULTS_PATH):
        print(f"Arquivo de resultados não encontrado: {RESULTS_PATH}")
        print("Execute o experimento completo antes de rodar analysis.py.")
        return

    df = pd.read_csv(RESULTS_PATH, header=None, names=COLS)

    print("\n=== RESUMO POR ARQUITETURA + OTIMIZADOR ===")
    summary = df.groupby(["architecture", "optimizer"])["f1"].agg(
        ["mean", "std", "min", "max", "count"]
    ).round(4)
    print(summary.to_string())

    # Melhor configuração por métrica
    for metric in ["precision", "recall", "f1"]:
        best_row = df.loc[df[metric].idxmax()]
        print(f"\nMelhor {metric.upper()}: {best_row[metric]:.4f}")
        print(f"  Arch={best_row['architecture']}  Opt={best_row['optimizer']}  LR={best_row['learning_rate']}")

    # Boxplot simples
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="architecture", y="f1", hue="optimizer", palette="Set2")
    plt.title("F1-Score por Arquitetura e Otimizador")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = "../results_dl/boxplot_python.png"
    plt.savefig(out_path, dpi=200)
    plt.close("all")
    print(f"\nBoxplot salvo em {out_path}")


if __name__ == "__main__":
    run_stats()
