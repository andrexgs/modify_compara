#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sanity_check.py
===============
Testa rapidamente se o ambiente está funcionando corretamente.

O que ele verifica:
  1. Carregamento de dados (data_manager)
  2. Criação do modelo ResNet18
  3. Overfitting em 1 batch (se a rede não consegue decorar 32 imagens, há problema)

Como usar:
  cd src
  python sanity_check.py

Resultado esperado:
  "SUCESSO TOTAL" antes da época 20.
  Se não atingir 100% em 20 épocas, revise a normalização
  e verifique se as pastas de dados estão corretas.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data_manager
from arch_optim import get_architecture
from hyperparameters import DATA_HYPERPARAMETERS, MODEL_HYPERPARAMETERS


def sanity_check():
    print("\n" + "=" * 60)
    print("  SANITY CHECK")
    print("  Objetivo: a rede deve conseguir decorar 1 batch (32 imgs)")
    print("=" * 60)

    device = MODEL_HYPERPARAMETERS["DEVICE"]
    print(f"  Device: {device}\n")

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    print("[1/3] Carregando dados...")
    try:
        train_aug, *_ = data_manager.get_al_data()
        dl = DataLoader(train_aug, batch_size=32, shuffle=True, num_workers=0)
        batch = next(iter(dl))
        x, y = batch[0].to(device), batch[1].to(device)
        print(f"  Imagens: {x.shape}  |  Labels: {y.shape}")
        print(f"  Classes encontradas: {train_aug.classes}")
    except Exception as e:
        print(f"  [ERRO] {e}")
        print("  Verifique se data/train/ existe e tem subpastas por classe.")
        return

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    print("\n[2/3] Criando modelo ResNet18...")
    try:
        num_classes = len(train_aug.classes)
        model = get_architecture("resnet18", 3, num_classes, pretrained=True).to(device)
        model.train()
        print(f"  Modelo criado para {num_classes} classes.")
    except Exception as e:
        print(f"  [ERRO] {e}")
        return

    # ── 3. Overfit em 1 batch ─────────────────────────────────────────────────
    print("\n[3/3] Overfit em 20 épocas...")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    acc = 0.0

    for ep in range(1, 21):
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        correct = out.argmax(1).eq(y).sum().item()
        acc     = 100.0 * correct / y.size(0)
        print(f"  Ep {ep:02d}  loss={loss.item():.4f}  acc={acc:.1f}%")

        if acc == 100.0:
            print("\n" + "=" * 60)
            print("  SUCESSO TOTAL! Ambiente configurado corretamente.")
            print("=" * 60)
            return

    print("\n" + "=" * 60)
    if acc >= 90.0:
        print("  SUCESSO PARCIAL (acc > 90%). Ambiente OK.")
    else:
        print("  FALHA: a rede não conseguiu decorar 32 imagens.")
        print("  Causas possíveis:")
        print("    - Imagens corrompidas ou pasta de treino vazia")
        print("    - Normalização incorreta em data_manager.py")
        print("    - Número de classes errado")
    print("=" * 60)


if __name__ == "__main__":
    sanity_check()
