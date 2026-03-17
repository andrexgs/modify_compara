#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
helper_functions.py
===================
Funções de treino, validação, teste e visualização.

Principais funções:
  fit()                  — loop de treino com early stopping e Active Learning
  validate()             — avaliação no conjunto de validação
  test()                 — avaliação final com matriz de confusão
  get_uncertainty_samples() — seleção de amostras para AL (incerteza máxima)
  save_gradcam_images()  — salva imagens de erro com sobreposição GradCAM
  plot_combined_history() — plota loss/acurácia de todos os ciclos AL juntos
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")   # sem display — essencial para rodar em servidor
import matplotlib.pyplot as plt

import numpy as np
if not hasattr(np, "float"):    # compatibilidade numpy < 1.24
    np.float = float

import torch
import torch.nn as nn
import seaborn as sns
import pandas as pd
from sklearn import metrics


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENTOS DE LINHA DE COMANDO
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> dict:
    """Lê e valida os argumentos passados ao script main.py."""
    parser = argparse.ArgumentParser(description="Comparador de classificadores PyTorch")
    parser.add_argument("-a", "--architecture", type=str, required=True,
                        help="Nome da arquitetura (ex: resnet18)")
    parser.add_argument("-o", "--optimizer",    type=str, default="sgd",
                        help="Nome do otimizador (ex: sgd, adamw, adagrad)")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001,
                        help="Taxa de aprendizado")
    parser.add_argument("-r", "--run",          type=str, default="1",
                        help="Número da dobra / identificador do experimento")
    parser.add_argument("-p", "--procedure",    type=str, default="completo",
                        choices=["treino", "teste", "completo"],
                        help="'treino', 'teste' ou 'completo'")
    return vars(parser.parse_args())


# ─────────────────────────────────────────────────────────────────────────────
# LOSS CUSTOMIZADA — GuidedAttentionLoss
# ─────────────────────────────────────────────────────────────────────────────

class GuidedAttentionLoss(nn.Module):
    """
    CrossEntropy + penalização suave para que a atenção da rede
    se concentre no centro da imagem.

    Parâmetros
    ----------
    alpha : float
        Peso do termo de atenção. Use 0 para desabilitar (equivale a CrossEntropy puro).
    class_weights : Tensor | None
        Pesos por classe para lidar com desbalanceamento.
    """

    def __init__(self, alpha: float = 0.0001, class_weights=None):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.alpha    = alpha

    def _center_mask(self, size: int, device) -> torch.Tensor:
        """Máscara gaussiana que penaliza ativações nas bordas."""
        xs = torch.arange(size, dtype=torch.float32, device=device)
        ys = torch.arange(size, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        center = size // 2
        mask   = (xx - center) ** 2 + (yy - center) ** 2
        return mask / (mask.max() + 1e-8)

    def forward(self, predictions, labels, feature_maps=None):
        cls = self.cls_loss(predictions, labels)

        if feature_maps is None or self.alpha == 0:
            return cls

        feature_maps = feature_maps.float()
        spatial_attn = torch.mean(feature_maps, dim=1).relu()
        mask         = self._center_mask(feature_maps.size(2), feature_maps.device)
        attn_loss    = torch.mean(spatial_attn * mask)

        return cls + self.alpha * attn_loss


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE LEARNING — seleção de amostras por incerteza
# ─────────────────────────────────────────────────────────────────────────────

def get_uncertainty_samples(model, unlabeled_loader, n_samples: int) -> list[int]:
    """
    Seleciona os índices das amostras mais incertas (menor confiança máxima).
    
    Retorna lista de índices no unlabeled_loader.dataset.
    """
    model.eval()
    device = next(model.parameters()).device
    uncertainties = []

    with torch.no_grad():
        for batch in unlabeled_loader:
            x      = batch[0].to(device)
            probs  = torch.softmax(model(x), dim=1)
            uncert = 1.0 - probs.max(dim=1).values
            uncertainties.extend(uncert.cpu().tolist())

    # Retorna os índices das N amostras mais incertas
    return np.argsort(uncertainties)[-n_samples:][::-1].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# TREINO
# ─────────────────────────────────────────────────────────────────────────────

def fit(train_dl, val_dl, model, optimizer, loss_fn, epochs: int,
        patience: int, tolerance: float, save_path: str) -> pd.DataFrame:
    """
    Loop de treino com:
      - Early stopping (patience + tolerance)
      - ReduceLROnPlateau scheduler
      - Gradient clipping (max_norm=1.0)
      - Suporte automático ao SAM (dois passos de gradiente)
      - Hook para capturar feature maps (GuidedAttentionLoss)

    Salva o melhor modelo (menor val_loss) em save_path.

    Retorna um DataFrame com as colunas:
      loss, val_loss, accuracy, val_accuracy
    """
    device    = next(model.parameters()).device
    is_sam    = hasattr(optimizer, "first_step")  # detecta SAM
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer if not is_sam else optimizer.base_optimizer,
        mode="min", factor=0.5, patience=3,
    )

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_loss  = float("inf")
    no_improve = 0
    handle     = None

    # Hook para capturar o último feature map (GuidedAttentionLoss)
    feature_maps: list = []
    if isinstance(loss_fn, GuidedAttentionLoss) and loss_fn.alpha > 0:
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            def _hook(m, i, o):
                feature_maps.clear()
                feature_maps.append(o)
            handle = last_conv.register_forward_hook(_hook)

    print(f"Treinando em {device} | {len(train_dl)} lotes por época")

    for ep in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for batch in train_dl:
            if len(batch) < 2:
                continue
            x, y = batch[0].to(device), batch[1].to(device)

            if is_sam:
                # SAM: primeiro passo
                feature_maps.clear() # <-- Força a limpeza do mapa anterior
                out  = model(x)
                fmap = feature_maps[0] if len(feature_maps) > 0 else None
                loss = loss_fn(out, y, fmap)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # SAM: segundo passo
                feature_maps.clear() # <-- Limpa novamente para o 2º passo
                out2 = model(x)
                fmap2 = feature_maps[0] if len(feature_maps) > 0 else None
                loss_fn(out2, y, fmap2).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                feature_maps.clear() # <-- GARANTE que não há lixo do passado
                out  = model(x)
                fmap = feature_maps[0] if len(feature_maps) > 0 else None
                loss = loss_fn(out, y, fmap)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            run_loss += loss.item()
            correct  += out.argmax(1).eq(y).sum().item()
            total    += y.size(0)

        val_loss, val_acc = validate(val_dl, model, loss_fn)
        ep_loss = run_loss / max(len(train_dl), 1)
        ep_acc  = 100.0 * correct / max(total, 1)

        scheduler.step(val_loss)

        history["loss"].append(ep_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(ep_acc)
        history["val_accuracy"].append(val_acc)

        marker = ""
        if val_loss < best_loss - tolerance:
            best_loss  = val_loss
            no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            marker = " ✓ salvo"
        else:
            no_improve += 1
            marker = f" [{no_improve}/{patience}]"

        print(
            f"  Ep {ep+1:03d} | "
            f"loss {ep_loss:.4f} acc {ep_acc:.1f}% | "
            f"val {val_loss:.4f} acc {val_acc:.1f}%"
            f"{marker}"
        )

        if no_improve >= patience:
            print(f"  Early stopping na época {ep+1}.")
            break

    if handle:
        handle.remove()

    return pd.DataFrame(history)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def validate(dl, model, loss_fn) -> tuple[float, float]:
    """Avalia o modelo no DataLoader dado. Retorna (loss, acurácia %)."""
    model.eval()
    device = next(model.parameters()).device
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in dl:
            x, y = batch[0].to(device), batch[1].to(device)
            out  = model(x)
            total_loss += loss_fn(out, y, None).item()
            correct    += out.argmax(1).eq(y).sum().item()
            total      += y.size(0)

    return total_loss / max(len(dl), 1), 100.0 * correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# TESTE FINAL
# ─────────────────────────────────────────────────────────────────────────────

def test(dl, model, path_csv: str, path_png: str, classes: list) -> tuple[float, float, float]:
    """
    Avalia o modelo no conjunto de teste.
    Salva a matriz de confusão em CSV (números) e PNG (visualização com nomes).

    Retorna (precision, recall, f1) médias macro.
    """
    model.eval()
    device = next(model.parameters()).device
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in dl:
            x, y = batch[0].to(device), batch[1].to(device)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(model(x).argmax(1).cpu().numpy())

    unique = sorted(set(y_true) | set(y_pred))
    names  = [classes[i] for i in unique] if max(unique, default=0) < len(classes) \
              else [str(i) for i in unique]

    report = metrics.classification_report(
        y_true, y_pred,
        labels=unique, target_names=names,
        zero_division=0, output_dict=True,
    )
    cm = metrics.confusion_matrix(y_true, y_pred, labels=unique)

    # CSV com apenas números (requisito do script R)
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    pd.DataFrame(cm).to_csv(path_csv)

    # PNG com nomes das classes
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.close("all")

    return (
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
        report["macro avg"]["f1-score"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# GRADCAM — salva imagens de erro com sobreposição de atenção
# ─────────────────────────────────────────────────────────────────────────────

def save_gradcam_images(model, dataloader, layer, class_names: list, run_name: str):
    """
    Para cada imagem classificada incorretamente no primeiro batch,
    gera e salva uma visualização GradCAM.
    Máximo de 10 imagens salvas por chamada.

    Requer: pip install captum
    """
    try:
        from captum.attr import LayerGradCam
        from captum.attr import visualization as viz
    except ImportError:
        print("[GradCAM] captum não instalado. Pulando geração de GradCAM.")
        return

    if layer is None:
        return

    model.eval()
    device    = next(model.parameters()).device
    save_dir  = f"../results/gradcam/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    grad_cam = LayerGradCam(model, layer)

    try:
        batch  = next(iter(dataloader))
        inputs = batch[0]
        labels = batch[1]
        fnames = batch[2] if len(batch) >= 3 else [f"img_{i}.jpg" for i in range(len(inputs))]
    except Exception:
        return

    inputs = inputs.to(device)
    preds  = model(inputs).argmax(dim=1)
    attrs  = grad_cam.attribute(inputs, target=preds)

    count = 0
    for i in range(len(inputs)):
        if labels[i] == preds[i]:
            continue
        count += 1
        img  = inputs[i].cpu().permute(1, 2, 0).detach().numpy()
        attr = attrs[i].cpu().permute(1, 2, 0).detach().numpy()
        img  = (img - img.min()) / (img.max() - img.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(5, 5))
        viz.visualize_image_attr(
            attr, img,
            method="blended_heat_map", sign="all",
            show_colorbar=True,
            title=(f"Real: {class_names[labels[i]]}\n"
                   f"Pred: {class_names[preds[i]]}"),
            plt_fig_axis=(fig, ax),
            use_pyplot=False,
        )
        fname = str(fnames[i]).split("/")[-1]
        fig.savefig(f"{save_dir}/ERRO_{fname}.png", bbox_inches="tight")
        plt.close("all")

        if count >= 10:
            break


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZAÇÃO — histórico de treino com múltiplos ciclos AL
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_history(all_histories: list[pd.DataFrame], path: str):
    """
    Plota loss e acurácia de todos os ciclos AL em um único gráfico.
    Linhas verticais separam os ciclos. Salva em path.
    """
    if not all_histories:
        return

    all_loss, all_val_loss = [], []
    all_acc,  all_val_acc  = [], []
    boundaries = []
    ep_offset  = 0

    for hist in all_histories:
        all_loss.extend(hist["loss"])
        all_val_loss.extend(hist["val_loss"])
        all_acc.extend(hist["accuracy"])
        all_val_acc.extend(hist["val_accuracy"])
        ep_offset += len(hist["loss"])
        boundaries.append(ep_offset)

    if not all_loss:
        return

    epochs = range(1, len(all_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss
    ax1.plot(epochs, all_loss,     label="Treino",    color="#1f77b4", linewidth=2)
    ax1.plot(epochs, all_val_loss, label="Validação", color="#ff7f0e", linestyle="--", linewidth=2)
    ax1.set_title("Loss ao longo dos ciclos AL")
    ax1.set_xlabel("Épocas totais"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Acurácia
    ax2.plot(epochs, all_acc,     label="Treino",    color="#2ca02c", linewidth=2)
    ax2.plot(epochs, all_val_acc, label="Validação", color="#d62728", linestyle="--", linewidth=2)
    ax2.set_title("Acurácia ao longo dos ciclos AL")
    ax2.set_xlabel("Épocas totais"); ax2.set_ylabel("Acurácia (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # Linhas de separação entre ciclos
    max_loss = max(max(all_loss), max(all_val_loss))
    min_acc  = min(min(all_acc),  min(all_val_acc))
    max_acc  = max(max(all_acc),  max(all_val_acc))
    prev     = 0

    for c_idx, boundary in enumerate(boundaries[:-1], start=1):
        ax1.axvline(x=boundary, color="black", linestyle=":", alpha=0.5)
        ax2.axvline(x=boundary, color="black", linestyle=":", alpha=0.5)
        mid = (prev + boundary) / 2
        ax1.text(mid, max_loss * 0.95, f"C{c_idx}", ha="center", fontsize=9, alpha=0.7, weight="bold")
        ax2.text(mid, min_acc + (max_acc - min_acc) * 0.05, f"C{c_idx}", ha="center", fontsize=9, alpha=0.7, weight="bold")
        prev = boundary

    # Último ciclo
    if boundaries:
        mid = (prev + boundaries[-1]) / 2
        ax1.text(mid, max_loss * 0.95, f"C{len(boundaries)}", ha="center", fontsize=9, alpha=0.7, weight="bold")
        ax2.text(mid, min_acc + (max_acc - min_acc) * 0.05, f"C{len(boundaries)}", ha="center", fontsize=9, alpha=0.7, weight="bold")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
