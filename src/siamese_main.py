#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
siamese_main.py
===============
Pipeline de treinamento e avaliação da rede Siamese.

A rede Siamese possui dois cabeçalhos:
  - Reconhecimento: aprende a calcular distância entre embeddings (ContrastiveLoss)
  - Classificação:  classifica a imagem diretamente (CrossEntropyLoss)

Uso:
  python siamese_main.py -a resnet18 -o adamw -l 0.0001 -r 1 -p completo
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from arch_optim import ARCHITECTURES, OPTIMIZERS, get_architecture, get_optimizer
from data_manager import get_siamese_data
from hyperparameters import SIAMESE_DATA_HYPERPARAMETERS, SIAMESE_MODEL_HYPERPARAMETERS
import helper_functions as hf

torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULOS DA REDE SIAMESE
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    Loss contrastiva para aprendizado de similaridade.
    Pares positivos (mesma classe) → distância pequena.
    Pares negativos (classes diferentes) → distância maior que margin.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, target):
        # target = 1 → par positivo (mesma classe)
        # target = 0 → par negativo
        pos_loss = target       * distance.pow(2)
        neg_loss = (1 - target) * torch.clamp(self.margin - distance, min=0).pow(2)
        return torch.mean(pos_loss + neg_loss)


class L2Distance(nn.Module):
    def forward(self, a, b):
        return torch.nn.functional.pairwise_distance(a, b)


class SiameseNetwork(nn.Module):
    """
    Rede Siamese com cabeçalho duplo:
      - recognition_head: projeta embedding para comparação por distância
      - classification_head: classifica diretamente a imagem
    """

    def __init__(self, embedding_model: nn.Module, num_attributes: int, num_classes: int):
        super().__init__()
        self.embedding = embedding_model
        self.distance  = L2Distance()

        self.recognition_head = nn.Sequential(
            nn.Linear(num_attributes, num_attributes),
            nn.ReLU(),
            nn.Linear(num_attributes, num_attributes // 2),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(num_attributes, num_attributes),
            nn.ReLU(),
            nn.Linear(num_attributes, num_classes),
        )

    def forward(self, img_a, img_b):
        emb_a   = self.embedding(img_a)
        emb_b   = self.embedding(img_b)
        proj_a  = self.recognition_head(emb_a)
        proj_b  = self.recognition_head(emb_b)
        dist    = self.distance(proj_a, proj_b)
        cls_out = self.classification_head(emb_a)
        return dist, cls_out


# ─────────────────────────────────────────────────────────────────────────────
# TREINO SIAMESE
# ─────────────────────────────────────────────────────────────────────────────

def fit_siamese(
    train_dl_pairs, train_dl_cls, val_dl,
    model, opt_rec, opt_cls,
    loss_rec, loss_cls,
    epochs, patience, tolerance, save_path
) -> pd.DataFrame:
    """
    Loop de treino alternando entre:
      - Fase de reconhecimento (pares)
      - Fase de classificação (imagens individuais)
    """
    device = next(model.parameters()).device
    history = {"loss_rec": [], "loss_cls": [], "val_loss_cls": [], "val_acc": []}
    best_val = float("inf")
    no_improve = 0

    for ep in range(epochs):
        model.train()

        # ── Reconhecimento ────────────────────────────────────────────────────
        rec_loss_sum = 0.0
        for img_a, img_b, label in train_dl_pairs:
            img_a, img_b, label = img_a.to(device), img_b.to(device), label.to(device)
            opt_rec.zero_grad()
            dist, _ = model(img_a, img_b)
            loss     = loss_rec(dist, label)
            loss.backward()
            opt_rec.step()
            rec_loss_sum += loss.item()

        # ── Classificação ─────────────────────────────────────────────────────
        cls_loss_sum, correct, total = 0.0, 0, 0
        for batch in train_dl_cls:
            x, y = batch[0].to(device), batch[1].to(device)
            opt_cls.zero_grad()
            # Passa a mesma imagem duas vezes (forward exige dois inputs)
            _, cls_out = model(x, x)
            loss = loss_cls(cls_out, y)
            loss.backward()
            opt_cls.step()
            cls_loss_sum += loss.item()
            correct      += cls_out.argmax(1).eq(y).sum().item()
            total        += y.size(0)

        # ── Validação ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_corr, val_tot = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_dl:
                x, y = batch[0].to(device), batch[1].to(device)
                _, cls_out = model(x, x)
                val_loss += loss_cls(cls_out, y).item()
                val_corr += cls_out.argmax(1).eq(y).sum().item()
                val_tot  += y.size(0)

        val_loss /= max(len(val_dl), 1)
        val_acc   = 100.0 * val_corr / max(val_tot, 1)

        history["loss_rec"].append(rec_loss_sum / max(len(train_dl_pairs), 1))
        history["loss_cls"].append(cls_loss_sum / max(len(train_dl_cls), 1))
        history["val_loss_cls"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Ep {ep+1:03d} | rec={history['loss_rec'][-1]:.4f} "
              f"cls={history['loss_cls'][-1]:.4f} | "
              f"val={val_loss:.4f} acc={val_acc:.1f}%", end="")

        if val_loss < best_val - tolerance:
            best_val   = val_loss
            no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(" ✓")
        else:
            no_improve += 1
            print(f" [{no_improve}/{patience}]")
            if no_improve >= patience:
                print(f"  Early stopping na época {ep+1}.")
                break

    return pd.DataFrame(history)


def test_siamese(test_dl, model, path_csv, path_png, class_names):
    """Avalia a rede siamese pelo cabeçalho de classificação."""
    from helper_functions import test as _test
    import torch.nn as nn

    # Wrapper para usar o test() padrão com a SiameseNetwork
    class _ClsWrapper(nn.Module):
        def __init__(self, siamese):
            super().__init__()
            self.siamese = siamese

        def forward(self, x):
            _, cls_out = self.siamese(x, x)
            return cls_out

    wrapper = _ClsWrapper(model)
    return _test(test_dl, wrapper, path_csv, path_png, class_names)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = hf.get_args()
    device = SIAMESE_MODEL_HYPERPARAMETERS["DEVICE"]

    print(f"\n{'='*55}")
    print(f"  SIAMESE | {device.upper()}")
    print(f"  Arch: {args['architecture']}  Opt: {args['optimizer']}  LR: {args['learning_rate']}")
    print(f"{'='*55}")

    assert args["architecture"].casefold() in ARCHITECTURES, \
        f"Arquitetura '{args['architecture']}' não reconhecida."
    assert args["optimizer"].casefold() in OPTIMIZERS, \
        f"Otimizador '{args['optimizer']}' não reconhecido."

    # Modelo
    num_attrs   = SIAMESE_MODEL_HYPERPARAMETERS["NUM_ATTRIBUTES"]
    num_classes = SIAMESE_DATA_HYPERPARAMETERS["NUM_CLASSES"]

    embedding = get_architecture(
        args["architecture"], 3, num_attrs,
        pretrained=SIAMESE_MODEL_HYPERPARAMETERS["USE_TRANSFER_LEARNING"],
    )
    model = SiameseNetwork(embedding, num_attrs, num_classes).to(device)

    # Otimizadores
    lr      = args["learning_rate"]
    lr_rec  = lr / SIAMESE_MODEL_HYPERPARAMETERS.get("LR_SCALE_FACTOR", 20)
    opt_rec = get_optimizer(args["optimizer"], model, lr_rec)
    opt_cls = get_optimizer(args["optimizer"], model, lr)

    # Losses
    loss_rec = ContrastiveLoss(margin=SIAMESE_MODEL_HYPERPARAMETERS["MARGIN"])
    loss_cls = nn.CrossEntropyLoss()

    # Dados
    train_dl_pairs, train_dl_cls, val_dl, test_dl = get_siamese_data()

    model_name  = f"{args['run']}_siamese_{args['architecture']}_{args['optimizer']}_{args['learning_rate']}"
    save_path   = f"../model_checkpoints/{model_name}.pth"

    # Treino
    if args["procedure"] != "teste":
        history = fit_siamese(
            train_dl_pairs, train_dl_cls, val_dl,
            model, opt_rec, opt_cls,
            loss_rec, loss_cls,
            epochs=SIAMESE_MODEL_HYPERPARAMETERS["NUM_EPOCHS"],
            patience=SIAMESE_MODEL_HYPERPARAMETERS["PATIENCE"],
            tolerance=SIAMESE_MODEL_HYPERPARAMETERS["TOLERANCE"],
            save_path=save_path,
        )
        os.makedirs("../results/history", exist_ok=True)
        history.to_csv(f"../results/history/{model_name}_HISTORY.csv")

    # Avaliação
    if args["procedure"] != "treino":
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=device))

        os.makedirs("../results/matrix", exist_ok=True)
        prec, rec, f1 = test_siamese(
            test_dl, model,
            f"../results/matrix/{model_name}_MATRIX.csv",
            f"../results/matrix/{model_name}_MATRIX.png",
            SIAMESE_DATA_HYPERPARAMETERS["CLASSES"],
        )
        print(f"  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

        with open("../results_dl/results.csv", "a") as f:
            f.write(f"{args['run']},{args['learning_rate']},"
                    f"siamese_{args['architecture']},{args['optimizer']},"
                    f"{prec},{rec},{f1}\n")


if __name__ == "__main__":
    main()
