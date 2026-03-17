#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py
=======
Ponto de entrada principal do experimento.

Fluxo:
  1. Lê argumentos de linha de comando (arquitetura, otimizador, LR, dobra)
  2. Carrega os dados via data_manager
  3. Executa N ciclos de Active Learning:
       a. Treina sobre o pool rotulado atual
       b. Seleciona as amostras mais incertas do pool não-rotulado
       c. Move as amostras selecionadas para o pool rotulado
  4. Avalia o melhor modelo no conjunto de teste
  5. Salva métricas, matriz de confusão, t-SNE e GradCAM

Uso:
  python main.py -a resnet18 -o adamw -l 0.0001 -r 1 -p completo

Parâmetros (-p):
  treino   → apenas treina, não avalia
  teste    → apenas avalia (requer modelo salvo)
  completo → treina e avalia (padrão)
"""

import os
import shutil
import torch
from torch.utils.data import Subset, DataLoader

from arch_optim import get_architecture, get_gradcam_layer
from data_manager import get_al_data
from hyperparameters import DATA_HYPERPARAMETERS, MODEL_HYPERPARAMETERS, ACTIVE_LEARNING
import helper_functions as hf
import dim_reduction


def main():
    args   = hf.get_args()
    device = MODEL_HYPERPARAMETERS["DEVICE"]

    print(f"\n{'='*55}")
    print(f"  EXPERIMENTO | {device.upper()}")
    print(f"  Arch: {args['architecture']}  |  Opt: {args['optimizer']}  |  LR: {args['learning_rate']}")
    print(f"{'='*55}")

    # ── Dados ────────────────────────────────────────────────────────────────
    train_aug, train_clean, idx_lbl, idx_unlbl, val_dl, test_dl = get_al_data()
    CLASSES = DATA_HYPERPARAMETERS["CLASSES"]

    # ── Balanceamento de classes ──────────────────────────────────────────────
    if hasattr(train_aug, "targets"):
        targets      = torch.tensor(train_aug.targets)
        counts       = torch.bincount(targets)
        counts[counts == 0] = 1
        weights      = 1.0 / torch.sqrt(counts.float())
        weights      = (weights / weights.sum() * len(CLASSES)).to(device)
        print("  Balanceamento de classes ativado.")
    else:
        weights = None

    # ── Modelo e loss ─────────────────────────────────────────────────────────
    model   = get_architecture(args["architecture"], 3, len(CLASSES), pretrained=True).to(device)
    loss_fn = hf.GuidedAttentionLoss(alpha=0.0001, class_weights=weights).to(device)

    base_name        = f"{args['run']}_{args['architecture']}_{args['optimizer']}"
    checkpoint_dir   = "../model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_histories   = []
    global_best_loss = float("inf")
    best_model_path  = os.path.join(checkpoint_dir, f"{base_name}_best.pth")

    # ── Loop de Active Learning ───────────────────────────────────────────────
    if args["procedure"] != "teste":
        n_cycles = ACTIVE_LEARNING["NUM_CYCLES"]

        for cycle in range(n_cycles):
            print(f"\n>>> Ciclo AL {cycle+1}/{n_cycles}  |  Pool rotulado: {len(idx_lbl)} imagens <<<")

            # Carrega checkpoint do ciclo anterior (se existir)
            if cycle > 0:
                prev_ckpt = os.path.join(checkpoint_dir, f"{base_name}_cycle_{cycle}.pth")
                if os.path.exists(prev_ckpt):
                    model.load_state_dict(torch.load(prev_ckpt, map_location=device))

            # Otimizador recriado a cada ciclo (LR parte do valor original)
            opt = _build_optimizer(args["optimizer"], model, args["learning_rate"])

            train_dl = DataLoader(
                Subset(train_aug, idx_lbl),
                batch_size=DATA_HYPERPARAMETERS["BATCH_SIZE"],
                shuffle=True,
                num_workers=DATA_HYPERPARAMETERS["NUM_WORKERS"],
                pin_memory=(device == "cuda"),
            )

            cycle_ckpt = os.path.join(checkpoint_dir, f"{base_name}_cycle_{cycle+1}.pth")

            hist = hf.fit(
                train_dl, val_dl, model, opt, loss_fn,
                epochs=MODEL_HYPERPARAMETERS["NUM_EPOCHS"],
                patience=MODEL_HYPERPARAMETERS["PATIENCE"],
                tolerance=MODEL_HYPERPARAMETERS["TOLERANCE"],
                save_path=cycle_ckpt,
            )

            # Mantém o melhor modelo global
            cycle_best = min(hist["val_loss"])
            if cycle_best < global_best_loss:
                global_best_loss = cycle_best
                if os.path.exists(cycle_ckpt):
                    shutil.copyfile(cycle_ckpt, best_model_path)

            all_histories.append(hist)

            # Salva gráfico de histórico acumulado
            os.makedirs("../results/history", exist_ok=True)
            hf.plot_combined_history(
                all_histories,
                f"../results/history/{base_name}_combined_cycles.png",
            )

            # Seleção de novas amostras para o próximo ciclo
            if cycle < n_cycles - 1 and len(idx_unlbl) > 0:
                if os.path.exists(cycle_ckpt):
                    model.load_state_dict(torch.load(cycle_ckpt, map_location=device))

                unlbl_dl = DataLoader(
                    Subset(train_clean, idx_unlbl),
                    batch_size=32, shuffle=False,
                    num_workers=DATA_HYPERPARAMETERS["NUM_WORKERS"],
                    pin_memory=(device == "cuda"),
                )
                n_new    = ACTIVE_LEARNING["SAMPLES_PER_CYCLE"]
                new_idxs = hf.get_uncertainty_samples(model, unlbl_dl, n_new)

                idx_lbl.extend([idx_unlbl[i] for i in new_idxs])
                for i in sorted(new_idxs, reverse=True):
                    idx_unlbl.pop(i)

                print(f"  +{n_new} amostras adicionadas ao pool. "
                      f"Pool rotulado: {len(idx_lbl)} | Não-rotulado: {len(idx_unlbl)}")

            # Remove checkpoint intermediário para economizar espaço
            if cycle > 0:
                old_ckpt = os.path.join(checkpoint_dir, f"{base_name}_cycle_{cycle}.pth")
                if os.path.exists(old_ckpt):
                    try:
                        os.remove(old_ckpt)
                    except OSError:
                        pass

    # ── Avaliação final ───────────────────────────────────────────────────────
    if args["procedure"] != "treino":
        print("\n" + "─" * 40)
        print("  AVALIAÇÃO FINAL")
        print("─" * 40)

        # Carrega o melhor modelo global
        fallback = os.path.join(checkpoint_dir, f"{base_name}_cycle_{ACTIVE_LEARNING['NUM_CYCLES']}.pth")
        ckpt_to_load = best_model_path if os.path.exists(best_model_path) \
                       else (fallback if os.path.exists(fallback) else None)

        if ckpt_to_load:
            model.load_state_dict(torch.load(ckpt_to_load, map_location=device))
            print(f"  Checkpoint carregado: {ckpt_to_load}")
        else:
            print("  Atenção: nenhum checkpoint encontrado — usando pesos do estado atual.")

        # Nome final dos arquivos (compatível com o script R)
        final_name = (
            f"{args['run']}_{args['architecture']}_"
            f"{args['optimizer']}_{args['learning_rate']}_MATRIX"
        )

        os.makedirs("../results/matrix", exist_ok=True)
        os.makedirs("../results_dl",     exist_ok=True)
        os.makedirs("../results/dr",     exist_ok=True)

        # Métricas + matriz de confusão
        prec, rec, f1 = hf.test(
            test_dl, model,
            f"../results/matrix/{final_name}.csv",
            f"../results/matrix/{final_name}.png",
            CLASSES,
        )
        print(f"  Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}")

        # Grava no CSV global de resultados
        with open("../results_dl/results.csv", "a") as f:
            f.write(f"{args['run']},{args['learning_rate']},"
                    f"{args['architecture']},{args['optimizer']},"
                    f"{prec},{rec},{f1}\n")

        # t-SNE
        try:
            feats, labs = dim_reduction.extract_features(model, test_dl, device)
            dim_reduction.plot_dimensionality_reduction(feats, labs, CLASSES, final_name, "t-SNE")
        except Exception as e:
            print(f"  [t-SNE] Pulado: {e}")

        # GradCAM
        try:
            layer = get_gradcam_layer(args["architecture"], model)
            if layer is not None:
                hf.save_gradcam_images(model, test_dl, layer, CLASSES, final_name)
        except Exception as e:
            print(f"  [GradCAM] Pulado: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# AUXILIAR — cria otimizador com nome dinâmico
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(name: str, model, lr: float):
    """Cria o otimizador pelo nome. Usado dentro do loop de ciclos AL."""
    import torch.optim as torch_optim
    mapping = {
        "sgd":     lambda: torch_optim.SGD(model.parameters(),   lr=lr, momentum=0.9, weight_decay=1e-4),
        "adam":    lambda: torch_optim.Adam(model.parameters(),  lr=lr, weight_decay=0),
        "adamw":   lambda: torch_optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4),
        "adagrad": lambda: torch_optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-4),
    }
    key = name.lower()
    if key in mapping:
        return mapping[key]()
    # Fallback para otimizadores externos (lion, sam) via optimizers.py
    from arch_optim import get_optimizer
    return get_optimizer(name, model, lr)


if __name__ == "__main__":
    main()
