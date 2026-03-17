#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
arch_optim.py
=============
Registro central de arquiteturas, otimizadores e camadas de GradCAM.

Como adicionar um novo item:
  Arquitetura → adicione em ARCHITECTURES e GRADCAM_LAYERS
  Otimizador  → adicione em OPTIMIZERS (e implemente em optimizers.py)
"""

import architectures as arch
import optimizers as optim

# ─────────────────────────────────────────────────────────────────────────────
# ARQUITETURAS DISPONÍVEIS
# chave  : nome usado na linha de comando (-a <nome>)
# valor  : função em architectures.py com assinatura
#          fn(in_channels, out_classes, pretrained) -> nn.Module
# ─────────────────────────────────────────────────────────────────────────────
ARCHITECTURES = {
    # ── torchvision ──────────────────────────────────────────────────────────
    "alexnet":          arch.alexnet,
    "vgg19":            arch.vgg19,
    "resnet18":         arch.get_resnet18,
    "resnet50":         arch.get_resnet50,
    "resnet101":        arch.get_resnet101,
    "convnext_base":    arch.get_convnext_base,
    "densenet201":      arch.get_densenet201,
    "mobilenetv3":      arch.get_mobilenetV3,
    # ── timm ─────────────────────────────────────────────────────────────────
    "coat_tiny":                              arch.coat_tiny,
    "maxvit_rmlp_tiny_rw_256":               arch.maxvit_rmlp_tiny_rw_256,
    "lambda_resnet26rpt_256":                arch.get_lambda_resnet26rpt_256,
    "vit_relpos_base_patch32_plus_rpn_256":  arch.get_vit_relpos_base_patch32_plus_rpn_256,
    "sebotnet33ts_256":                       arch.get_sebotnet33ts_256,
    "lamhalobotnet50ts_256":                  arch.get_lamhalobotnet50ts_256,
    "swinv2_base_window16_256":               arch.get_swinv2_base_window16_256,
    "swinv2_cr_base_224":                     arch.get_swinv2_cr_base_224,
    # ── externas ─────────────────────────────────────────────────────────────
    "ielt":             arch.get_ielt,
    # ── siamese ──────────────────────────────────────────────────────────────
    "default_siamese":  arch.get_default_siamese,
}

# ─────────────────────────────────────────────────────────────────────────────
# CAMADAS PARA GRADCAM
# chave  : mesmo nome da arquitetura
# valor  : função fn(model) -> layer | None
#          Retorne None se GradCAM não for suportado para essa arquitetura.
# ─────────────────────────────────────────────────────────────────────────────
GRADCAM_LAYERS = {
    "alexnet":                                arch.get_alexnet_gradcam_layer,
    "vgg19":                                  arch.get_vgg19_gradcam_layer,
    "resnet18":                               arch.get_resnet18_gradcam_layer,
    "resnet50":                               arch.get_resnet50_gradcam_layer,
    "resnet101":                              arch.get_resnet101_gradcam_layer,
    "convnext_base":                          arch.get_convnext_base_gradcam_layer,
    "densenet201":                            arch.get_densenet201_gradcam_layer,
    "mobilenetv3":                            arch.get_mobilenetV3_gradcam_layer,
    "coat_tiny":                              arch.get_coat_tiny_gradcam_layer,
    "maxvit_rmlp_tiny_rw_256":               arch.get_maxvit_rmlp_tiny_rw_256_gradcam_layer,
    "lambda_resnet26rpt_256":                arch.get_lambda_resnet26rpt_256_gradcam_layer,
    "vit_relpos_base_patch32_plus_rpn_256":  arch.get_vit_relpos_base_patch32_plus_rpn_256_gradcam_layer,
    "sebotnet33ts_256":                       arch.get_sebotnet33ts_256_gradcam_layer,
    "lamhalobotnet50ts_256":                  arch.get_lamhalobotnet50ts_256_gradcam_layer,
    "swinv2_base_window16_256":               arch.get_swinv2_base_window16_256_gradcam_layer,
    "swinv2_cr_base_224":                     arch.get_swinv2_cr_base_224_gradcam_layer,
    "ielt":                                   arch.get_ielt_gradcam_layer,
    "default_siamese":                        arch.get_siamese_gradcam_layer,
}

# ─────────────────────────────────────────────────────────────────────────────
# OTIMIZADORES DISPONÍVEIS
# chave  : nome usado na linha de comando (-o <nome>)
# valor  : função em optimizers.py com assinatura
#          fn(params, learning_rate) -> Optimizer
# ─────────────────────────────────────────────────────────────────────────────
OPTIMIZERS = {
    "sgd":     optim.sgd,
    "adam":    optim.adam,
    "adamw":   optim.adamw,
    "adagrad": optim.adagrad,
    "lion":    optim.lion,
    "sam":     optim.sam,
}


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def get_architecture(name: str, in_channels: int, out_classes: int, pretrained: bool):
    """Instancia e retorna a arquitetura pelo nome (case-insensitive)."""
    key = name.casefold()
    if key not in ARCHITECTURES:
        raise ValueError(
            f"Arquitetura '{name}' não encontrada.\n"
            f"Disponíveis: {list(ARCHITECTURES.keys())}"
        )
    return ARCHITECTURES[key](in_channels=in_channels, out_classes=out_classes, pretrained=pretrained)


def get_optimizer(name: str, model, learning_rate: float):
    """Instancia e retorna o otimizador pelo nome (case-insensitive)."""
    key = name.casefold()
    if key not in OPTIMIZERS:
        raise ValueError(
            f"Otimizador '{name}' não encontrado.\n"
            f"Disponíveis: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[key](params=model.parameters(), learning_rate=learning_rate)


def get_gradcam_layer(name: str, model):
    """
    Retorna a camada alvo para GradCAM.
    Retorna None se a arquitetura não suportar GradCAM.
    """
    key = name.casefold()
    if key not in GRADCAM_LAYERS:
        return None
    return GRADCAM_LAYERS[key](model)
