#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
architectures.py
================
Define todas as arquiteturas de redes disponíveis.

Como adicionar uma nova arquitetura:
1. Crie uma função seguindo o padrão:
       def minha_rede(in_channels, out_classes, pretrained) -> nn.Module
2. Ajuste a primeira camada (in_channels) e a última (out_classes)
3. Registre no dicionário `ARCHITECTURES` em arch_optim.py
4. (Opcional) Crie uma função get_<nome>_gradcam_layer(model)
   e registre em `GRADCAM_LAYERS` em arch_optim.py
"""

import os
import wget
import timm
import numpy as np
from torch import nn
from torchvision import models

from hyperparameters import DATA_HYPERPARAMETERS
from IELT.models.vit import get_b16_config
from IELT.models.IELT import InterEnsembleLearningTransformer


# ─────────────────────────────────────────────────────────────────────────────
# REDES DO TORCHVISION
# ─────────────────────────────────────────────────────────────────────────────

def alexnet(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if pretrained else None)
    model.features[0]   = nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, out_classes)
    return model

def get_alexnet_gradcam_layer(model):
    return model.features[-3]


def vgg19(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT if pretrained else None)
    model.features[0]   = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, out_classes)
    return model

def get_vgg19_gradcam_layer(model):
    return model.features[-3]


def get_resnet18(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc    = nn.Linear(512, out_classes)
    return model

def get_resnet18_gradcam_layer(model):
    return model.layer4[1].conv2


def get_resnet50(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc    = nn.Linear(2048, out_classes)
    return model

def get_resnet50_gradcam_layer(model):
    return model.layer4[2].conv3


def get_resnet101(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc    = nn.Linear(2048, out_classes)
    return model

def get_resnet101_gradcam_layer(model):
    return model.layer4[2].conv3


def get_convnext_base(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    model = models.convnext_base(
        weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
    )
    model.features[0][0] = nn.Conv2d(in_channels, 128, kernel_size=4, stride=4)
    model.classifier[2]  = nn.Linear(1024, out_classes)
    return model

def get_convnext_base_gradcam_layer(model):
    return model.features[7][2].block[0]


def get_densenet201(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    model = models.densenet201(
        weights=models.DenseNet201_Weights.DEFAULT if pretrained else None
    )
    model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # DenseNet201 usa classifier, não fc
    model.classifier = nn.Linear(1920, out_classes)
    return model

def get_densenet201_gradcam_layer(model):
    return model.features.denseblock4.denselayer32.conv2


def get_mobilenetV3(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    )
    model.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[3]  = nn.Linear(1280, out_classes)
    return model

def get_mobilenetV3_gradcam_layer(model):
    return model.features[-1][0]


# ─────────────────────────────────────────────────────────────────────────────
# REDES DO TIMM
# ─────────────────────────────────────────────────────────────────────────────

def maxvit_rmlp_tiny_rw_256(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    """Multi-Axis Vision Transformer — https://arxiv.org/abs/2204.01697"""
    return timm.create_model(
        "maxvit_rmlp_tiny_rw_256",
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=out_classes,
    )

def get_maxvit_rmlp_tiny_rw_256_gradcam_layer(model):
    return model.stages[3].blocks[1].conv.conv3_1x1


def coat_tiny(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    """Co-Scale Conv-Attentional Transformers — https://arxiv.org/abs/2104.06399"""
    return timm.create_model(
        "coat_tiny",
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=out_classes,
    )

def get_coat_tiny_gradcam_layer(model):
    return model.parallel_blocks[5].factoratt_crpe4.crpe.conv_list[2]


def get_lambda_resnet26rpt_256(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(
        "lambda_resnet26rpt_256",
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=out_classes,
    )

def get_lambda_resnet26rpt_256_gradcam_layer(model):
    return model.stages[3][1].conv1_1x1.conv


def get_vit_relpos_base_patch32_plus_rpn_256(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(
        "vit_relpos_base_patch32_plus_rpn_256",
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=out_classes,
    )

def get_vit_relpos_base_patch32_plus_rpn_256_gradcam_layer(model):
    # GradCAM não suportado para ViTs sem adaptação extra
    return None


def get_sebotnet33ts_256(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(
        "sebotnet33ts_256",
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=out_classes,
    )

def get_sebotnet33ts_256_gradcam_layer(model):
    return model.final_conv.conv


def get_lamhalobotnet50ts_256(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(
        "lamhalobotnet50ts_256",
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=out_classes,
    )

def get_lamhalobotnet50ts_256_gradcam_layer(model):
    return model.stages[3][2].conv3_1x1.conv


def get_swinv2_base_window16_256(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(
        "swinv2_base_window16_256",
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=out_classes,
    )

def get_swinv2_base_window16_256_gradcam_layer(model):
    # GradCAM não suportado nativamente para Swin Transformers
    return None


def get_swinv2_cr_base_224(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(
        "swinv2_cr_base_224",
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=out_classes,
    )

def get_swinv2_cr_base_224_gradcam_layer(model):
    return None


# ─────────────────────────────────────────────────────────────────────────────
# IELT (arquitetura externa, não é do torchvision nem do timm)
# ─────────────────────────────────────────────────────────────────────────────

def get_ielt(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    """
    Inter-Ensemble Learning Transformer.
    Baixa os pesos pré-treinados do Google Storage na primeira execução.
    """
    weights_dir  = "./IELT/pretrained"
    weights_path = os.path.join(weights_dir, "ViT-B_16.npz")

    os.makedirs(weights_dir, exist_ok=True)

    if not os.path.exists(weights_path):
        url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"
        print(f"[IELT] Baixando pesos pré-treinados de {url} ...")
        wget.download(url, weights_path)

    config = get_b16_config()
    model  = InterEnsembleLearningTransformer(
        config,
        img_size=DATA_HYPERPARAMETERS["IMAGE_SIZE"],
        num_classes=out_classes,
    )

    if pretrained:
        model.load_from(np.load(weights_path))

    model.embeddings.patch_embeddings = nn.Conv2d(in_channels, 768, kernel_size=16, stride=16)
    model.softmax = nn.Identity()
    return model

def get_ielt_gradcam_layer(model):
    # GradCAM não suportado para IELT
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIAMESE (rede de embedding para aprendizado métrico)
# ─────────────────────────────────────────────────────────────────────────────

def get_default_siamese(in_channels: int, out_classes: int, pretrained: bool) -> nn.Module:
    """
    Rede de embedding simples para uso na SiameseNetwork.
    Recebe imagens e produz vetores de features (out_classes = dimensão do embedding).
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, 64,  kernel_size=10),  nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64,  128, kernel_size=7),  nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 128, kernel_size=4),  nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=4),  nn.ReLU(),
        nn.Flatten(),
        nn.Linear(9216, out_classes),
    )

def get_siamese_gradcam_layer(model):
    return None
