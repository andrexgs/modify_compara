#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hyperparameters.py
==================
Arquivo CENTRAL de configuração do experimento.

Mexa AQUI para:
- Mudar o tamanho das imagens, batch size, número de épocas, etc.
- Ativar/desativar augmentation e suas intensidades
- Escolher o device (cpu/cuda)
"""

import os
import torch

# ─────────────────────────────────────────────────────────────────────────────
# CAMINHOS (ajustados automaticamente, não precisa mexer)
# ─────────────────────────────────────────────────────────────────────────────
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

# Suporte ao Google Colab: se /content/data existir, usa esse caminho
ROOT_DATA_DIR = "/content/data" if os.path.exists("/content/data") \
                else os.path.join(_PROJECT_DIR, "data")

TRAIN_DATA_DIR = os.path.join(ROOT_DATA_DIR, "train")
TEST_DATA_DIR  = os.path.join(ROOT_DATA_DIR, "test")

# Lê as classes automaticamente a partir das subpastas de treino
CLASSES: list[str] = []
if os.path.exists(TRAIN_DATA_DIR):
    CLASSES = sorted([
        d for d in os.listdir(TRAIN_DATA_DIR)
        if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))
    ])

# ─────────────────────────────────────────────────────────────────────────────
# DADOS
# ─────────────────────────────────────────────────────────────────────────────
DATA_HYPERPARAMETERS = {
    # Tamanho para redimensionar todas as imagens (altura e largura iguais)
    # Atenção: algumas arquiteturas exigem tamanhos específicos (ex: coat_tiny → 224)
    "IMAGE_SIZE": 256,

    # Número de imagens por lote de treino
    # Reduza se a GPU ficar sem memória (ex: 8 para MaxViT em GPU pequena)
    "BATCH_SIZE": 16,

    # Fração do treino reservada para validação
    "VAL_SPLIT": 0.2,

    # Ativa/desativa o data augmentation durante o treino
    "USE_DATA_AUGMENTATION": True,

    # Workers para carregamento paralelo de imagens
    # 8 é um bom valor para Linux; reduza para 0 se tiver erros de DataLoader
    "NUM_WORKERS": 8,

    # Caminhos (preenchidos automaticamente acima)
    "ROOT_DATA_DIR":  ROOT_DATA_DIR,
    "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
    "TEST_DATA_DIR":  TEST_DATA_DIR,
    "CLASSES":        CLASSES,
    "NUM_CLASSES":    len(CLASSES),
    "IN_CHANNELS":    3,
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# Só tem efeito se DATA_HYPERPARAMETERS["USE_DATA_AUGMENTATION"] == True
# ─────────────────────────────────────────────────────────────────────────────
DATA_AUGMENTATION = {
    "HORIZONTAL_FLIP": 0.5,   # Probabilidade de flip horizontal
    "VERTICAL_FLIP":   0.5,   # Probabilidade de flip vertical
    "ROTATION":        90,    # Graus máximos de rotação aleatória
    "COLOR_JITTER":    0.1,   # Intensidade de variação de brilho/cor (0 = desativado)
}

# ─────────────────────────────────────────────────────────────────────────────
# MODELO / TREINO
# ─────────────────────────────────────────────────────────────────────────────
MODEL_HYPERPARAMETERS = {
    # Número máximo de épocas por ciclo de Active Learning
    "NUM_EPOCHS": 5,

    # Early stopping: para o treino se não melhorar por N épocas
    "PATIENCE": 1,

    # Melhoria mínima na val_loss para considerar progresso
    "TOLERANCE": 0.001,

    # Usar pesos pré-treinados (ImageNet) — fortemente recomendado
    "USE_TRANSFER_LEARNING": True,

    # Device: detectado automaticamente (cuda se disponível, senão cpu)
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE LEARNING
# Controla o loop de seleção iterativa de amostras
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_LEARNING = {
    # Número de ciclos de AL (a cada ciclo, novas amostras são selecionadas)
    "NUM_CYCLES": 5,

    # Quantidade de amostras rotuladas no início (ciclo 0)
    # Se o dataset for pequeno, use None para começar com 100% dos dados
    "INITIAL_LABELED": 100,

    # Amostras selecionadas por ciclo (as mais incertas do pool)
    "SAMPLES_PER_CYCLE": 50,
}

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE SIAMESE
# Usado apenas por siamese_main.py
# ─────────────────────────────────────────────────────────────────────────────
SIAMESE_DATA_HYPERPARAMETERS = {
    "IMAGE_SIZE":       256,
    "BATCH_SIZE_REC":   16,    # Batch do cabeçalho de reconhecimento (pares)
    "BATCH_SIZE_CLS":   16,    # Batch do cabeçalho de classificação
    "VAL_SPLIT":        0.2,
    "USE_DATA_AUGMENTATION": True,
    "ROOT_DATA_DIR":  ROOT_DATA_DIR,
    "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
    "TEST_DATA_DIR":  TEST_DATA_DIR,
    "CLASSES":        CLASSES,
    "NUM_CLASSES":    len(CLASSES),
    "IN_CHANNELS":    3,
    "CLASS_SAMPLE_SIZE": 30,   # Amostras por classe para geração de pares
}

SIAMESE_MODEL_HYPERPARAMETERS = {
    "NUM_EPOCHS":            5,
    "PATIENCE":              2,
    "TOLERANCE":             0.001,
    "USE_TRANSFER_LEARNING": False,
    "NUM_ATTRIBUTES":        512,   # Dimensão do vetor de embedding
    "DEVICE":                "cuda" if torch.cuda.is_available() else "cpu",
    "MARGIN":                1.0,   # Margem para a ContrastiveLoss
    "THRESHOLD":             0.5,   # Distância para classificar como "mesma classe"
    "LR_SCALE_FACTOR":       20,    # LR do reconhecimento = LR / LR_SCALE_FACTOR
}
