#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optimizers.py
=============
Define os otimizadores disponíveis no experimento.

Como adicionar um novo otimizador:
1. Crie uma função seguindo o padrão:
       def meu_optim(params, learning_rate) -> Optimizer
2. Declare TODOS os hiperparâmetros explicitamente (mesmo os padrão),
   para facilitar a reprodução dos experimentos.
3. Registre no dicionário OPTIMIZERS em arch_optim.py.

Nota sobre SAM: por usar dois passos de gradiente, o loop de treino em
helper_functions.py tem tratamento especial para esse otimizador.
"""

from torch import optim
from lion_pytorch import Lion
from sam import SAM


# ─────────────────────────────────────────────────────────────────────────────
# OTIMIZADORES PADRÃO (torch.optim)
# ─────────────────────────────────────────────────────────────────────────────

def sgd(params, learning_rate: float):
    return optim.SGD(
        params=params,
        lr=learning_rate,
        momentum=0.9,       # momentum clássico — 0.9 é o valor mais comum em visão
        weight_decay=1e-4,
        nesterov=False,
    )


def adam(params, learning_rate: float):
    return optim.Adam(
        params=params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


def adamw(params, learning_rate: float):
    return optim.AdamW(
        params=params,
        lr=learning_rate,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=1e-4,  # weight decay separado do gradiente (diferença do Adam)
    )


def adagrad(params, learning_rate: float):
    return optim.Adagrad(
        params=params,
        lr=learning_rate,
        lr_decay=0,
        weight_decay=1e-4,
        eps=1e-10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OTIMIZADORES EXTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def lion(params, learning_rate: float):
    """
    Lion (EvoLved Sign Momentum) — Google Research, 2023.
    Mais eficiente em memória que Adam. Recomenda-se LR ~3-10x menor que Adam.
    Referência: https://arxiv.org/abs/2302.06675
    """
    return Lion(
        params=params,
        lr=learning_rate,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )


def sam(params, learning_rate: float):
    """
    Sharpness-Aware Minimization (SAM).
    ATENÇÃO: exige modificação no loop de treino (dois passos por batch).
    O helper_functions.fit() detecta automaticamente se o otimizador é SAM.
    Referência: https://arxiv.org/abs/2010.01412
    """
    optimizer = SAM(
        params=params,
        base_optimizer=optim.SGD,
        lr=learning_rate,
        momentum=0.9,
        rho=0.05,
    )
    optimizer.__name__ = "sam"
    return optimizer
