#!/usr/bin/env bash
# roda.sh
# =======
# Executa todas as combinações de arquitetura × otimizador × LR
# para uma dobra específica.
#
# Parâmetros posicionais (passados por rodaCruzada.sh):
#   $1 = número da dobra (ex: 1, 2, 3...)
#   $2 = true/false — rodar pipeline padrão
#   $3 = true/false — rodar pipeline siamese
#   $4 = procedimento: treino | teste | completo
#
# ─────────────────────────────────────────────────────────────────
# CONFIGURE AQUI: arquiteturas, otimizadores e learning rates
# ─────────────────────────────────────────────────────────────────

ARQUITETURAS=(
    resnet18
    vgg19
    #vit_relpos_base_patch32_plus_rpn_256
    #maxvit_rmlp_tiny_rw_256
    # alexnet
    # coat_tiny
    # convnext_base
    # densenet201
    # mobilenetv3
    # resnet50
    # resnet101
    # ielt
)

OTIMIZADORES=(
    sgd
    #adagrad
    #adamw
    # adam
    # lion
    # sam
)

LEARNING_RATES=(
    0.0001
)

# ─────────────────────────────────────────────────────────────────
# EXECUÇÃO (não precisa mexer abaixo)
# ─────────────────────────────────────────────────────────────────
DOBRA=$1
RODA_PADRAO=${2:-true}
RODA_SIAMESE=${3:-false}
PROCEDIMENTO=${4:-completo}

cd ../src || exit 1

for lr in "${LEARNING_RATES[@]}"; do
    for arch in "${ARQUITETURAS[@]}"; do
        for opt in "${OTIMIZADORES[@]}"; do

            if [ "$RODA_PADRAO" = "true" ]; then
                echo ""
                echo "▶ Iniciando: arch=${arch}  opt=${opt}  lr=${lr}  dobra=${DOBRA}"
                python3 main.py \
                    -a "$arch" -o "$opt" -l "$lr" -r "$DOBRA" -p "$PROCEDIMENTO" \
                    > >(tee -a "../results/${arch}_${opt}_${lr}.output") \
                    2> >(tee    "../results/error_log_${arch}_${opt}_${lr}.txt" >&2)
            fi

            if [ "$RODA_SIAMESE" = "true" ]; then
                echo ""
                echo "▶ [Siamese] arch=${arch}  opt=${opt}  lr=${lr}  dobra=${DOBRA}"
                python3 siamese_main.py \
                    -a "$arch" -o "$opt" -l "$lr" -r "$DOBRA" -p "$PROCEDIMENTO" \
                    > >(tee -a "../results/siamese_${arch}_${opt}_${lr}.output") \
                    2> >(tee    "../results/error_log_siamese_${arch}_${opt}_${lr}.txt" >&2)
            fi

        done
    done
done

cd ../run || exit 1
