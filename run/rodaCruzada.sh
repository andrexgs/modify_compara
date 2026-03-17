#!/usr/bin/env bash
# rodaCruzada.sh
# ==============
# Executa a validação cruzada completa em N dobras.
#
# Pré-requisito: rodar utils/splitFolds.sh antes para criar as dobras.
#
# Uso:
#   ./rodaCruzada.sh          → usa N_DOBRAS padrão (5)
#   ./rodaCruzada.sh -k 10    → usa 10 dobras
#
# ─────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES=0       # GPU a usar (0 = primeira; mude para 1 se tiver duas)

N_DOBRAS=5                          # Número de dobras padrão
RODA_PADRAO=true                    # Roda o pipeline principal (main.py)
RODA_SIAMESE=false                  # Roda o pipeline siamese (siamese_main.py)

# "treino"   → apenas treina (não avalia)
# "teste"    → apenas avalia (requer modelos já treinados)
# "completo" → treina e avalia (padrão)
PROCEDIMENTO="completo"

# ─────────────────────────────────────────────────────────────────
# PARSE DE ARGUMENTOS
# ─────────────────────────────────────────────────────────────────
while getopts "k:" flag; do
    case "${flag}" in
        k) N_DOBRAS=${OPTARG} ;;
        *) echo "Uso: $0 [-k N_DOBRAS]"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────
# CAMINHOS
# ─────────────────────────────────────────────────────────────────
DIR_DOBRAS_IMGS="../data/dobras"
DIR_TREINO="../data/train"
DIR_TESTE="../data/test"
DIR_RESULTADOS="../results"
DIR_RESULTADOS_DOBRAS="../resultsNfolds"
DIR_RESULTADOS_FINAIS="../results_dl"

# ─────────────────────────────────────────────────────────────────
# INICIALIZAÇÃO
# ─────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════╗"
echo "  Validação Cruzada — ${N_DOBRAS} dobras"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}  |  Procedimento: ${PROCEDIMENTO}"
echo "╚══════════════════════════════════════════════╝"

# Gera lista de dobras: fold_1, fold_2, ..., fold_N
folds=()
for ((i=1; i<=N_DOBRAS; i++)); do folds+=("fold_${i}"); done

# Prepara diretórios de resultado
mkdir -p "$DIR_RESULTADOS_FINAIS"
rm -rf   "${DIR_RESULTADOS_FINAIS:?}"/*

if [ "$PROCEDIMENTO" != "teste" ]; then
    mkdir -p "../model_checkpoints"
    rm -rf   "../model_checkpoints"/*
fi

# Cabeçalho do CSV de resultados
echo "run,learning_rate,architecture,optimizer,precision,recall,fscore" \
    > "${DIR_RESULTADOS_FINAIS}/results.csv"

mkdir -p "$DIR_TREINO" "$DIR_TESTE" "$DIR_RESULTADOS" "$DIR_RESULTADOS_DOBRAS"
rm -rf "${DIR_RESULTADOS_DOBRAS:?}"/*

# ─────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL — uma iteração por dobra
# ─────────────────────────────────────────────────────────────────
for DOBRA_TESTE in "${folds[@]}"; do

    echo ""
    echo "══════════════════════════════════════════════"
    echo "  Dobra de teste: ${DOBRA_TESTE}"
    echo "══════════════════════════════════════════════"

    # Limpa e reconstrói treino/teste para esta dobra
    rm -rf "${DIR_TREINO:?}"/*
    rm -rf "${DIR_TESTE:?}"/*
    rm -rf "${DIR_RESULTADOS:?}"/*

    # Copia a dobra atual para teste
    cp -R "${DIR_DOBRAS_IMGS}/${DOBRA_TESTE}/"* "${DIR_TESTE}/"

    # Copia todas as outras dobras para treino
    for OUTRA in "${folds[@]}"; do
        if [ "$OUTRA" != "$DOBRA_TESTE" ]; then
            echo "  + Treino: ${OUTRA}"
            cp -R "${DIR_DOBRAS_IMGS}/${OUTRA}/"* "${DIR_TREINO}/"
        fi
    done

    # Prepara pastas de resultado intermediário
    mkdir -p "${DIR_RESULTADOS}/history"
    mkdir -p "${DIR_RESULTADOS}/matrix"
    mkdir -p "${DIR_RESULTADOS}/dr"

    # Número da dobra (ex: "fold_3" → "3")
    NUM_DOBRA="${DOBRA_TESTE#fold_}"

    # Executa todas as combinações de arch/opt/lr para esta dobra
    bash ./roda.sh "$NUM_DOBRA" "$RODA_PADRAO" "$RODA_SIAMESE" "$PROCEDIMENTO"

    # Move resultados para a pasta permanente desta dobra
    mkdir -p "${DIR_RESULTADOS_DOBRAS}/${DOBRA_TESTE}"
    mv "${DIR_RESULTADOS}/"* "${DIR_RESULTADOS_DOBRAS}/${DOBRA_TESTE}/"

done

# ─────────────────────────────────────────────────────────────────
# PÓS-PROCESSAMENTO — estatísticas e boxplots (via R)
# ─────────────────────────────────────────────────────────────────
if [ "$PROCEDIMENTO" != "treino" ]; then
    echo ""
    echo "  Gerando estatísticas e gráficos (R)..."
    cd ../src || exit 1
    Rscript ./graphics.R
    cd ../run || exit 1
fi

echo ""
echo "  Concluído. Resultados em: ${DIR_RESULTADOS_FINAIS}/"
