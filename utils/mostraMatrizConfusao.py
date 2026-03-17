# Código para gerar matriz de confusão isoladas a partir de um arquivo .csv
# Usar quando precisar de algo diferente para o artigo
# Permite juntar as classes para outros tipos de análise
#
# Modo de uso:
# python3 mostraMatrizConfusao.py arquivo.csv [bloco]


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

# Nome do arquivo .csv
arquivo = "2_resnet18_sgd_0.01_MATRIX.csv"

# Tamanho do bloco para agrupar as classes
# 1 indica que não haverá agrupamento
bloco=4

if len(sys.argv) == 2:
   arquivo = sys.argv[1]

if len(sys.argv) == 3:
   arquivo = sys.argv[1]
   bloco = int(sys.argv[2])

# Lê o arquivo removendo a primeira linha
matriz = np.loadtxt(arquivo, delimiter=',', skiprows=1)

# Remove a primeira coluna do array
matriz = np.delete(matriz, 0, axis=1)

print(matriz)

# Se o tamanho do bloco não for um divisor do tamanho da matriz
# completa a matriz com zeros
if bloco > 1:
   if matriz.shape[0] % bloco != 0:
      matriz = np.vstack([matriz, np.zeros((bloco - matriz.shape[0] % bloco, matriz.shape[1]))])
   if matriz.shape[1] % bloco != 0:
      matriz = np.hstack([matriz, np.zeros((matriz.shape[0], bloco - matriz.shape[1] % bloco))])

# Se bloco for maior que 1, agrupa as classes
if bloco > 1:
   # Soma as linhas
   matriz = matriz.reshape(-1, bloco, matriz.shape[1]).sum(axis=1)
   # Soma as colunas
   matriz = matriz.reshape(matriz.shape[0], -1, bloco).sum(axis=2)

print(matriz)

# Salva em um .png a matriz como um mapa de calor colorido
# com os valores da matriz em cada célula do mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(matriz, annot=True, fmt="g", cmap='hot')
plt.savefig(arquivo + ".bloco_" + str(bloco) + ".png")
plt.show()

