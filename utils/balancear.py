# Código em python para usar aumento de dados
# para balancear um conjunto de imagens

# Função que recebe uma pasta e um fator de aumento
# e gera imagens aumentadas para balancear o conjunto
# usando pytorch

import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import random

# Além de igualar com a classe majoritária, é possível
# aumentar todas as classes. No exempo, estamos usando
# 1 para aumentar todas as classes em 100%.
# Troque por 0 para apenas balancear sem aumentar
# o total da classe majoritária
percentual_aumento = 0

# Verifica se passado um parâmetro -p n para aumentar todas as classes
# e usa o valor de n como percentual de aumento
import sys
if len(sys.argv) > 1:
    if sys.argv[1] == '-p':
        percentual_aumento = int(sys.argv[2])/100


def aumentar_dados(pasta,pasta_aumentada,fold):

    print('Balanceando dobra ' + str(fold) + '...')
    # Transformações para aumentar os dados
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
#        transforms.RandomRotation(30),
#        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor()
    ])

    # Carregar o conjunto de dados
    dataset = ImageFolder(pasta, transform=transform)

    # Carregar o conjunto de dados no DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Criar a pasta para salvar as imagens aumentadas
    pasta_aumentada = pasta_aumentada+'/'+'fold_'+str(fold)
    os.makedirs(pasta_aumentada)


    # Número de imagens em cada classe
    classes = dataset.classes
    n_classes = len(classes)
    n_imgs = [0] * n_classes

    # Contar o número de imagens em cada classe
    for img, label in loader:
        n_imgs[label] += 1

    # Fator de aumento para cada classe necessário
    # para igualar a quantidade da maior classe
    fator_por_classe = [max(n_imgs) - n for n in n_imgs]

    novas_imagens = int(max(n_imgs) * percentual_aumento)

    # Contador que será usado nos nomes dos arquivos
    n_imgs_aumentadas = [0] * n_classes



    # Crie um dicionário para armazenar os índices de cada classe
    indices_por_classe = {i: [] for i in range(n_classes)}

    # Preencha o dicionário com os índices de cada classe
    for i, (_, label) in enumerate(dataset):
        indices_por_classe[label].append(i)

    # Função para obter uma imagem aleatória de uma classe específica
    def get_random_image(class_index):
        # Obtenha um índice aleatório para a classe especificada
        random_index = random.choice(indices_por_classe[class_index])
    
        # Use esse índice para obter a imagem correspondente do conjunto de dados
        image, label = dataset[random_index]
    
        return image, label

    # Aumentar as imagens
    for label in range(n_classes):
        print('   Aumentando classe ' + classes[label] + '...')
        for i in range(fator_por_classe[label]+novas_imagens):
            img, teste = get_random_image(label)
            # Transform to PIL image
            img = img.squeeze(0)
            img = transforms.ToPILImage()(img)

            # Aplica transformações à imagem
            img = transform(img)
            # Salva a imagem aumentada
            img = img.squeeze(0)
            img = transforms.ToPILImage()(img)

            pasta_da_classe = pasta_aumentada + '/' + classes[label]

            # Cria pasta da classe se ela não existir
            if not os.path.exists(pasta_da_classe):
                os.makedirs(pasta_da_classe)

            img.save(pasta_da_classe + '/' + 'SINTETICA_F'+str(fold)+'_'+str(n_imgs_aumentadas[label]) + '.jpg')
            n_imgs_aumentadas[label] += 1
        print('   Criou mais ' + str(fator_por_classe[label]+novas_imagens) + ' imagens para esta classe')



# Rodar aumentar_dados em cada uma das dobras de
# validação cruzada que estão na pasta "../dados/dobras"
# e salvar as imagens aumentadas na pasta "../dados/dobras_aumentadas"

# Conta o total de subpastas dentro de uma pasta
# sem contar subpastas recursivamente
def contar_arquivos(pasta):
    return len([name for name in os.listdir(pasta) if os.path.isdir(os.path.join(pasta, name))])

totalDobras = contar_arquivos('../data/dobras/')

# Apaga a pasta de dobras sintéticas se ela já existir
if os.path.exists('../data/dobras_sinteticas'):
    os.system('rm -r ../data/dobras_sinteticas')

# Cria a pasta de dobras sintéticas
os.makedirs('../data/dobras_sinteticas')

for i in range(totalDobras):
    aumentar_dados('../data/dobras/fold_' + str(i+1), '../data/dobras_sinteticas/', i+1)
    print('Dobra ' + str(i+1) + ' aumentada com sucesso!')







