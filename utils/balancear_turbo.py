# TÁ BUGADO !!! TÁ BUGADO !!!
# Tem uns problemas de conversão para poder usar o "seq" e depois salvar
# Também tem que fazer downgrade do python para 3.5 por conta de um erro no "imgaug"
# O ideal é tentar arrumar o imgaug para funcionar com versão mais novas do python e numpy
#
# Site do imgaug: https://github.com/aleju/imgaug
#
# Dá para instalar com o pip install imgaug, mas tem que ser em um ambiente conda com o python 3.5


import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random
from torchvision import transforms

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)

#images_aug = seq(images=images)

# Código em python para usar aumento de dados
# para balancear um conjunto de imagens

# Função que recebe uma pasta e um fator de aumento
# e gera imagens aumentadas para balancear o conjunto
# usando pytorch

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

            # ESTE É O TRECHO BUGADO ... TEM QUE VER COMO FAZER AS CONVERSÕES CORRETAS
            # PARA PODER USAR O seq E SALVAR AS IMAGENS

            img, teste = get_random_image(label)

            img = seq(images=[img.numpy()])[0]

            # Converte para uint8
            img = img.astype(np.uint8)

            # Convertendo a imagem para o formato PIL
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


