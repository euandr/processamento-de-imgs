import mahotas
import pandas
import numpy
import glob
import cv2

labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'NEUTROPHIL']

features_list = list()

base_path = 'C:/Users/andre/Documents/atividade/teste/opencv/6-separando_ComBasenosNomes/imagens_etapa07/'

for label in labels:
    
    path = base_path + label

    images_list = glob.glob(path + '/*.png')

    for image_path in images_list:

        img = cv2.imread(image_path, 0)

        label = 1 if 'covid' in image_path else 0

        features_img = mahotas.features.haralick(img, compute_14th_feature = True, return_mean = True)

        features_img = numpy.append(features_img, label)

        features_list.append(features_img, '\n')

features_names = mahotas.features.texture.haralick_labels

features_names = numpy.append(features_names, 'Label')

saida_csv = 'C:/Users/andre/Documents/atividade/teste/opencv/07 - extração e classificação/dados.csv'

df = pandas.DataFrame(data = features_list, columns = features_names)



df.to_csv(saida_csv, index = False, sep = ';')

    
