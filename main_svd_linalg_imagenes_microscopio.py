#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 09:14:16 2023

@author: Luis
"""
import numpy as np
import copy
import random
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import GraphData as sgl
from subroutines_imagenes_microscopio import *
#from main_jorge_cortar_imagenes import *
###########################################################
f = open("data_images_4096x4096.bin","rb")
np_imagenes = np.load(f)
f.close()

#f = open("data_imagenes_recortadas.bin","rb")
#np_subimagenes = np.load(f)
#np_sublabels = np.load(f)
#f.close()



#plt.imshow(np_imagenes50[0][0:500, 0:500])
#plt.show()
#plt.imshow(np_images[0][0:500, 0:500])
#plt.show()

nimages, height, width = np_imagenes.shape

image_reduced = np.zeros((len(np_imagenes), height, width))
output_folder = 'Reduced SVD'

#imstack1 = skio.imread("C:/Users/PORTATIL/Desktop/TFG/Reconstrucciones/imagen_7.tiff", plugin="tifffile")
##imstack1 = np.moveaxis(imstack1, -1, 0)
#plt.imshow(imstack1[:,:])
#plt.show()

for i in range(len(np_imagenes)):
        if i == 0:
                image = np_imagenes[i]
                maximum = image.max()
                image = image / maximum

                number_of_components = np.min([width, height])

                #### SVD

                u, s, vt = svd(image)
                var, svd_cumsum = variance(s, number_of_components)
                #number_vector = number_of_svd_vectors(s, 0.6)
                #print(number_vector)
                image_reduced[i, :, :] = reconstruction_image(u, s, vt, 50)
                #if not os.path.exists(output_folder):
                #    os.makedirs(output_folder)
                #tiff.imwrite(output_folder + "/" + "imagen_{}.tiff".format(i), image_reduced[i])
                f = open("data_imagenes_reducidas_SVD.bin","wb")
                np.save(f,image_reduced)
                f.close()

                ##### Gráficos

               ##plot_one_image_and_prediction(image[990:1054, 1000:1064], image_reduced[i][990:1054, 1000:1064])
               ##plt.savefig('C:/Users/PORTATIL/Desktop/TFG/Figuras Definitivas/SVD provisional/SVD_comparison_jetmap_50.png')
               ##plt.show()

                #plt.imshow(image[448:512, 576:640], figsize = (8,8))
                #plt.show()
                #plt.imshow(image, figsize = (8,8))
                #plt.show()
                #plt.imshow(image_reduced[i][448:512, 576:640])
                #plt.show()
                ##plt.savefig('C:/Users/PORTATIL/Desktop/TFG/Figuras Definitivas/SVD provisional/SVD_comparison_50_total.png')
                #plt.show()
                #
                #index = list(range(number_of_components))
                #sgl.GraphData([[(index[0:]), var[0:]]], ['g'],\
                # ['SVD'], 'Variance', 'C:/Users/PORTATIL/Desktop/TFG/Figuras Definitivas/SVD provisional/Variance_{}.png'.format(i), Axx = '$Eigenvalue$ $index$', Axy = '$Explained$ $Variance$')

                #sgl.GraphData([[(index[0:]), svd_cumsum[0:]]], ['tab:olive'],['SVD'],
                #       'Explained Variance', 'C:/Users/PORTATIL/Desktop/TFG/Figuras Definitivas/SVD provisional/Varianza acumulada_{}.png'.format(i), Axx = '$Eigenvalue$ $index$',
                #        Axy = '$Expl.$ $Var.$')

                #error = image - image_reduced[i]
                #plt.hist([image.flatten(), image_reduced[i].flatten()], bins=100, label=['Original', 'Reduced'])
                #plt.title('Histogram', fontsize=30,fontweight = 'bold', loc = 'center')
                #plt.legend(loc='upper right')
                #plt.savefig('C:/Users/PORTATIL/Desktop/TFG/Figuras Definitivas/SVD provisional/Histograma_svd_{}.png'.format(i), bbox_inches='tight', facecolor="#f1f1f1")
                #plt.show()

                ########################################
               #x = []
               #y0 = []
               #curt = []
               #asim = []
               #curt_error = []
               #asim_error = []
               #mean_original = image.mean()
               #for j in range(0, 255, 5):
               #        x.append(j)
               #        imagen_reduced = reconstruction_image(u, s, vt, j)
               #        error = image - imagen_reduced
               #        m0_er, kt_er, sk_er = moments_distribution(error)
               #        curt_error.append(kt_er)
               #        asim_error.append(sk_er)
               #        m0, kt, sk = moments_distribution(imagen_reduced)
               #        y0.append(m0)
               #        curt.append(kt)
               #        asim.append(sk)

               #sgl.GraphData([[x, asim_error]], ['tab:green'],['SVD'],
               #       'Error Skewness', 'C:/Users/PORTATIL/Desktop/TFG/Figuras Definitivas/SVD provisional/Asimetría error_{}.png'.format(i), Axx = '$Vectors$',
               #        Axy = '$Skewness$')

               #sgl.GraphData([[x, curt_error]], ['tab:red'],['SVD'],
               #       'Error Kurtosis', 'C:/Users/PORTATIL/Desktop/TFG/Figuras Definitivas/SVD provisional/Curtosis error_{}.png'.format(i), Axx = '$Vectors$',
               #        Axy = '$Kurtosis$')
               #sgl.GraphData([[x, curt]], ['tab:olive'],['SVD'],
               #       'Prediction Kurtosis', 'C:/Users/PORTATIL/Desktop/TFG/Figuras Definitivas/SVD provisional/Curtosis_{}.png'.format(i), Axx = '$Vectors$',
               #        Axy = '$Kurtosis$')

               #sgl.GraphData([[x, asim]], ['tab:cyan'],['SVD'],
               #       'Prediction Skewness', 'C:/Users/PORTATIL/Desktop/TFG/Figuras Definitivas/SVD provisional/Asimetría_{}.png'.format(i), Axx = '$Vectors$',
               #        Axy = '$Skewness$')
# Generar una señal con ruido
#t = np.linspace(0, 1, 1000)
#signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)  # Señal original
#noise = np.random.normal(0, 0.5, 1000)  # Ruido aleatorio
#signal_with_noise = signal + noise

## Aplicar SVD a la matriz de señal con ruido
#U, S, V = np.linalg.svd(np_imagenes[0][448:512, 576:640])
#print(S)
#var = S*S / (64)
#print(var)
#sumvar = np.sum(var)
#print(sumvar)
#svd_cumsum = np.cumsum(var/sumvar)
#print(svd_cumsum)
#
## Filtrar el ruido truncando los valores singulares
#k = 3  # Número de valores singulares a conservar
#filtered_signal = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
## Graficar la señal original, la señal con ruido y la señal filtrada
#plt.imshow(np_imagenes[0][448:512, 576:640])
#plt.show()
#plt.imshow(filtered_signal)
#plt.show()
#plt.figure(figsize=(10, 6))
#plt.plot(t, signal_with_noise, label='Señal con ruido')
#plt.plot(t, signal, label='Señal original', color = 'k')
#plt.plot(t, filtered_signal, label='Señal filtrada', color = 'tab:olive')
#plt.legend()
#plt.xlabel('Tiempo')
#plt.ylabel('Amplitud')
#plt.title('Filtrado de ruido con SVD')
#plt.show()


#### PCA

#pca = PCA(n_components=5)
#pca.fit(image)
#image_low_dimension = pca.fit_transform(image)
#image_reduced2 = pca.inverse_transform(image_low_dimension)

#f = open("data_imagenes_reducidas.bin","rb")
#np_images = np.load(f)
#f.close()

#for i in range(len(np_imagenes50)):
#        plot_one_image_and_prediction(np_imagenes50[i][0:50, 0:50], np_images[i][0:50, 0:50])
#        plt.show()

        

#plot_one_image_and_prediction(images_test[ind], images_predict_test[ind])

# f = open("data_compressed_100_svd.bin","wb")
# np.save(f,images_predict_test)
# f.close()
