import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os 
import skimage.io as skio


def particion_imagenes(np_imagenes_50, num_subimg):  # EL NÚMERO DE RECORTES SERÁ num_subimg AL CUADRADO

    image = np_imagenes_50[0]
    width = image.shape[1]
    height = image.shape[0]
    num_images = len(np_imagenes_50) 
    cut_size = int(height / num_subimg)
    output_folder = "Recortes"

    subimages = np.zeros((num_images*num_subimg*num_subimg, cut_size, cut_size)) # AQUÍ CREO LOS ARRAYS DONDE GUARDARÉ LOS RECORTES Y EL INDICE CORRESPONDIENTE
    sublabels = np.zeros(num_images*num_subimg*num_subimg) 
    index_subimages = np.zeros(num_images*num_subimg*num_subimg)

    index = 0
    for i in range(num_images):
        image = np_imagenes_50[i]
        for j in range(num_subimg):
                for k in range(num_subimg):
                    # CALCULA LAS COORDENADAS DEL RECORTE
                    x_start = j * cut_size
                    x_end = (j + 1) * cut_size
                    y_start = k * cut_size
                    y_end = (k + 1) * cut_size
                    
                    # EXTRAE EL RECORTE Y AÑADE EL ÍNDICE DE LA IMAGEN PRINCIPAL
                    recorte = image[y_start:y_end, x_start:x_end]
                    subimages[index, :, :] = recorte
                    sublabels[index] = i
                    index_subimages[index] = index
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    tiff.imwrite(output_folder + "/" + "recorte{}_{}_{}_{}.tiff".format(i, index, x_start, y_start), recorte)
                    index = index + 1
    
    subimages_not_zero = subimages[:index, :, :]
    sublabels_not_zero = sublabels[:index]
    index_not_zero = index_subimages[:index]
    
    f = open("data_imagenes_recortadas.bin","wb")
    np.save(f,subimages_not_zero)
    np.save(f,sublabels_not_zero)
    np.save(f, index_not_zero)
    f.close()


def recomposicion_imagenes(np_imagenes_50, num_subimg, np_subimagenes_filtradas):  # EL NÚMERO DE RECORTES SERÁ num_subimg AL CUADRADO

    image = np_imagenes_50[0]
    width = image.shape[1]
    height = image.shape[0]
    num_images = len(np_imagenes_50) 
    cut_size = int(height / num_subimg)

    images = np.zeros((num_images, width, height))

    output_folder = "Reconstrucciones AutoencoderConv"
    index = 0
    for i in range(num_images):
        for j in range(num_subimg):
                for k in range(num_subimg):
                    # CALCULA LAS COORDENADAS DEL RECORTE
                    x_start = j * cut_size
                    x_end = (j + 1) * cut_size
                    y_start = k * cut_size
                    y_end = (k + 1) * cut_size
                    
                    # RECONSTRUYE LA IMAGEN INICIAL CON LOS RECORTES FILTRADOS
                    images[i, y_start:y_end, x_start:x_end] = np_subimagenes_filtradas[index]
                    index = index + 1
        # GUARDA LA IMAGEN EN UNA CARPETA (SI ESTA NO EXISTE LA CREA)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        tiff.imwrite(output_folder + "/" + "imagen_{}.tiff".format(i), images[i])
    
    images_reconstructed = images[:num_images, :, :]
    
    f = open("data_imagenes_reconstruidas_model1.bin","wb")
    np.save(f,images_reconstructed)
    f.close()

