import numpy as np
import os
import copy
import random
import time
import matplotlib.pyplot as plt
import tifffile as tiff
import skimage.io as skio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def svd(y):
    u, s, vt = np.linalg.svd(y, full_matrices=True)
    return u, s, vt

def number_of_svd_vectors(s, svd_threshold):
  # provides the number of svd vector that should be considered 
  # for a given percentage of the total sum
    scut = svd_threshold * sum(s)
    suma = 0.
    for ind, ss in enumerate(s):
        suma = suma + ss
        if suma >= scut:
            ind_result = ind
            break
    return ind_result

def reconstruction_image(u, s, vh, nvec):
  dim = vh.shape[0]
  s_app = np.zeros_like(s)
  for ind in range(nvec):
    s_app[ind] = s[ind]
  x_recons = np.dot(u *s_app, vh)
  return x_recons

def variance(s, number_of_components):
    var = s*s / (number_of_components-1)
    sumvar = np.sum(var)
    svd_cumsum = np.cumsum(var/sumvar)
    return var, svd_cumsum

def split_train_val_test(data, labels, train_ratio, val_ratio, test_ratio):
    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                            test_size= 1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, 
                            test_size = test_ratio/(test_ratio + val_ratio)) 
    return x_train, x_val, x_test, y_val, y_train, y_test

def split_data(train, val, test, data):
    ntrain = int(train*len(data))
    nval = int(val*len(data))
    ntest = int(test*len(data))
    x_train = np.zeros((ntrain, data.shape[1], data.shape[2]))
    x_val = np.zeros((nval, data.shape[1], data.shape[2]))
    x_test = np.zeros((ntest, data.shape[1], data.shape[2]))
    for i in range(len(data)):
        if i < ntrain :
            x_train[i, :, :] = data[i, :, :]
        else:
            if i<(ntrain+nval) :
                x_val[i-ntrain, :, :] = data[i, :, :]
            else:
                if i<ntrain+nval+ntest :
                    x_test[i-ntrain-nval, :, :] = data[i, :, :]
    return x_train, x_val, x_test

def plot_one_image_and_prediction(image, image_predict):
    #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    plt.subplot(1, 2, 1)
    plt.title('Original', fontsize = '15', fontweight = 'bold')
    plt.imshow(image, cmap='jet')
    plt.subplot(1, 2, 2)
    plt.title('Filtered', fontsize = '15', fontweight = 'bold')
    plt.imshow(image_predict, cmap='jet')

def plot_image_and_prediction(images, model_1, model_2, model_3, model_4, model_5, model_rand ):
    f, ax = plt.subplots(2, 4, sharex = True)
    ax[0,0].imshow(images, cmap='jet', aspect = 'auto')
    ax[0,1].imshow(model_1, cmap='jet', aspect = 'auto')
    ax[0,2].imshow(model_2, cmap='jet', aspect = 'auto')
    ax[0,3].imshow(model_3, cmap='jet', aspect = 'auto')
    ax[1,0].imshow(model_4, cmap='jet', aspect = 'auto')
    ax[1,1].imshow(model_5, cmap='jet', aspect = 'auto')
    ax[1,2].imshow(model_rand, cmap='jet', aspect = 'auto')

    ax[0,0].set_title('Original', fontweight = 'bold')
    ax[0,1].set_title('Model 1', fontweight = 'bold')
    ax[0,2].set_title('Model 2', fontweight = 'bold')
    ax[0,3].set_title('Model 3', fontweight = 'bold')
    ax[1,0].set_title('Model 4', fontweight = 'bold')
    ax[1,1].set_title('Model 5', fontweight = 'bold')
    ax[1,2].set_title('Model rand', fontweight = 'bold')
 
def moments_distribution(distrib):
    from scipy.stats import moment
    from scipy.stats import kurtosis, skew
    m0 = distrib.mean()
    kt = kurtosis(distrib.flatten(), fisher=True)
    sk = skew(distrib.flatten())
    return m0, kt, sk

def peak_signal_to_noise_ratio(filtered, original):
    # CALCULAMOS MEAN SQUARED ERROR (MSE)
    height = original.shape[0]
    width = original.shape[1]
    suma = 0
    for i in range (height):
        for j in range (width):
            suma = (filtered[i, j] - original[i, j])**2 + suma
            MSE = (1/(height*width))*suma

    maximum = filtered.max()
    #PSNR = 20*np.log10(maximum/((MSE)**(1/2)))
    PSNR = 10*np.log10((maximum**2)/(MSE))
    return PSNR

def plot_score(pathresults,history_keras):

    history_dict = history_keras.history
    training_cost=history_dict['loss']
    training_accuracy=history_dict['root_mean_squared_error']
    evaluation_cost=history_dict['val_loss']
    evaluation_accuracy=history_dict['val_root_mean_squared_error']

    epochs=len(evaluation_cost)
    xx = np.linspace(0,epochs-1,epochs)
    ##### FILE
    if not os.path.exists(pathresults):
        os.makedirs(pathresults)
    filename = "/loss_model5.dat"
    file = os.path.exists(pathresults + filename)
    if(file):
        os.remove(pathresults + filename)
        print(pathresults + filename + "removed")
    with open(pathresults + filename, 'w') as f1:
        for i in range (0, epochs):
            sumary= str(xx[i])+'\t'+str(evaluation_cost[i])+'\t'+str(evaluation_accuracy[i])+'\t'+str(training_cost[i])+'\t'+str(training_accuracy[i])+'\n'
            f1.write(sumary)
    f1.close()

    ##### FIGURE
    fig2, ax2 = plt.subplots(2,2, figsize=(10,10))
    ax2[0,0].plot(xx, evaluation_cost, color="red", label="evaluation cost")
    ax2[0,1].plot(xx, evaluation_accuracy, color="blue", label="evaluation rmse")
    ax2[1,0].plot(xx, training_cost, color="green", label="training cost")
    ax2[1,1].plot(xx, training_accuracy, color="orange", label="training rmse")
    txt = "$Epochs$"
    ax2[0,0].set_xlabel(txt)
    ax2[0,0].legend()
    ax2[0,1].set_xlabel(txt)
    ax2[0,1].legend()
    ax2[1,0].set_xlabel(txt)
    ax2[1,0].legend()
    ax2[1,1].set_xlabel(txt)
    ax2[1,1].legend()
    fig2.savefig(pathresults+"/epochs_evolution_model5.png", dpi=200, facecolor="#f1f1f1")

def recomposicion_imagenes(np_imagenes_50, num_subimg, np_subimagenes_filtradas):  # EL NÚMERO DE RECORTES SERÁ num_subimg AL CUADRADO

    image = np_imagenes_50
    print(image.shape)
    width = image.shape[2]
    height = image.shape[1]
    num_images = len(np_imagenes_50)
    cut_size = int(height / num_subimg)

    images = np.zeros((num_images, width, height))

    output_folder = "Reconstrucciones model conv2 4096x4096"
    index = 0
    for i in range(num_images):
        if i == 0:
            for j in range(num_subimg):
                for k in range(num_subimg):
                    if index != 4096:
                        # CALCULA LAS COORDENADAS DEL RECORTE
                        x_start = j * cut_size
                        x_end = (j + 1) * cut_size
                        y_start = k * cut_size
                        y_end = (k + 1) * cut_size
                        
                        # RECONSTRUYE LA IMAGEN INICIAL CON LOS RECORTES FILTRADOS
                        print(index)
                        images[i, y_start:y_end, x_start:x_end] = np_subimagenes_filtradas[index]
                        print(index)
                        index = index + 1
        # GUARDA LA IMAGEN EN UNA CARPETA (SI ESTA NO EXISTE LA CREA)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        tiff.imwrite(output_folder + "/" + "imagen_{}.tiff".format(i), images[i])
    
    images_reconstructed = images[:num_images, :, :]
    
    f = open("C:/Users/PORTATIL/Desktop/TFG/Prueba Convolucional/data_imagenes_reconstruidas_model_conv2_4096x4096.bin","wb")
    np.save(f,images_reconstructed)
    f.close()

def FourierTransform(images):
    image = images

    # Calcular la Transformada de Fourier 2D de la imagen
    f = np.fft.fft2(image)

    # Centrar la Transformada de Fourier
    fshift = np.fft.fftshift(f)

    # Calcular la magnitud de la Transformada de Fourier
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    f_section = np.mean(magnitude_spectrum, axis=1)
    print(f_section.shape)
    print(magnitude_spectrum.shape)
    # Crear un filtro pasa bajos con una ventana circular
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    radius = 140
    mask = np.zeros((rows, cols))
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask_area = x**2 + y**2 <= radius**2
    mask[mask_area] = 1

    # Aplicar el filtro a la Transformada de Fourier centrada
    fshift_filtered = fshift * mask

    # Calcular la Transformada de Fourier inversa del espectro filtrado
    f_filtered = np.fft.ifftshift(fshift_filtered)
    image_filtered = np.fft.ifft2(f_filtered)

    # Obtener la magnitud de la Transformada de Fourier de la imagen filtrada
    img_filtered_magnitude = 20*np.log(np.abs(fshift_filtered))

    # Calcular la Transformada de Fourier 2D de la imagen filtrada
    f_f = np.fft.fft2(image_filtered)

    # Centrar la Transformada de Fourier de la imagen filtrada
    fshift_f = np.fft.fftshift(f_f)

    # Calcular la magnitud de la Transformada de Fourier de la imagen filtrada
    magnitude_spectrum_f = 20*np.log(np.abs(fshift_f))
    f_section_f = np.mean(magnitude_spectrum_f, axis=0)

    # Visualizar la magnitud de la Transformada de Fourier
    plt.subplot(121),plt.imshow(image)
    plt.title('Imagen de entrada'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum)
    plt.title('Magnitud de la Transformada de Fourier'), plt.xticks([]), plt.yticks([])
    plt.show()

    #Visualizar el filtro Pasa-Baja
    plt.subplot(121),plt.imshow(mask)
    plt.title('Filtro Pasa-Baja ideal'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_filtered_magnitude)
    plt.title('Filtro Pasa-Baja sobre el espectro'), plt.xticks([]), plt.yticks([])
    plt.show()

    #Visualizar los espectros de frecuencia de la sección transversal de las imágenes
    plt.subplot(121), plt.plot(f_section)
    plt.title('Sección transversal de la magnitud de la Transformada de Fourier'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.plot(f_section_f)
    plt.title('Sección transversal de la magnitud de la TF filtrada'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Visualizar la imagen original y la imagen filtrada
    plt.subplot(121),plt.imshow(image[1000:2000, 1000:2000])
    plt.title('Imagen de entrada'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(np.abs(image_filtered[1000:2000, 1000:2000]))
    plt.title('Imagen filtrada'), plt.xticks([]), plt.yticks([])
    plt.show()

    #return f_section, magnitude_spectrum