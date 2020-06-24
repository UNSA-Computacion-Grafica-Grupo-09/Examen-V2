import cv2
import numpy as np
import math


#########################  EJEMPLO ############################
# Pide una imagen y devuelve una imagen
def algoritmo_dummy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


##############################################################

def threholding(image):
    imagen = image.copy()
    resultado = image.copy()

    height, width, chanels = imagen.shape
    limi = 200  # limite inicial
    limf = 230  # limite final

    for i in range(height):
        for j in range(width):
            if (imagen[i][j][0] > limi or imagen[i][j][1] > limf or imagen[i][j][2] < limi):
                resultado[i][j] = 0

    return resultado  # importante


def constrast_streching(image):
    imagen_original = image.copy()  # importante
    imagen_resultado = image.copy()
    # Detallamos los valores de las variables de Contrast stretching
    a = 0  # límite inferior
    b = 255  # límite superior
    c = np.min(imagen_original)  # El menor valor de los pixeles
    d = np.max(imagen_original)  # El mayor valor de los pixeles

    alto, ancho, canales = imagen_original.shape

    def point_operator(pixel_RGB):
        return (pixel_RGB - c) * ((b - a) / (d - c) + a)

    for x in range(alto):
        for y in range(ancho):
            imagen_resultado[x][y] = point_operator(imagen_original[x][y])

    return imagen_resultado  # importante


def constrast_streching_out(image):
    res = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)  # importante
    imagen_original = image.copy()

    # Detallamos los valores de las variables de Contrast stretching
    a = 0  # límite inferior
    b = 255  # límite superior
    # c = 69
    # d = 149
    c = np.min(imagen_original)  # El menor valor de los pixeles
    d = np.max(imagen_original)  # El mayor valor de los pixeles

    longi = d - c  # calculamos la longitud del rango
    limitec = (longi * 5) / 100  # calculamos el limite a partir del porcentaje

    newc = c - limitec  # El menor valor  en un limite de 5%

    limited = (longi * 5) / 100  # calculamos el limite a partir del porcentaje
    newd = d + limited

    alto, ancho, canales = imagen_original.shape

    for x in range(alto):
        for y in range(ancho):
            re = (res[x][y] - newc) * ((b - a) / (newd - newc) + a)
            if (re < 0):
                res[x][y] = 0
            elif (re > 255):
                res[x][y] = 255
            else:
                res[x][y] = re

    return res  # importante


def histogram_equalization(image):
    imagen_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    cantidad_pixeles = imagen_gray.size
    shape = imagen_gray.shape
    height = shape[0]
    width = shape[1]

    L = 256
    S_n = []
    imagen_array1D = imagen_gray.flatten().tolist()

    suma_acumulada = 0

    # Realizamos S_n
    for index in range(L):
        P_n = imagen_array1D.count(index) / cantidad_pixeles
        suma_acumulada = suma_acumulada + P_n
        s_k = int(round(suma_acumulada * (L - 1)))
        S_n.append(s_k)

    # Realizamos el mapeo lineal
    for index in range(cantidad_pixeles):
        imagen_array1D[index] = S_n[imagen_array1D[index]]

    img_result = np.asarray(imagen_array1D)
    img_result = img_result.reshape(height, width)

    return img_result


def operador_logaritmico(image):
    imagen_resultado = image.copy()
    imagen_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    c = 255 / np.log10(1 + np.max(imagen_gray))
    alto, ancho = imagen_gray.shape

    for x in range(alto):
        for y in range(ancho):
            imagen_resultado[x][y] = c * (np.log10(1 + imagen_gray[x][y]))

    return imagen_resultado


def operador_raiz(image):
    imagen_resultado = image.copy()
    imagen_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    c = 50 / np.log10(1 + np.max(imagen_gray))
    alto, ancho = imagen_gray.shape

    for i in range(alto):
        for j in range(ancho):
            raiz = c * (np.sqrt(imagen_gray[i][j]))
            if (raiz < 0):  # Si el resultado del pixel es un valor menor que 0
                imagen_resultado[i][j] = 0  # Se le asignara 0
            elif (raiz > 255):  # Si el resultado del pixel es un valor menor que 255
                imagen_resultado[i][j] = 255  # Se le asignara 0
            else:
                imagen_resultado[i][
                    j] = raiz  # Si los valores estan entre el rango de [0,255] se guardan sin mofiicacion

    return imagen_resultado


def operador_exponencial(imagen, c):
    imagen_resultado = imagen.copy()
    imagen_gray = cv2.cvtColor(imagen.copy(), cv2.COLOR_BGR2GRAY)
    b = 1.01

    alto, ancho = imagen_gray.shape

    for x in range(alto):
        for y in range(ancho):
            resultado = c * (np.power(b, imagen_gray[x][y]) - 1)
            if resultado > 255:
                imagen_resultado[x][y] = 255
            else:
                imagen_resultado[x][y] = c * (np.power(b, imagen_gray[x][y]) - 1)

    return imagen_resultado


def power_raise(image, c):
    imagen_resultado = image.copy()
    imagen_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    r = 0.8
    # c = 4
    # c = 2.5
    # #valores de c 1 1.2 0.5 0.8 3 cuando r = 1
    alto, ancho = imagen_gray.shape

    for x in range(alto):
        for y in range(ancho):
            resultado = (c * (np.power(imagen_gray[x][y], r)))
            if resultado > 255:
                imagen_resultado[x][y] = 255
            elif resultado < 0:
                imagen_resultado[x][y] = 0
            else:
                imagen_resultado[x][y] = (c * (np.power(imagen_gray[x][y], r)))

    return imagen_resultado  # importante


##############################################################
############# Algoritmos de la 2da unidad #######################
##############################################################

def arithmetic_add(image1, image2):
    img = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2GRAY)
    # Adecuamos los tamaños de Ambos archivos para que sean iguales
    imgnew = cv2.resize(img, (324, 324))
    img2 = cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2GRAY)
    imgnew2 = cv2.resize(img2, (324, 324))
    res = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2GRAY)

    x, y = imgnew.shape  # tomamos la altura y el ancho de la imagen

    # print(x,y)

    for i in range(x):
        for j in range(y):
            # escalamos la imagen para evitar overflow
            s = ((imgnew[i][j] / 2) + (imgnew2[i][j] / 2))  # Sumamos ambos valores de cada pixel
            # escalamos la imagen para evitar overflow

            s = int(s)  # Transmitimos la imagen a int, antes de la operación de agregar.

            if (s < 0):
                res[i][j] = 0
            elif (s > 255):
                res[i][j] = 255
            else:
                res[i][j] = s  # el resultado lo guaradmos en el picxel de la imagen de salida

    return res


def arithmetic_add_colores(image1, image2):
    # Cargamos la imagen 
    imagen = image1.copy()
    # Adecuamos los tamaños de Ambos archivos para que sean iguales
    imgnew = cv2.resize(imagen, (832, 543))
    img2 = image2.copy()
    img2new = cv2.resize(img2, (832, 543))

    res = image1.copy()

    height, width, chanels = imagen.shape  # tomamos la altura y el ancho de la imagen

    # recorremos los pixeles de la imagen
    for i in range(height):
        for j in range(width):
            suma = (imgnew[i][j] / 2) + (img2new[i][j] / 2)  # Sumamos ambos valores de cada pixel
            # escalamos la imagen para evitar overflow
            res[i][j] = suma  # el resultado lo guaradmos en el picxel de la imagen de salida

    return res


######## ejercicio 3

def pixel_sustraction(image1, image2):
    img1 = image1.copy()
    img2 = image2.copy()
    img1 = img1.astype(int)
    img2 = img2.astype(int)

    alto1, ancho1, canal = image1.shape

    for i in range(canal):
        for j in range(image1.shape[0]):
            for k in range(image1.shape[1]):
                img1[j][k][i] = abs(img1[j][k][i] - img2[j][k][i])
                img1[j][k][i] = abs(img1[j][k][i] - 95)
                if ((70 < img1[j][k][i]) and (img1[j][k][i] < 130)):
                    img1[j][k][i] = 255
                else:
                    img1[j][k][i] = 0

    res = img1
    return res


######### ejercicio 4

def pixel_sustraction_Contrast(image1, image2):
    img1 = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2GRAY)

    img1 = img1.astype(int)
    img2 = img2.astype(int)

    # x,y = imgnew.shape #tomamos la altura y el ancho de la imagen
    # ajustamos el tamaño de la imagen 2 a la de la imagen 1
    img2 = cv2.resize((img2), (img1.shape[1], img1.shape[0]))

    alto1, ancho1 = img1.shape
    alto2, ancho2 = img2.shape

    for i in range(alto1):
        for j in range(ancho1):
            img = abs(img2[i][j] - img1[i][j])

            if (img == 0):
                img1[i][j] = 0
            else:
                img1[i][j] = img

    ##############Contrast######################

    a = 0  # límite inferior
    b = 255  # límite superior
    c = np.min(img1)  # El menor valor de los pixeles
    d = np.max(img1)  # El mayor valor de los pixeles

    # alto, ancho = img.shape

    result = img1

    def point_operator(pixel_RGB):
        return (pixel_RGB - c) * ((b - a) / (d - c) + a)

    for x in range(alto1):
        for y in range(ancho1):
            result[x][y] = point_operator(img1[x][y])

    return result


######## Practica 7   ########
############# EJERCICO 1 - especial
def op_multiplicacion(image1, c):
    img = image1.copy()
    img = img.astype(int)
    h, w, canal = img.shape

    for i in range(canal):
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                tmp = img[j][k][i] * c
                if (tmp >= 255):
                    img[j][k][i] = 255
                else:
                    img[j][k][i] = tmp
    res = img
    return res


###### EJERCICIO 2

def pixel_division_Thresholding(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    imageOut = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    img1 = img1.astype(int)
    img2 = img2.astype(int)

    img2.shape = img1.shape  # Adecuamos los tamaños de Ambos archivos para que sean iguales

    alto1, ancho1 = img1.shape
    alto2, ancho2 = img2.shape
    alto3, ancho3 = imageOut.shape

    c = 100
    for x in range(alto1):
        for y in range(ancho1):
            imageOut[x][y] = int((img1[x][y] / img2[x][y]) * c)

    imaMin = np.min(imageOut)
    imaMax = np.max(imageOut)
    newMin = 0
    newMax = 255

    def escalar(pixel):
        return (pixel - imaMin) * ((newMax - newMin) / (imaMax - imaMin) + newMin)

    for x in range(alto3):
        for y in range(ancho3):
            imageOut[x][y] = escalar(imageOut[x][y])

    ###############Thresholding################
    result = img1

    for x in range(alto3):
        for y in range(ancho3):
            if (10 < imageOut[x][y] and imageOut[x][y] < 170):
                result[x, y] = 0
            else:
                result[x, y] = 255

    return result


############# EJERCICIO 3

def pixel_division_Contrast(image1, image2):
    img1 = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2GRAY)

    img1 = img1.astype(int)
    img2 = img2.astype(int)

    # x,y = imgnew.shape #tomamos la altura y el ancho de la imagen
    # ajustamos el tamaño de la imagen 2 a la de la imagen 1
    img2 = cv2.resize((img2), (img1.shape[1], img1.shape[0]))

    alto1, ancho1 = img1.shape
    alto2, ancho2 = img2.shape

    for i in range(alto1):
        for j in range(ancho1):
            img = abs((img2[i][j] / img1[i][j])) * 30

            if (img == 0):
                img1[i][j] = 0
            else:
                img1[i][j] = img

    ##############Contrast######################

    a = 0  # límite inferior
    b = 255  # límite superior
    c = np.min(img1)  # El menor valor de los pixeles
    d = np.max(img1)  # El mayor valor de los pixeles

    # alto, ancho = img.shape

    result = img1

    def point_operator(pixel_RGB):
        return (pixel_RGB - c) * ((b - a) / (d - c) + a)

    for x in range(alto1):
        for y in range(ancho1):
            result[x][y] = point_operator(img1[x][y])

    return result


####### ejercicio 4

def ope_blend(image1, image2, x):
    img1 = image1.copy()
    img2 = image2.copy()

    # ajustamos el tamaño de la imagen 2 a la de la imagen 1
    img2 = cv2.resize((img2), (img1.shape[1], img1.shape[0]))

    # creamos una imagen en negro
    out = np.zeros(shape=img1.shape, dtype=np.uint8)

    for i in range(len(img1[0][0])):
        for j in range(img1.shape[0]):
            for k in range(img1.shape[1]):
                out[j][k][i] = x * img1[j][k][i] + (1 - x) * img2[j][k][i]

    return out  # importante


#####################Practica 8 #########################

#### EJER 1
def operador_AND(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    img1 = img1.astype(int)
    img2 = img2.astype(int)

    img2.shape = img1.shape  # Adecuamos los tamaños de Ambos archivos para que sean iguales

    alto1, ancho1 = img1.shape

    # alto2,ancho2 = img2.shape
    ##################NOT####################
    def NOT(imagen):
        alto1, ancho1 = imagen.shape
        res = imagen
        for i in range(alto1):
            for j in range(ancho1):
                img = abs((255 - imagen[i][j]))

                if (img == 0):
                    res[i][j] = 0
                else:
                    res[i][j] = img
        return res

    #################Threshold#####################
    def Thresholding(imagen):
        alto, ancho = imagen.shape
        result = imagen
        for x in range(alto):
            for y in range(ancho):
                if (0 < imagen[x][y] and imagen[x][y] < 120):
                    result[x, y] = 0
                else:
                    result[x, y] = 255
        return result

    ############################AND###########

    img1Not = NOT(img1)
    img2Not = NOT(img2)
    img1Bin = Thresholding(img1Not)
    img2Bin = Thresholding(img2Not)

    for i in range(alto1):
        for j in range(ancho1):
            img = abs((img1Bin[i][j] & img2Bin[i][j]))

            if (img == 0):
                img1[i][j] = 0
            else:
                img1[i][j] = img

    return img1


########## EJER 2
############## EJER 2 

def operador_OR_Thresholding(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    img1 = img1.astype(int)
    img2 = img2.astype(int)

    img2.shape = img1.shape  # Adecuamos los tamaños de Ambos archivos para que sean iguales

    alto1, ancho1 = img1.shape

    # alto2,ancho2 = img2.shape
    ##################NOT####################
    def NOT(imagen):
        alto1, ancho1 = imagen.shape
        res = imagen
        for i in range(alto1):
            for j in range(ancho1):
                img = abs((255 - imagen[i][j]))

                if (img == 0):
                    res[i][j] = 0
                else:
                    res[i][j] = img
        return res

    #################Threshold#####################
    def Thresholding(imagen):
        alto, ancho = imagen.shape
        result = imagen
        for x in range(alto):
            for y in range(ancho):
                if (10 < imagen[x][y] and imagen[x][y] < 100):
                    result[x, y] = 0
                else:
                    result[x, y] = 255
        return result

    ################OR#############

    img1Not = NOT(img1)
    img2Not = NOT(img2)
    img1Bin = Thresholding(img1Not)
    img2Bin = Thresholding(img2Not)

    for i in range(alto1):
        for j in range(ancho1):
            img = abs((img1Bin[i][j] | img2Bin[i][j]))

            if (img == 0):
                img1[i][j] = 0
            else:
                img1[i][j] = img

    res = img1
    return res


########## EJER 3

def op_XOR_Thresholding(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    img1 = img1.astype(int)
    img2 = img2.astype(int)
    img2.shape = img1.shape  # Adecuamos los tamaños de Ambos archivos para que sean iguales

    alto1, ancho1 = img1.shape

    # alto2,ancho2 = img2.shape
    ##################NOT####################
    def NOT(imagen):
        alto1, ancho1 = imagen.shape
        res = imagen
        for i in range(alto1):
            for j in range(ancho1):
                img = abs((255 - imagen[i][j]))

                if (img == 0):
                    res[i][j] = 0
                else:
                    res[i][j] = img
        return res

    #################Threshold#####################
    def Thresholding(imagen):
        alto, ancho = imagen.shape
        result = imagen
        for x in range(alto):
            for y in range(ancho):
                if (10 < imagen[x][y] and imagen[x][y] < 100):
                    result[x, y] = 0
                else:
                    result[x, y] = 255
        return result

    ###########################XOR###########

    img1Not = NOT(img1)
    img2Not = NOT(img2)
    img1Bin = Thresholding(img1Not)
    img2Bin = Thresholding(img2Not)
    img1Bin = NOT(img1Bin)
    img2Bin = NOT(img2Bin)

    for i in range(alto1):
        for j in range(ancho1):
            img = abs((img1Bin[i][j] ^ img2Bin[i][j]))

            if (img == 0):
                img1[i][j] = 0
            else:
                img1[i][j] = img

    res = img1
    return res

#############################################################
############## PRACTICA 9.1  ################################
#############################################################

########### EJERCICIO 1

##### TRASLADAR IMAGEN
def trasladar_imagen(image, tx, ty):
    img = image.copy()
    ancho = img.shape[1]  # Columnas
    alto = img.shape[0]  # Filas

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    imageOut = cv2.warpAffine(img, M, (ancho, alto))

    return imageOut

##### ROTAR IMAGEN

def rotar_imagen(image, a):
    img = image.copy()
    ancho = img.shape[1]  # Columnas
    alto = img.shape[0]  # Filas

    def matrix_rotacion(angulo, tx, ty):
        coseno = math.cos(angulo)
        seno = math.sin(angulo)
        calcular_1 = (1 - coseno) * tx - seno * ty
        calcular_2 = seno * tx + (1 - coseno) * ty
        return np.array([[coseno, seno, calcular_1], [-seno, coseno, calcular_2]], dtype=np.float32)

    M = matrix_rotacion(a, ancho // 2, alto // 2)
    imageOut = cv2.warpAffine(img, M, (ancho, alto))

    return imageOut