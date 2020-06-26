import os
import cv2
from algoritmos import algoritmo_dummy, operador_logaritmico, threholding, constrast_streching, constrast_streching_out, \
    histogram_equalization, operador_raiz, arithmetic_add, arithmetic_add_colores, operador_exponencial, power_raise, \
    op_multiplicacion, ope_blend, pixel_sustraction_Contrast, pixel_division_Thresholding, pixel_division_Contrast, operador_AND, \
    operador_OR_Thresholding, op_XOR_Thresholding, trasladar_imagen, rotar_imagen


# lee la imagen, aplica el algoritmo y retorna la ruta completa de
# la imagen resultado
def procesar_algoritmo_dummy(ruta, nombre_imagen, prefijo):
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = algoritmo_dummy(imagen)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_threholding(ruta, nombre_imagen, prefijo):
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = threholding(imagen)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_constrast_streching(ruta, nombre_imagen, prefijo):
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = constrast_streching(imagen)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_constrast_streching_out(ruta, nombre_imagen, prefijo):
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = constrast_streching_out(imagen)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_histogram_equalization(ruta, nombre_imagen, prefijo):
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = histogram_equalization(imagen)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_logaritmico(ruta, nombre_imagen, prefijo):
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = operador_logaritmico(imagen)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_operador_raiz(ruta, nombre_imagen, prefijo):
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = operador_raiz(imagen)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_exponencial(c, ruta, nombre_imagen, prefijo):
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = operador_exponencial(imagen, c)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_power_raise(c, ruta, nombre_imagen, prefijo):
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = power_raise(imagen, c)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


#################### Segunda unidad ####################

# PRACTICA 6
# ejer1
def procesar_algoritmo_arithmetic_add(ruta, nombre_imagen1, nombre_imagen2, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    ruta_imagen2 = os.path.join(ruta, nombre_imagen2)
    imagen2 = cv2.imread(ruta_imagen2)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = arithmetic_add(imagen1, imagen2)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen1)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_arithmetic_add_color(ruta, nombre_imagen1, nombre_imagen2, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    ruta_imagen2 = os.path.join(ruta, nombre_imagen2)
    imagen2 = cv2.imread(ruta_imagen2)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = arithmetic_add_colores(imagen1, imagen2)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen1)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado

#pract 7
def procesar_algoritmo_op_multiplicacion(c, ruta, nombre_imagen, prefijo):
    ruta_imagen = os.path.join(ruta, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = op_multiplicacion(imagen, c)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado

def procesar_algoritmo_pixel_division_Thresholding(ruta, nombre_imagen1, nombre_imagen2, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    ruta_imagen2 = os.path.join(ruta, nombre_imagen2)
    imagen2 = cv2.imread(ruta_imagen2)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = pixel_division_Thresholding (imagen1, imagen2)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen2)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado

def procesar_algoritmo_pixel_division_Contrast(ruta, nombre_imagen1, nombre_imagen2, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    ruta_imagen2 = os.path.join(ruta, nombre_imagen2)
    imagen2 = cv2.imread(ruta_imagen2)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = pixel_division_Contrast(imagen1, imagen2)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen1)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado

def procesar_algoritmo_ope_blend(x, ruta, nombre_imagen1, nombre_imagen2, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    ruta_imagen2 = os.path.join(ruta, nombre_imagen2)
    imagen2 = cv2.imread(ruta_imagen2)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = ope_blend(imagen1, imagen2, x)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen1)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_sustraction_Contrast(ruta, nombre_imagen1, nombre_imagen2, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    ruta_imagen2 = os.path.join(ruta, nombre_imagen2)
    imagen2 = cv2.imread(ruta_imagen2)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = pixel_sustraction_Contrast(imagen1, imagen2)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen1)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado

##### Practica 8
def procesar_algoritmo_operador_AND(ruta, nombre_imagen1, nombre_imagen2, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    ruta_imagen2 = os.path.join(ruta, nombre_imagen2)
    imagen2 = cv2.imread(ruta_imagen2)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = operador_AND(imagen1, imagen2)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen1)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado

def procesar_algoritmo_operador_OR_Thresholding(ruta, nombre_imagen1, nombre_imagen2, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    ruta_imagen2 = os.path.join(ruta, nombre_imagen2)
    imagen2 = cv2.imread(ruta_imagen2)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = operador_OR_Thresholding(imagen1, imagen2)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen1)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado

def procesar_algoritmo_op_XOR_Thresholding(ruta, nombre_imagen1, nombre_imagen2, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    ruta_imagen2 = os.path.join(ruta, nombre_imagen2)
    imagen2 = cv2.imread(ruta_imagen2)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = op_XOR_Thresholding (imagen1, imagen2)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen1)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado


def procesar_algoritmo_trasladar_imagen(tx, ty, ruta, nombre_imagen1, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = trasladar_imagen(imagen1, tx, ty)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen1)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado

def procesar_algoritmo_rotar_imagen(x, ruta, nombre_imagen1, prefijo):
    ruta_imagen1 = os.path.join(ruta, nombre_imagen1)
    imagen1 = cv2.imread(ruta_imagen1)

    # Llamo al algoritmo y guardo el resultado
    imagen_resultado = rotar_imagen(imagen1, x)

    ruta_imagen_resultado = os.path.join(ruta, prefijo + nombre_imagen1)
    cv2.imwrite(ruta_imagen_resultado, imagen_resultado)
    return ruta_imagen_resultado