import os

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from procesar_algoritmos import procesar_algoritmo_logaritmico, procesar_algoritmo_threholding, \
    procesar_algoritmo_constrast_streching, procesar_algoritmo_constrast_streching_out, \
    procesar_algoritmo_histogram_equalization, procesar_algoritmo_operador_raiz, procesar_algoritmo_arithmetic_add, \
    procesar_algoritmo_arithmetic_add_color, procesar_algoritmo_exponencial, procesar_algoritmo_power_raise, \
    procesar_algoritmo_op_multiplicacion, procesar_algoritmo_ope_blend, procesar_algoritmo_sustraction_Contrast, \
    procesar_algoritmo_pixel_division_Thresholding, procesar_algoritmo_pixel_division_Contrast, procesar_algoritmo_operador_AND, \
    procesar_algoritmo_operador_OR_Thresholding, procesar_algoritmo_op_XOR_Thresholding, procesar_algoritmo_trasladar_imagen, \
    procesar_algoritmo_rotar_imagen

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/media'


@app.route('/')
def index():
    return render_template('index.html')


#################################################

# pedir recursos html
# Devolver html's

@app.route('/thresholding.html')
def thresholding_html():
    return render_template('thresholding.html')


@app.route('/contrast.html')
def contrast_html():
    return render_template('contrast.html')


@app.route('/outStret.html')
def out_stret_html():
    return render_template('outStret.html')


@app.route('/histo.html')
def histo_html():
    return render_template('histo.html')


@app.route('/log.html')
def log_html():
    return render_template('log.html')


@app.route('/raiz.html')
def raiz_html():
    return render_template('raiz.html')


@app.route('/expo.html')
def expo_html():
    return render_template('expo.html')


@app.route('/raisePower.html')
def raise_power_html():
    return render_template('raisePower.html')


@app.route('/subtraction.html')
def subtraction_html():
    return render_template('subtraction.html')


@app.route('/multiplication.html')
def multiplication_html():
    return render_template('multiplication.html')


@app.route('/blending.html')
def blending_html():
    return render_template('/blending.html')


@app.route('/divisionThresholding.html')
def div_html():
    return render_template('divisionThresholding.html')


@app.route('/add.html')
def add_html():
    return render_template('add.html')


@app.route('/addColor.html')
def addColor_html():
    return render_template('addColor.html')


@app.route('/subtraccContras.html')
def subtraccContras_html():
    return render_template('subtraccContras.html')

@app.route('/divisionContrastStre.html')
def divisionContrastStre_html():
    return render_template('divisionContrastStre.html')

@app.route('/and.html')
def operadorAND_html():
    return render_template('and.html')

@app.route('/or.html')
def operadorOR_html():
    return render_template('or.html')

@app.route('/Xor.html')
def operadorXOR_html():
    return render_template('Xor.html')

@app.route('/trasladar.html')
def trasladar_html():
    return render_template('trasladar.html')

@app.route('/rotarImage.html')
def rotar_html():
    return render_template('rotarImage.html')


#################################################
# Algoritmos

@app.route('/thres', methods=['POST'])
def thres():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)
        ruta_imagen_resultado = procesar_algoritmo_threholding(app.config['UPLOAD_FOLDER'], filename, 'thres')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/contr', methods=['POST'])
def contr():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)
        ruta_imagen_resultado = procesar_algoritmo_constrast_streching(app.config['UPLOAD_FOLDER'], filename, 'contr')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/out', methods=['POST'])
def out():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)
        ruta_imagen_resultado = procesar_algoritmo_constrast_streching_out(app.config['UPLOAD_FOLDER'], filename, 'out')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/histo', methods=['POST'])
def histo():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)
        ruta_imagen_resultado = procesar_algoritmo_histogram_equalization(app.config['UPLOAD_FOLDER'], filename,
                                                                          'histo')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/log', methods=['POST'])
def log():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)
        ruta_imagen_resultado = procesar_algoritmo_logaritmico(app.config['UPLOAD_FOLDER'], filename, 'log')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/raiz', methods=['POST'])
def raiz():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)
        ruta_imagen_resultado = procesar_algoritmo_operador_raiz(app.config['UPLOAD_FOLDER'], filename, 'raiz')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/expo', methods=['POST'])
def expo():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)

        C = int(request.form.get('C'))

        ruta_imagen_resultado = procesar_algoritmo_exponencial(C, app.config['UPLOAD_FOLDER'], filename, 'expo-')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/raise', methods=['POST'])
def raise_power():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)

        C = float(request.form.get('C'))

        ruta_imagen_resultado = procesar_algoritmo_power_raise(C, app.config['UPLOAD_FOLDER'], filename, 'raizPower-')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/multi', methods=['POST'])
def op_multiplicacion():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)

        C = int(request.form.get('C'))

        ruta_imagen_resultado = procesar_algoritmo_op_multiplicacion(C, app.config['UPLOAD_FOLDER'], filename,
                                                                     'raizPower-')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/add', methods=['POST'])
def add():
    if request.method == 'POST':
        image1 = request.files['archivo1']
        filename1 = secure_filename(image1.filename)
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image1.save(full_filename1)

        image2 = request.files['archivo2']
        filename2 = secure_filename(image2.filename)
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image2.save(full_filename2)

        ruta_imagen_resultado = procesar_algoritmo_arithmetic_add(app.config['UPLOAD_FOLDER'], filename1, filename2,
                                                                  'add-')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/addColor', methods=['POST'])
def add_color():
    if request.method == 'POST':
        image1 = request.files['archivo1']
        filename1 = secure_filename(image1.filename)
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image1.save(full_filename1)

        image2 = request.files['archivo2']
        filename2 = secure_filename(image2.filename)
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image2.save(full_filename2)

        ruta_imagen_resultado = procesar_algoritmo_arithmetic_add_color(app.config['UPLOAD_FOLDER'], filename1,
                                                                        filename2, 'addColor-')
        print(ruta_imagen_resultado)
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/divT', methods=['POST'])
def divT():
    if request.method == 'POST':
        image1 = request.files['archivo1']
        filename1 = secure_filename(image1.filename)
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image1.save(full_filename1)

        image2 = request.files['archivo2']
        filename2 = secure_filename(image2.filename)
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image2.save(full_filename2)

        ruta_imagen_resultado = procesar_algoritmo_pixel_division_Thresholding(app.config['UPLOAD_FOLDER'], filename1,
                                                                        filename2, 'divT-')
        print(ruta_imagen_resultado)
        return render_template('result.html', imagen=ruta_imagen_resultado)

@app.route('/divContras', methods=['POST'])
def divContras():
    if request.method == 'POST':
        image1 = request.files['archivo1']
        filename1 = secure_filename(image1.filename)
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image1.save(full_filename1)

        image2 = request.files['archivo2']
        filename2 = secure_filename(image2.filename)
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image2.save(full_filename2)

        ruta_imagen_resultado = procesar_algoritmo_pixel_division_Contrast(app.config['UPLOAD_FOLDER'], filename1,
                                                                        filename2, 'divContras-')
        return render_template('result.html', imagen=ruta_imagen_resultado)

@app.route('/blen', methods=['POST'])
def op_blen():
    if request.method == 'POST':
        image1 = request.files['archivo1']
        filename1 = secure_filename(image1.filename)
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image1.save(full_filename1)

        image2 = request.files['archivo2']
        filename2 = secure_filename(image2.filename)
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image2.save(full_filename2)

        X = float(request.form.get('X'))

        ruta_imagen_resultado = procesar_algoritmo_ope_blend(X, app.config['UPLOAD_FOLDER'], filename1,
                                                             filename2, 'blen-')
        return render_template('result.html', imagen=ruta_imagen_resultado)


@app.route('/subcontrast', methods=['POST'])
def subcontrast():
    if request.method == 'POST':
        image1 = request.files['archivo1']
        filename1 = secure_filename(image1.filename)
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image1.save(full_filename1)

        image2 = request.files['archivo2']
        filename2 = secure_filename(image2.filename)
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image2.save(full_filename2)

        ruta_imagen_resultado = procesar_algoritmo_sustraction_Contrast(app.config['UPLOAD_FOLDER'], filename1,
                                                                        filename2, 'subcontrast-')
        return render_template('result.html', imagen=ruta_imagen_resultado)

@app.route('/and', methods=['POST'])
def operadorAnd():
    if request.method == 'POST':
        image1 = request.files['archivo1']
        filename1 = secure_filename(image1.filename)
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image1.save(full_filename1)

        image2 = request.files['archivo2']
        filename2 = secure_filename(image2.filename)
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image2.save(full_filename2)

        ruta_imagen_resultado = procesar_algoritmo_operador_AND(app.config['UPLOAD_FOLDER'], filename1,
                                                                        filename2, 'and-')
        return render_template('result.html', imagen=ruta_imagen_resultado)

@app.route('/or', methods=['POST'])
def operadorOr():
    if request.method == 'POST':
        image1 = request.files['archivo1']
        filename1 = secure_filename(image1.filename)
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image1.save(full_filename1)

        image2 = request.files['archivo2']
        filename2 = secure_filename(image2.filename)
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image2.save(full_filename2)

        ruta_imagen_resultado = procesar_algoritmo_operador_OR_Thresholding(app.config['UPLOAD_FOLDER'], filename1,
                                                                        filename2, 'Or-')
        return render_template('result.html', imagen=ruta_imagen_resultado)

@app.route('/Xor', methods=['POST'])
def operadorXor():
    if request.method == 'POST':
        image1 = request.files['archivo1']
        filename1 = secure_filename(image1.filename)
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image1.save(full_filename1)

        image2 = request.files['archivo2']
        filename2 = secure_filename(image2.filename)
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image2.save(full_filename2)

        ruta_imagen_resultado = procesar_algoritmo_op_XOR_Thresholding(app.config['UPLOAD_FOLDER'], filename1,
                                                                        filename2, 'Xor-')
        return render_template('result.html', imagen=ruta_imagen_resultado)

@app.route('/trasladar', methods=['POST'])
def trasladarImg():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)

        x = float(request.form.get('x'))
        y = float(request.form.get('y'))

        ruta_imagen_resultado = procesar_algoritmo_trasladar_imagen(x, y, app.config['UPLOAD_FOLDER'], filename, 'tras-')
        return render_template('result.html', imagen=ruta_imagen_resultado)

@app.route('/rotar', methods=['POST'])
def rotarImg():
    if request.method == 'POST':
        image = request.files['archivo']
        filename = secure_filename(image.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(full_filename)

        x = float(request.form.get('x'))

        ruta_imagen_resultado = procesar_algoritmo_rotar_imagen(x, app.config['UPLOAD_FOLDER'], filename, 'rotar-')
        return render_template('result.html', imagen=ruta_imagen_resultado)

######################################################################################################################


@app.route('/cascada')
def cascada():
    return render_template('cascada.html')


def aplicar_algoritmo(codigo, filename):
    if codigo == '1':
        prefijo = 'thre-'
        ruta_imagen_resultado = procesar_algoritmo_threholding(app.config['UPLOAD_FOLDER'], filename, prefijo)
        return prefijo + filename
    elif codigo == '2':
        prefijo = 'cons-'
        ruta_imagen_resultado = procesar_algoritmo_constrast_streching(app.config['UPLOAD_FOLDER'], filename, prefijo)
        return prefijo + filename
    elif codigo == '3':
        prefijo = 'outContras-'
        ruta_imagen_resultado = procesar_algoritmo_constrast_streching_out(app.config['UPLOAD_FOLDER'], filename,
                                                                           prefijo)
        return prefijo + filename
    elif codigo == '4':
        prefijo = 'histo-'
        ruta_imagen_resultado = procesar_algoritmo_histogram_equalization(app.config['UPLOAD_FOLDER'], filename,
                                                                          prefijo)
        return prefijo + filename
    elif codigo == '5':
        prefijo = 'log-'
        ruta_imagen_resultado = procesar_algoritmo_logaritmico(app.config['UPLOAD_FOLDER'], filename, prefijo)
        return prefijo + filename
    elif codigo == '6':
        prefijo = 'raiz-'
        ruta_imagen_resultado = procesar_algoritmo_operador_raiz(app.config['UPLOAD_FOLDER'], filename, prefijo)
        return prefijo + filename


@app.route('/cascadax', methods=['POST', 'GET'])
def cascadax():
    image = request.files['archivo']
    filename = secure_filename(image.filename)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(full_filename)

    arr = request.form.get('arr')
    newArr = arr.split(',')

    nombre_nueva_imagen = filename
    for algoritmo in newArr:
        if algoritmo != '':
            nombre_nueva_imagen = aplicar_algoritmo(algoritmo, nombre_nueva_imagen)

    ruta_imagen_resultado = os.path.join(app.config['UPLOAD_FOLDER'], nombre_nueva_imagen)
    return render_template('result.html', imagen=ruta_imagen_resultado)


if __name__ == '__main__':
    app.run()
