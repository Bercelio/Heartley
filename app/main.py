from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import gdown
import gzip
import shutil

app = Flask(__name__)

# Descargar y descomprimir modelo si no está disponible
def descargar_y_cargar_modelo():
    url = "https://drive.google.com/uc?export=download&id=1L3SdBGjZUOxL7ForTYDw64L50K6U_h2B"
    gz_path = "model/Heartly_model_17col.joblib.gz"
    model_path = "model/Heartly_model_17col.joblib"
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(model_path):
        print("Descargando modelo comprimido desde Google Drive...")
        gdown.download(url, gz_path, quiet=False)

        print("Descomprimiendo modelo...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return joblib.load(model_path)

# Cargar modelo
model = descargar_y_cargar_modelo()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resultado', methods=['POST'])
def resultado():
    try:
        edad = int(request.form['age'])
        genero = int(request.form['gender'])
        estatura = int(request.form['height'])
        peso = int(request.form['weight'])
        sistolica = int(request.form['ap_hi'])
        diastolica = int(request.form['ap_lo'])
        colesterol = int(request.form['cholesterol'])
        glucosa = int(request.form['gluc'])
        fumador = int(request.form['smoke'])
        alcohol = int(request.form['alco'])
        actividad = int(request.form['active'])
        cardiovascular = 0  # Suponemos que no está en el formulario

        # Datos para el modelo
        datos = np.array([[edad, genero, estatura, peso, sistolica, diastolica,
                           colesterol, glucosa, fumador, alcohol, actividad, cardiovascular]])

        # Predicción
        resultado_modelo = model.predict(datos)[0]

        # Clasificación por presión
        def clasificar_presion(s, d):
            if s < 120 and d < 80:
                return "Normal"
            elif 120 <= s <= 129 and d < 80:
                return "Elevada"
            elif 130 <= s <= 139 or 80 <= d <= 89:
                return "Hipertensión etapa 1"
            elif s >= 140 or d >= 90:
                return "Hipertensión etapa 2"
            elif s > 180 or d > 120:
                return "Crisis hipertensiva"
            else:
                return "No clasificado"

        fase = clasificar_presion(sistolica, diastolica)

        # IMC
        estatura_m = estatura / 100
        imc = round(peso / (estatura_m ** 2), 1)

        # Gráfica
        plt.figure(figsize=(6, 4))
        valores_usuario = [sistolica, diastolica, imc]
        valores_ideales = [120, 80, 24.9]
        etiquetas = ['Sistólica', 'Diastólica', 'IMC']
        x = np.arange(len(etiquetas))

        plt.bar(x - 0.2, valores_usuario, width=0.4, label='Tú')
        plt.bar(x + 0.2, valores_ideales, width=0.4, label='Ideal')
        plt.xticks(x, etiquetas)
        plt.ylabel('Valores')
        plt.title('Tu presión e IMC vs valores ideales')
        plt.legend()
        plt.tight_layout()
        os.makedirs('static', exist_ok=True)
        ruta_imagen = 'static/grafico_resultado.png'
        plt.savefig(ruta_imagen)
        plt.close()

        return render_template('resultado.html',
                               ap_hi=sistolica,
                               ap_lo=diastolica,
                               fase=fase,
                               bmi=imc,
                               resultado=resultado_modelo,
                               imagen=ruta_imagen)

    except Exception as e:
        return f"Ocurrió un error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
