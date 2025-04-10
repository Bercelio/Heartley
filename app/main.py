from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import gdown

app = Flask(__name__)

# Descargar el modelo desde Google Drive si no existe
if not os.path.exists("heartly_model.pkl"):
    print("🔽 Descargando modelo desde Google Drive...")
    url = "https://drive.google.com/uc?id=1EHLpKvhgAn7OBrmdxxOaNRG7gpJsOqC9"
    gdown.download(url, "heartly_model.pkl", quiet=False)

# Cargar modelo
with open("heartly_model.pkl", "rb") as f:
    model = pickle.load(f)

# Función para calcular IMC
def calcular_imc(peso, estatura):
    estatura_m = estatura / 100
    return round(peso / (estatura_m ** 2), 2)

# Clasificar fase de presión arterial
def clasificar_fase_presion(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80:
        return "Normal"
    elif 120 <= ap_hi < 130 and ap_lo < 80:
        return "Elevada"
    elif (130 <= ap_hi < 140) or (80 <= ap_lo < 90):
        return "Hipertensión Etapa 1"
    elif (140 <= ap_hi <= 180) or (90 <= ap_lo <= 120):
        return "Hipertensión Etapa 2"
    elif ap_hi > 180 or ap_lo > 120:
        return "Crisis Hipertensiva"
    else:
        return "No Clasificada"

# Página de inicio
@app.route('/')
def formulario():
    return render_template("index.html")

# Página de resultado
@app.route('/resultado', methods=['POST'])
def resultado():
    # Datos del formulario
    edad = int(request.form['age'])
    genero = int(request.form['gender'])
    altura = int(request.form['height'])
    peso = float(request.form['weight'])
    ap_hi = int(request.form['ap_hi'])
    ap_lo = int(request.form['ap_lo'])
    colesterol = int(request.form['cholesterol'])
    glucosa = int(request.form['gluc'])
    fuma = int(request.form['smoke'])
    alcohol = int(request.form['alco'])
    activo = int(request.form['active'])

    # Datos para el modelo (sin cardio)
    entrada_modelo = np.array([[edad, genero, altura, peso, ap_hi, ap_lo,
                                colesterol, glucosa, fuma, alcohol, activo]])

    # Predicción del riesgo cardiovascular
    prediccion = model.predict(entrada_modelo)[0]

    # Cálculos adicionales
    imc = calcular_imc(peso, altura)
    fase = clasificar_fase_presion(ap_hi, ap_lo)

    # Interpretación del riesgo
    if prediccion == 2:
        resultado_texto = "RIESGO CARDIOVASCULAR ALTO - PRESENTE"
    elif prediccion == 1:
        resultado_texto = "RIESGO CARDIOVASCULAR MODERADO - POSIBLE"
    else:
        resultado_texto = "SIN INDICIOS DE RIESGO CARDIOVASCULAR"

    # Imagen estática (puedes reemplazar luego por una generada)
    imagen = "static/grafica_ejemplo.png"

    return render_template("resultado.html",
                           ap_hi=ap_hi,
                           ap_lo=ap_lo,
                           fase=fase,
                           bmi=imc,
                           resultado=resultado_texto,
                           imagen=imagen)

# Lanzar servidor
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Cloud Run espera el 8080
    app.run(host='0.0.0.0', port=port, debug=True)
