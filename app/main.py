from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import requests

app = Flask(__name__)

# ================================
# DESCARGAR MODELO DESDE GOOGLE DRIVE SI NO EXISTE
# ================================
def descargar_modelo():
    model_path = "model/Heartly_model_17col.joblib"
    if not os.path.exists(model_path):
        print("Descargando modelo desde Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1w2dM0kW041q6NAWXeqDkHKUkdSibdqF9"
        response = requests.get(url)
        os.makedirs("model", exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Modelo descargado y guardado.")
    else:
        print("Modelo ya disponible localmente.")

# Ejecutar descarga si es necesario
descargar_modelo()

# ================================
# CARGAR MODELO
# ================================
modelo = joblib.load("model/Heartly_model_17col.joblib")

# ================================
# OPCIONES Y FUNCIONES DE APOYO
# ================================

fase_opciones = [
    'fase_presion_Crisis Hipertensiva',
    'fase_presion_Elevada',
    'fase_presion_Hipertensión Etapa 1',
    'fase_presion_Hipertensión Etapa 2',
    'fase_presion_Normal'
]

def clasificar_presion(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80:
        return "Normal"
    elif 120 <= ap_hi < 130 and ap_lo < 80:
        return "Elevada"
    elif (130 <= ap_hi < 140) or (80 <= ap_lo < 90):
        return "Hipertensión Etapa 1"
    elif (140 <= ap_hi) or (90 <= ap_lo):
        return "Hipertensión Etapa 2"
    elif ap_hi >= 180 or ap_lo >= 120:
        return "Crisis Hipertensiva"
    else:
        return "No clasificada"

def generar_grafica(ap_hi, ap_lo, bmi):
    categorias = ["Presión Sistólica", "Presión Diastólica", "IMC"]
    valores = [ap_hi, ap_lo, bmi]
    ideales = [120, 80, 22]
    colores = ["tomato" if v > i else "green" for v, i in zip(valores, ideales)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categorias, valores, color=colores)
    ax.plot(categorias, ideales, color="blue", linestyle="--", marker="o", label="Ideal")
    for i, v in enumerate(valores):
        ax.text(i, v + 1, f"{v:.1f}", ha='center')

    ax.set_ylim(0, max(valores + ideales) + 20)
    ax.set_title("Comparación con Valores Ideales")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(True)

    path = os.path.join("static", "grafica_resultado.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

# ================================
# RUTAS DE LA APLICACIÓN
# ================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/resultado", methods=["POST"])
def resultado():
    datos = request.form

    # Obtener variables del formulario
    edad = int(datos["age"])
    genero = int(datos["gender"])
    estatura = float(datos["height"])
    peso = float(datos["weight"])
    ap_hi = int(datos["ap_hi"])
    ap_lo = int(datos["ap_lo"])
    colesterol = int(datos["cholesterol"])
    glucosa = int(datos["gluc"])
    fuma = int(datos["smoke"])
    alcohol = int(datos["alco"])
    activo = int(datos["active"])

    bmi = peso / ((estatura / 100) ** 2)
    fase = clasificar_presion(ap_hi, ap_lo)
    fase_codificada = [1 if f == f"fase_presion_{fase}" else 0 for f in fase_opciones]

    entrada = [genero, estatura, peso, ap_hi, ap_lo, colesterol, glucosa,
               fuma, alcohol, activo, edad, bmi] + fase_codificada

    pred = modelo.predict([entrada])[0]
    riesgo = "RIESGO PRESENTE" if pred == 1 else "SIN RIESGO DETECTADO"

    # Generar gráfica
    grafica_path = generar_grafica(ap_hi, ap_lo, bmi)

    return render_template("resultado.html",
                           resultado=riesgo,
                           ap_hi=ap_hi,
                           ap_lo=ap_lo,
                           bmi=round(bmi, 1),
                           fase=fase,
                           imagen=grafica_path)

if __name__ == "__main__":
    app.run(debug=True)
