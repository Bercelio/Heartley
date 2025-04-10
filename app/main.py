from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar modelo
with open("heartly_model.pkl", "rb") as f:
    model = pickle.load(f)

# Funciones auxiliares
def calcular_imc(peso, estatura):
    estatura_m = estatura / 100
    return round(peso / (estatura_m ** 2), 2)

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

@app.route('/')
def formulario():
    return render_template("index.html")

@app.route('/resultado', methods=['POST'])
def resultado():
    # Obtener datos del formulario
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
    cardio = int(request.form['cardio'])  # solo para análisis, no entra al modelo

    # Datos al modelo (excluye 'cardio')
    entrada_modelo = np.array([[edad, genero, altura, peso, ap_hi, ap_lo,
                                colesterol, glucosa, fuma, alcohol, activo]])
    
    # Predicción
    prediccion = model.predict(entrada_modelo)[0]

    # Análisis clínico
    imc = calcular_imc(peso, altura)
    fase = clasificar_fase_presion(ap_hi, ap_lo)

    # Diagnóstico interpretativo
    if prediccion == 2:
        nivel = "Alto"
        mensaje = "Alto riesgo cardiovascular. Requiere atención médica inmediata."
    elif prediccion == 1:
        nivel = "Medio"
        mensaje = "Riesgo moderado. Se recomienda mejorar hábitos de salud y realizar chequeos periódicos."
    else:
        nivel = "Bajo"
        mensaje = "Buen estado general. Mantén un estilo de vida saludable."

    # Análisis histórico
    historico = "El paciente ha tenido diagnóstico previo de enfermedad cardiovascular." if cardio == 1 else "No se reporta diagnóstico previo de enfermedad cardiovascular."

    return render_template("resultado.html",
                           edad=edad, genero=genero, altura=altura, peso=peso,
                           ap_hi=ap_hi, ap_lo=ap_lo, colesterol=colesterol, glucosa=glucosa,
                           fuma=fuma, alcohol=alcohol, activo=activo,
                           imc=imc, fase=fase, nivel=nivel, mensaje=mensaje, historico=historico)
