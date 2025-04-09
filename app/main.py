# app/main.py

from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import requests

app = Flask(__name__)

# Función para descargar el modelo desde Google Drive si no existe localmente
def descargar_modelo():
    model_path = "model/Heartly_model_17col.joblib"
    if not os.path.exists(model_path):
        print("Descargando modelo desde Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1w2dM0kW041q6NAWXeqDkHKUkdSibdqF9"
        response = requests.get(url)
        os.makedirs("model", exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Modelo descargado.")
    else:
        print("Modelo ya disponible localmente.")

descargar_modelo()

# Cargar modelo
model = joblib.load("model/Heartly_model_17col.joblib")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resultado', methods=['POST'])
def resultado():
    try:
        edad = int(request.form['edad'])
        genero = int(request.form['genero'])
        estatura = int(request.form['estatura'])
        peso = int(request.form['peso'])
        sistolica = int(request.form['sistolica'])
        diastolica = int(request.form['diastolica'])
        colesterol = int(request.form['colesterol'])
        glucosa = int(request.form['glucosa'])
        fumador = int(request.form['fumador'])
        alcohol = int(request.form['alcohol'])
        actividad = int(request.form['actividad'])
        cardiovascular = int(request.form['cardio'])

        datos = np.array([[edad, genero, estatura, peso, sistolica, diastolica,
                           colesterol, glucosa, fumador, alcohol, actividad, cardiovascular]])
        
        # Predicción
        resultado_modelo = model.predict(datos)[0]

        # Clasificación personalizada de presión
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

        clasificacion_presion = clasificar_presion(sistolica, diastolica)

        # Cálculo de IMC
        estatura_m = estatura / 100
        imc = peso / (estatura_m ** 2)
        imc = round(imc, 1)

        # Gráfico de comparación
        plt.figure(figsize=(6, 4))
        valores_usuario = [sistolica, diastolica, imc]
        valores_ideales = [120, 80, 24.9]
        etiquetas = ['Sistólica', 'Diastólica', 'IMC']
        x = np.arange(len(etiquetas))

        plt.bar(x - 0.2, valores_usuario, width=0.4, label='Tú')
        plt.bar(x + 0.2, valores_ideales, width=0.4, label='Ideal')
        plt.xticks(x, etiquetas)
        plt.ylabel('Valores')
        plt.title('Comparación de tu presión e IMC vs valores ideales')
        plt.legend()
        plt.tight_layout()
        os.makedirs('app/static', exist_ok=True)
        plt.savefig('app/static/grafico_resultado.png')
        plt.close()

        return render_template('resultado.html',
                               resultado=resultado_modelo,
                               clasificacion=clasificacion_presion,
                               imc=imc,
                               edad=edad,
                               genero="Femenino" if genero == 1 else "Masculino")
    
    except Exception as e:
        return f"Ocurrió un error: {str(e)}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
