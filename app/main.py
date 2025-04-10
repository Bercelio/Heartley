import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('heartly_model.pkl')

# Función para generar la gráfica de presión arterial
def generar_grafica(sistolica, diastolica):
    try:
        # Asegurar que la carpeta 'static' exista
        os.makedirs("static", exist_ok=True)

        # Datos para la gráfica (simples, solo "hoy")
        fechas = ["Hoy"]
        sistolicas = [sistolica]
        diastolicas = [diastolica]

        # Crear el gráfico
        plt.figure(figsize=(6, 4))
        plt.plot(fechas, sistolicas, label="Sistólica", marker='o', color='red')
        plt.plot(fechas, diastolicas, label="Diastólica", marker='o', color='orange')
        plt.axhline(y=120, color='green', linestyle='--', label="Sistólica Ideal")
        plt.axhline(y=80, color='blue', linestyle='--', label="Diastólica Ideal")
        plt.title("Presión Arterial del Paciente")
        plt.ylabel("Presión (mmHg)")
        plt.tight_layout()
        plt.legend()

        # Guardar imagen en la ruta correcta
        plt.savefig("static/grafica_paciente.png")
        plt.close()
        print("✅ Gráfico generado exitosamente.")
    except Exception as e:
        print(f"❌ Error al generar la gráfica: {e}")

# Ruta principal con el formulario
@app.route('/')
def formulario():
    return render_template('formulario.html')

# Ruta que procesa el resultado del formulario
@app.route('/resultado', methods=['POST'])
def resultado():
    try:
        # Extraer datos del formulario
        edad = int(request.form['edad'])
        sistolica = int(request.form['sistolica'])
        diastolica = int(request.form['diastolica'])
        peso = float(request.form['peso'])
        estatura = float(request.form['estatura'])
        colesterol = int(request.form['colesterol'])
        glucosa = int(request.form['glucosa'])

        # Calcular IMC
        imc = peso / ((estatura / 100) ** 2)

        # Construir input para el modelo
        datos = np.array([[edad, sistolica, diastolica, peso, estatura, colesterol, glucosa, imc]])

        # Predicción con el modelo
        fase = int(modelo.predict(datos)[0])

        # Generar gráfico
        generar_grafica(sistolica, diastolica)

        # Renderizar resultados
        return render_template('resultado.html',
                               edad=edad,
                               sistolica=sistolica,
                               diastolica=diastolica,
                               peso=peso,
                               estatura=estatura,
                               colesterol=colesterol,
                               glucosa=glucosa,
                               imc=round(imc, 2),
                               fase=fase)
    except Exception as e:
        return f"❌ Error al procesar los datos:<br><pre>{e}</pre>"

# Ejecutar la app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
