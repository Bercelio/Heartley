# Heartly – Plataforma Inteligente de Diagnóstico Cardiovascular

![Heartly Logo](app/static/logo.png)

**Predicción temprana, corazón sano.**  
*Cuida tu corazón, cambia tu historia.*

---

## 🚀 Propósito del Proyecto
Heartly es una plataforma desarrollada con inteligencia artificial para predecir de forma temprana el riesgo cardiovascular y entregar recomendaciones personalizadas a los pacientes.

Este proyecto nace como una iniciativa personal de pacientes hipertensos que, conscientes de los riesgos de esta enfermedad silenciosa, quisieron construir una herramienta accesible, visual, y empática para acompañar a otros en su proceso de cuidado y prevención.

---

## 🤖 Tecnologías utilizadas
- Python 3
- Flask (para el backend web)
- scikit-learn (modelo predictivo Random Forest)
- joblib (para cargar el modelo)
- matplotlib (para las gráficas comparativas)
- HTML + CSS (interfaz web)

---

## 🔬 Variables del modelo
El modelo fue entrenado con 17 variables, incluyendo:
- Edad, género, estatura, peso
- Presión sistólica y diastólica
- IMC (calculado)
- Colesterol, glucosa, actividad física, tabaquismo, alcoholismo
- Fase de hipertensión (calculada según guías clínicas internacionales)

---

## 📊 Resultados del modelo
- Precisión: **70.7%**
- Algoritmo: RandomForestClassifier
- Variable objetivo: presencia de enfermedad cardiovascular (`cardio`)

---

## 🚹 Estructura del proyecto
```
Heartly/
├── app/
│   ├── templates/
│   │   ├── index.html
│   │   └── resultado.html
│   ├── static/
│   │   └── logo.png
│   │   └── grafica_resultado.png (auto-generada)
│   └── main.py
├── model/
│   └── Heartly_model_17col.joblib
├── heartly_dataset.csv (opcional)
├── requirements.txt
├── README.md
├── .gitignore
└── Procfile
```

---

## 🚫 .gitignore
```
__pycache__/
*.pyc
.env
.DS_Store
.ipynb_checkpoints/
```

---

## 📂 Requisitos de instalación
```
pip install -r requirements.txt
```

---

## ⚖️ Ejecución local
```bash
cd Heartly
python app/main.py
```
Abre en el navegador: http://localhost:5000

---

## 🚧 Despliegue en Render
1. Crear repositorio en GitHub con esta estructura
2. Subir todos los archivos incluyendo el modelo `.joblib`
3. Crear una nueva app en [Render.com](https://render.com/)
4. Seleccionar: **Web Service**
5. Conectar el repositorio
6. En Start Command: `python app/main.py`
7. Render detecta automáticamente el `requirements.txt` y el `Procfile`

---

## 🌟 Créditos
Desarrollado por **Equipo Heartly - Bercelio Bolaños Rendón, Yeniffer Martínez Díaz, Ennys García Camelo, Esteban Cogollo Urzola, Luis Daniel Torres** como parte de un proyecto con propósito social: 
> "Hacer de la inteligencia artificial una herramienta accesible para cuidar lo más importante: el corazón."

---

❤️ Gracias por visitar Heartly. ¡Súbete a esta revolución de salud inteligente!
