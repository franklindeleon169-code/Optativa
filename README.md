# 🎓 SISTEMA DE CLASIFICACIÓN PROFESIONAL CON REDES NEURONALES

Este proyecto proporciona un **sistema completo de Machine Learning** basado en **Redes Neuronales (PyTorch)**, diseñado para clasificar cualquier tipo de datos tabulares (CSV). Es ideal para usar con tus propios datasets, para practicar con un sistema robusto, o como base para proyectos profesionales.

---

## 🚀 Características Principales

* **Implementación de PyTorch:** Utiliza la potencia y flexibilidad de PyTorch para construir y entrenar la red neuronal.
* **Manejo de Datos Tabulares:** Carga archivos CSV, maneja valores nulos y codifica automáticamente columnas categóricas (`one-hot/label encoding`).
* **Preprocesamiento Avanzado:** Incluye escalado de datos (`StandardScaler`) y división automática en conjuntos de entrenamiento y validación.
* **Ciclo de Vida Completo:** Funciones para la creación del modelo, entrenamiento con Early Stopping, evaluación profesional y predicción.
* **Persistencia:** Guarda y carga el modelo, el escalador y el codificador de etiquetas (`LabelEncoder`) usando `joblib`.
* **Visualizaciones:** Genera gráficas de pérdida/precisión y matrices de confusión para un análisis profundo.

---

## 📦 Estructura del Proyecto

| Archivo | Descripción |
| :--- | :--- |
| `clasificador_profesional.py` | **Clase principal** `ClasificadorProfesional` con toda la lógica de ML. |
| `entrenar_con_ejemplo.py` | Script para **entrenar rápidamente** usando el archivo `datos_ejemplo.csv`. |
| `ejemplo_uso_simple.py` | Script **interactivo** para probar la demo, entrenar con tu CSV o predecir con un modelo guardado. |
| `datos_ejemplo.csv` | Dataset de ejemplo de **categorías de empleados** (Junior, Medio, Senior). |
| `LEEME.txt` | El archivo de documentación original con conceptos clave. |

---

## ⚙️ Requisitos

Asegúrate de tener instaladas las siguientes librerías de Python:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn joblib
````

-----

## 💻 Guía de Inicio Rápido

Tienes tres formas de empezar a usar este sistema:

### Opción 1: Demo Rápida (Datos Sintéticos)

Prueba la funcionalidad completa con un dataset generado al instante:

```bash
python ejemplo_uso_simple.py
```

> **Selecciona la opción `1`**

### Opción 2: Entrenar con el CSV de Ejemplo

Usa el archivo `datos_ejemplo.csv` para entrenar un modelo que clasifica empleados:

```bash
python entrenar_con_ejemplo.py
```

Este script guardará el modelo entrenado y generará las gráficas de resultados.

### Opción 3: Entrenar con Tus Propios Datos

1.  Asegúrate de que tus datos estén en un archivo **CSV**.
2.  Ejecuta el script interactivo:
    ```bash
    python ejemplo_uso_simple.py
    ```
    > **Selecciona la opción `2`**
3.  Ingresa la ruta de tu archivo CSV (ej: `mis_datos.csv`).
4.  Ingresa el nombre de la **columna objetivo** (la variable que quieres predecir).

-----

## 💡 Cómo Usar la Clase `ClasificadorProfesional`

Para integrar el sistema en tus propios scripts:

```python
from clasificador_profesional import ClasificadorProfesional, Config
from sklearn.datasets import make_classification
import numpy as np

# 1. Preparar datos (X: características, y: etiquetas)
X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

# 2. Configurar y crear el clasificador
config = Config()
config.epochs = 50
clf = ClasificadorProfesional(config)

# 3. Preparar (Escalado y Split)
X_train, X_val, y_train, y_val = clf.preparar_datos(X, y)

# 4. Crear y entrenar el modelo
clf.crear_modelo(input_size=X.shape[1], output_size=len(np.unique(y)))
clf.entrenar(X_train, y_train, X_val, y_val)

# 5. Evaluar
clf.evaluar(X_val, y_val)

# 6. Predecir
X_nuevo = X[:5] # Nuevos datos para predecir
predicciones, probabilidades = clf.predecir(X_nuevo)

print("Predicciones:", predicciones)
# print("Probabilidades:", probabilidades) 

# 7. Guardar y cargar (si es necesario)
ruta = clf.guardar_modelo("mi_modelo_clasificacion")
# clf.cargar_modelo(ruta)
```


