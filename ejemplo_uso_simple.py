"""
EJEMPLO SIMPLE DE USO
=====================
Este script muestra cómo usar el clasificador profesional paso a paso
"""

from clasificador_profesional import ClasificadorProfesional, Config

# ========================================
# OPCIÓN 1: USO CON DATOS SINTÉTICOS (DEMO)
# ========================================

def demo_rapida():
    """Demo rápida con datos sintéticos"""
    from sklearn.datasets import make_classification
    import numpy as np
    
    print("🚀 Iniciando demo rápida...")
    print()
    
    # Creo datos de ejemplo
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_classes=3,
        random_state=42
    )
    
    # Configuro el clasificador
    config = Config()
    config.epochs = 30  # Menos épocas para demo rápida
    config.hidden_layers = [32, 32]  # Red más pequeña
    
    # Creo el clasificador
    clf = ClasificadorProfesional(config)
    
    # Preparo datos
    X_train, X_val, y_train, y_val = clf.preparar_datos(X, y)
    
    # Creo y entreno el modelo
    clf.crear_modelo(input_size=X.shape[1], output_size=len(np.unique(y)))
    clf.entrenar(X_train, y_train, X_val, y_val)
    
    # Evalúo
    clf.evaluar(X_val, y_val)
    clf.plot_historial()
    
    # Guardo el modelo
    ruta = clf.guardar_modelo("demo_rapida")
    
    print("\n✅ ¡Demo completada!")
    print(f"📁 Modelo guardado en: {ruta}")
    
    return clf


# ========================================
# OPCIÓN 2: USO CON TU ARCHIVO CSV
# ========================================

def entrenar_con_csv(ruta_csv, columna_objetivo):
    """
    Entrena un modelo con tus propios datos CSV
    
    Args:
        ruta_csv: Ruta a tu archivo CSV
        columna_objetivo: Nombre de la columna que quieres predecir
    """
    import numpy as np
    
    print(f"📊 Entrenando modelo con: {ruta_csv}")
    print()
    
    # Creo el clasificador
    clf = ClasificadorProfesional()
    
    # Cargo los datos desde tu CSV
    X, y = clf.cargar_datos_csv(
        ruta_csv=ruta_csv,
        columna_objetivo=columna_objetivo
        # columnas_excluir=["id"]  # Descomenta si tienes columnas que no quieres usar
    )
    
    # Preparo los datos
    X_train, X_val, y_train, y_val = clf.preparar_datos(X, y)
    
    # Creo el modelo
    clf.crear_modelo(input_size=X.shape[1], output_size=len(np.unique(y)))
    
    # Entreno
    clf.entrenar(X_train, y_train, X_val, y_val)
    
    # Evalúo
    clf.evaluar(X_val, y_val)
    clf.plot_historial()
    
    # Guardo el modelo
    ruta = clf.guardar_modelo("modelo_csv")
    
    print("\n✅ ¡Entrenamiento completado!")
    print(f"📁 Modelo guardado en: {ruta}")
    
    return clf, ruta


# ========================================
# OPCIÓN 3: CARGAR MODELO Y PREDECIR
# ========================================

def predecir_con_modelo_guardado(ruta_modelo, datos_nuevos):
    """
    Carga un modelo guardado y hace predicciones
    
    Args:
        ruta_modelo: Ruta al modelo guardado (.pkl)
        datos_nuevos: Datos para predecir (array o DataFrame)
    """
    print(f"🔮 Haciendo predicciones con: {ruta_modelo}")
    print()
    
    # Cargo el clasificador y el modelo
    clf = ClasificadorProfesional()
    clf.cargar_modelo(ruta_modelo)
    
    # Hago predicciones
    predicciones, probabilidades = clf.predecir(datos_nuevos)
    
    # Muestro los resultados
    print("\n📋 RESULTADOS DE LAS PREDICCIONES:")
    print("=" * 60)
    for i, (pred, probs) in enumerate(zip(predicciones, probabilidades)):
        confianza = probs.max() * 100
        print(f"Ejemplo {i+1}:")
        print(f"  → Predicción: {pred}")
        print(f"  → Confianza: {confianza:.2f}%")
        print(f"  → Probabilidades: {dict(zip(clf.label_encoder.classes_, probs))}")
        print()
    
    return predicciones, probabilidades


# ========================================
# EJECUCIÓN PRINCIPAL
# ========================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "🎓 CLASIFICADOR PROFESIONAL")
    print(" " * 20 + "Ejemplos de Uso")
    print("=" * 70)
    print()
    print("Selecciona una opción:")
    print()
    print("1️⃣  Demo rápida con datos sintéticos")
    print("2️⃣  Entrenar con tu archivo CSV")
    print("3️⃣  Cargar modelo y hacer predicciones")
    print()
    
    opcion = input("Ingresa el número de opción (1, 2 o 3): ").strip()
    print()
    
    if opcion == "1":
        # Demo rápida
        clf = demo_rapida()
        
    elif opcion == "2":
        # Entrenar con CSV
        ruta = input("Ingresa la ruta de tu archivo CSV: ").strip()
        columna = input("Ingresa el nombre de la columna objetivo: ").strip()
        clf, ruta_modelo = entrenar_con_csv(ruta, columna)
        
    elif opcion == "3":
        # Predecir con modelo guardado
        ruta = input("Ingresa la ruta del modelo guardado (.pkl): ").strip()
        
        print("\nOpciones para los datos nuevos:")
        print("1. Usar datos de ejemplo")
        print("2. Cargar desde CSV")
        sub_opcion = input("Selecciona (1 o 2): ").strip()
        
        if sub_opcion == "1":
            # Uso datos sintéticos de ejemplo
            from sklearn.datasets import make_classification
            X_nuevo, _ = make_classification(n_samples=5, n_features=15, n_informative=10, random_state=99)
            predecir_con_modelo_guardado(ruta, X_nuevo)
        else:
            import pandas as pd
            ruta_csv = input("Ingresa la ruta del CSV con datos nuevos: ").strip()
            df = pd.read_csv(ruta_csv)
            predecir_con_modelo_guardado(ruta, df)
    
    else:
        print("❌ Opción no válida")
    
    print("\n" + "=" * 70)
    print(" " * 25 + "¡Hasta pronto!")
    print("=" * 70)

