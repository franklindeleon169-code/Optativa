"""
ENTRENAMIENTO CON DATOS DE EJEMPLO
===================================
Este script entrena un modelo con el archivo datos_ejemplo.csv
para predecir la categoría de empleado (Junior, Medio, Senior)
"""

from clasificador_profesional import ClasificadorProfesional, Config
import numpy as np

def main():
    print("\n" + "=" * 70)
    print(" " * 15 + "🎯 ENTRENAMIENTO CON DATOS DE EJEMPLO")
    print(" " * 20 + "Dataset: Categorías de Empleados")
    print("=" * 70)
    print()
    
    # Configuro el clasificador para este problema específico
    config = Config()
    config.epochs = 40  # 40 épocas es suficiente para este dataset
    config.hidden_layers = [32, 16]  # Red moderada
    config.learning_rate = 0.001
    config.dropout_rate = 0.2
    
    print("⚙️  CONFIGURACIÓN DEL MODELO:")
    print(f"   • Épocas: {config.epochs}")
    print(f"   • Arquitectura: {config.hidden_layers}")
    print(f"   • Learning Rate: {config.learning_rate}")
    print(f"   • Dropout: {config.dropout_rate}")
    print()
    
    # Creo el clasificador
    clf = ClasificadorProfesional(config)
    
    # Cargo los datos del CSV de ejemplo
    print("📊 El dataset contiene:")
    print("   • edad: Edad del empleado")
    print("   • salario: Salario anual")
    print("   • experiencia_anos: Años de experiencia")
    print("   • educacion: Nivel educativo (1=Básico, 2=Medio, 3=Superior)")
    print("   • horas_trabajo: Horas trabajadas por semana")
    print("   • satisfaccion: Nivel de satisfacción (1-10)")
    print("   • rendimiento: Porcentaje de rendimiento")
    print("   • proyectos_completados: Número de proyectos completados")
    print("   • categoria_empleado: Junior, Medio o Senior (OBJETIVO)")
    print()
    
    X, y = clf.cargar_datos_csv(
        ruta_csv="datos_ejemplo.csv",
        columna_objetivo="categoria_empleado"
    )
    
    # Preparo los datos
    X_train, X_val, y_train, y_val = clf.preparar_datos(X, y)
    
    # Creo el modelo
    # input_size = número de características (8 columnas)
    # output_size = número de clases (3: Junior, Medio, Senior)
    clf.crear_modelo(
        input_size=X.shape[1],
        output_size=len(np.unique(y))
    )
    
    # Entreno el modelo
    print("🎓 Interpretación del entrenamiento:")
    print("   • Loss (Pérdida): Qué tan equivocado está el modelo")
    print("     → Valores bajos son mejores")
    print("     → Debe disminuir con el tiempo")
    print()
    print("   • Accuracy (Precisión): % de predicciones correctas")
    print("     → Valores altos son mejores")
    print("     → Debe aumentar con el tiempo")
    print()
    
    clf.entrenar(X_train, y_train, X_val, y_val)
    
    # Evalúo el modelo
    print("\n📊 INTERPRETACIÓN DE RESULTADOS:")
    print("-" * 70)
    
    accuracy, f1, cm = clf.evaluar(X_val, y_val)
    
    print("\n💡 ¿Qué significan estas métricas?")
    print()
    print("• Precision (Precisión): De los que predije como X, ¿cuántos son realmente X?")
    print("  → Ejemplo: Si predigo 10 como 'Senior' y 8 realmente lo son → 80% precisión")
    print()
    print("• Recall (Exhaustividad): De todos los que son realmente X, ¿cuántos encontré?")
    print("  → Ejemplo: Si hay 10 'Senior' y encontré 8 → 80% recall")
    print()
    print("• F1-Score: Balance entre Precision y Recall")
    print("  → Es el promedio armónico de ambos")
    print()
    print("• Support: Cuántos ejemplos hay de cada clase en validación")
    print()
    
    # Grafico el historial
    clf.plot_historial()
    
    # Guardo el modelo
    ruta_modelo = clf.guardar_modelo("modelo_empleados")
    
    # Demostración de predicciones
    print("\n" + "=" * 70)
    print("🔮 DEMOSTRACIÓN: PREDICCIONES EN NUEVOS EMPLEADOS")
    print("=" * 70)
    print()
    
    # Creo algunos ejemplos de nuevos empleados ficticios
    nuevos_empleados = np.array([
        # [edad, salario, experiencia, educacion, horas, satisfaccion, rendimiento, proyectos]
        [23, 30000, 0.5, 1, 38, 6, 65, 1],   # Empleado nuevo
        [35, 65000, 9, 3, 48, 8, 88, 22],    # Empleado experimentado
        [52, 110000, 28, 3, 60, 10, 99, 70], # Empleado muy senior
    ])
    
    predicciones, probabilidades = clf.predecir(nuevos_empleados)
    
    descripciones = [
        "Empleado Nuevo: 23 años, $30K, 6 meses exp.",
        "Empleado Experimentado: 35 años, $65K, 9 años exp.",
        "Empleado Muy Senior: 52 años, $110K, 28 años exp."
    ]
    
    for i, (desc, pred, probs) in enumerate(zip(descripciones, predicciones, probabilidades)):
        print(f"\n{i+1}. {desc}")
        print(f"   → Predicción: {pred}")
        print(f"   → Confianza: {probs.max()*100:.1f}%")
        print(f"   → Probabilidades:")
        for clase, prob in zip(clf.label_encoder.classes_, probs):
            barra = "█" * int(prob * 30)
            print(f"      {clase:8s}: {barra} {prob*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO CON ÉXITO")
    print("=" * 70)
    print()
    print("📁 Archivos generados:")
    print(f"   • Modelo: {ruta_modelo}")
    print(f"   • Gráficas en carpeta: resultados/")
    print()
    print("💡 Próximos pasos:")
    print("   1. Revisa las gráficas generadas")
    print("   2. Analiza la matriz de confusión")
    print("   3. Usa el modelo guardado para hacer predicciones")
    print("   4. Prueba con tus propios datos CSV")
    print()
    print("🎓 Para usar este modelo más tarde:")
    print("   clf = ClasificadorProfesional()")
    print(f"   clf.cargar_modelo('{ruta_modelo}')")
    print("   predicciones = clf.predecir(mis_datos)")
    print()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("\n❌ ERROR: No se encontró el archivo 'datos_ejemplo.csv'")
        print("   Asegúrate de estar en la carpeta correcta.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n💡 Posibles soluciones:")
        print("   1. Verifica que todas las librerías estén instaladas")
        print("   2. Ejecuta: pip install torch scikit-learn pandas matplotlib seaborn joblib")
        print("   3. Revisa que el archivo CSV esté en la misma carpeta")

