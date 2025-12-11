"""
=====================================
SISTEMA DE CLASIFICACIÓN PROFESIONAL
Entrenamiento de Redes Neuronales con Datos Reales
MODIFICADO: 11/12/2025
=====================================

Este programa puede:
- Cargar datos desde archivos CSV con manejo de categóricos y nulos.
- Entrenar modelos con tus propios datos.
- Guardar y cargar modelos entrenados.
- Hacer predicciones sobre nuevos datos.
- Mostrar métricas profesionales y generar visualizaciones.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import os
from datetime import datetime
import sys

# ========================================
# CONFIGURACIÓN PRINCIPAL
# ========================================

class Config:
    """
    Configuración del modelo - Puedo modificar estos valores según mis necesidades
    """
    # Arquitectura de la red
    hidden_layers = [128, 64]  # Aumento las neuronas para mayor capacidad
    dropout_rate = 0.3         # Aumento dropout para evitar sobreajuste
    
    # Entrenamiento
    epochs = 100               # Aumento épocas, Early Stopping lo controlará
    learning_rate = 0.0005     # Tasa de aprendizaje ajustada
    batch_size = 64            # Tamaño del lote ajustado
    validation_split = 0.2     # Porcentaje de datos para validación
    
    # Rutas de archivos
    models_dir = "modelos_guardados"
    results_dir = "resultados"
    
    # Opciones
    use_gpu = True
    random_seed = 42
    
    def display(self):
        """Imprime la configuración actual."""
        print("=" * 60)
        print("⚙️ CONFIGURACIÓN ACTUAL")
        print("-" * 60)
        print(f"  Capas Ocultas: {self.hidden_layers}")
        print(f"  Dropout Rate: {self.dropout_rate}")
        print(f"  Épocas Máx: {self.epochs}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Validación Split: {self.validation_split * 100:.0f}%")
        print(f"  Uso de GPU: {self.use_gpu}")
        print("-" * 60)

# ========================================
# CLASE DE RED NEURONAL FLEXIBLE
# ========================================

class FlexibleMLP(nn.Module):
    # La clase FlexibleMLP se mantiene igual a la original
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.2):
        super(FlexibleMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ========================================
# CLASE PRINCIPAL DEL CLASIFICADOR
# ========================================

class ClasificadorProfesional:
    """
    Clase principal que maneja todo el proceso de clasificación
    """
    
    def __init__(self, config=Config()):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = self._get_device()
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        self.feature_names = None # Almacenar nombres de características para One-Hot
        
        os.makedirs(config.models_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        
        print("=" * 60)
        print("🚀 CLASIFICADOR PROFESIONAL INICIALIZADO")
        self.config.display()
        print(f"📱 Dispositivo: {self.device}")
        print(f"📂 Modelos guardados en: {config.models_dir}")
        print(f"📊 Resultados guardados en: {config.results_dir}")
        print()
    
    def _get_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    # MODIFICACIÓN CLAVE: Mayor robustez en el manejo de datos
    def cargar_datos_csv(self, ruta_csv, columna_objetivo, columnas_excluir=None):
        """
        Carga, limpia y pre-procesa datos desde un archivo CSV.
        Implementa manejo de nulos y codificación One-Hot para categóricos.
        """
        print("=" * 60)
        print("📁 CARGANDO Y PRE-PROCESANDO DATOS DESDE CSV")
        print("=" * 60)
        
        df = pd.read_csv(ruta_csv)
        print(f"✓ Archivo cargado: {ruta_csv}")
        print(f"✓ Dimensiones iniciales: {df.shape}")
        
        if columna_objetivo not in df.columns:
            raise ValueError(f"La columna objetivo '{columna_objetivo}' no existe en el CSV.")
        
        # 1. Separar la columna objetivo
        y_raw = df[columna_objetivo]
        
        # 2. Excluir columnas y la objetivo
        columnas_a_eliminar = [columna_objetivo]
        if columnas_excluir:
            columnas_a_eliminar.extend(columnas_excluir)
        
        X_df = df.drop(columns=columnas_a_eliminar, errors='ignore')
        print(f"✓ Columnas a procesar: {len(X_df.columns)}")

        # 3. Manejo de valores nulos (Imputación simple: Media para numéricos, Moda para categóricos)
        for col in X_df.columns:
            if X_df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_df[col]):
                    media = X_df[col].mean()
                    X_df[col].fillna(media, inplace=True)
                    print(f"⚠ Imputación: Nulos en '{col}' (numérica) rellenados con la media ({media:.2f}).")
                else:
                    moda = X_df[col].mode()[0]
                    X_df[col].fillna(moda, inplace=True)
                    print(f"⚠ Imputación: Nulos en '{col}' (categórica) rellenados con la moda ('{moda}').")
                    
        # 4. Codificación One-Hot para variables categóricas
        X_df = pd.get_dummies(X_df, drop_first=True)
        self.feature_names = X_df.columns.tolist() # Guardo los nombres de las features
        
        # 5. Convertir a NumPy para procesamiento
        X = X_df.values.astype(np.float32)
        y = y_raw.values
        
        print(f"✓ Dimensiones finales de características (X): {X.shape}")
        print(f"✓ Clases únicas encontradas: {np.unique(y)}")
        print()
        
        return X, y
    
    # Las siguientes funciones se mantienen funcionales con PyTorch
    
    def preparar_datos(self, X, y):
        # ... (Se mantiene igual, solo se corrige un error tipográfico)
        print("=" * 60)
        print("🔧 PREPARANDO DATOS")
        print("=" * 60)
        
        y = self.label_encoder.fit_transform(y)
        print(f"✓ Etiquetas codificadas: {self.label_encoder.classes_}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config.validation_split,
            random_state=self.config.random_seed,
            stratify=y
        )
        
        print(f"✓ Datos de entrenamiento: {X_train.shape[0]} ejemplos")
        print(f"✓ Datos de validación: {X_val.shape[0]} ejemplos")
        
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        print("✓ Datos normalizados (StandardScaler)")
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        # Corrección de un error tipográfico: '9self.device' -> 'self.device'
        y_train = torch.LongTensor(y_train).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        print("✓ Datos convertidos a tensores de PyTorch")
        print()
        
        return X_train, X_val, y_train, y_val
    
    def crear_modelo(self, input_size, output_size):
        # ... (Se mantiene igual)
        print("=" * 60)
        print("🏗 CONSTRUYENDO RED NEURONAL")
        print("=" * 60)
        
        self.model = FlexibleMLP(
            input_size=input_size,
            hidden_layers=self.config.hidden_layers,
            output_size=output_size,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Arquitectura: {input_size} → {' → '.join(map(str, self.config.hidden_layers))} → {output_size}")
        print(f"✓ Parámetros totales: {total_params:,}")
        print(f"✓ Parámetros entrenables: {trainable_params:,}")
        print()
    
    def entrenar(self, X_train, y_train, X_val, y_val):
        # ... (Se mantiene igual: Adam, CrossEntropyLoss, Early Stopping)
        print("=" * 60)
        print("🎯 INICIANDO ENTRENAMIENTO")
        print("=" * 60)
        print()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(self.config.epochs):
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train)
            train_loss = criterion(outputs, y_train)
            
            train_loss.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_predictions = val_outputs.argmax(dim=1)
                val_accuracy = (val_predictions == y_val).float().mean().item()
            
            self.history['train_loss'].append(train_loss.item())
            self.history['val_loss'].append(val_loss.item())
            self.history['val_accuracy'].append(val_accuracy)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Época {epoch+1:03d}/{self.config.epochs} | "
                      f"Loss Entreno: {train_loss.item():.4f} | "
                      f"Loss Val: {val_loss.item():.4f} | "
                      f"Precisión Val: {val_accuracy*100:.2f}%")
            
            if patience_counter >= max_patience:
                print(f"\n⚠ Early stopping activado en época {epoch+1}")
                break
        
        self.model.load_state_dict(self.best_model_state)
        
        print()
        print("=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"🏆 Mejor precisión en validación: {max(self.history['val_accuracy'])*100:.2f}%")
        print()
    
    def evaluar(self, X_val, y_val):
        # ... (Se mantiene igual)
        print("=" * 60)
        print("📊 EVALUACIÓN DEL MODELO")
        print("=" * 60)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_val).argmax(dim=1).cpu().numpy()
            y_true = y_val.cpu().numpy()
        
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='weighted')
        
        print(f"\n🎯 Precisión (Accuracy): {accuracy*100:.2f}%")
        print(f"📈 F1-Score: {f1:.4f}\n")
        
        print("📋 REPORTE DETALLADO POR CLASE:")
        print("-" * 60)
        report = classification_report(
            y_true, predictions,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print(report)
        
        cm = confusion_matrix(y_true, predictions)
        self._plot_confusion_matrix(cm, self.label_encoder.classes_)
        
        return accuracy, f1, cm
    
    def _plot_confusion_matrix(self, cm, class_names):
        # ... (Se mantiene igual)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
        plt.ylabel('Etiqueta Real', fontsize=12)
        plt.xlabel('Predicción', fontsize=12)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.results_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusión guardada en: {path}")
        plt.close()
    
    def plot_historial(self):
        # ... (Se mantiene igual)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self.history['train_loss'], label='Entrenamiento', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Validación', linewidth=2)
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Pérdida (Loss)', fontsize=12)
        axes[0].set_title('Evolución de la Pérdida', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['val_accuracy'], color='green', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('Precisión', fontsize=12)
        axes[1].set_title('Precisión en Validación', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.results_dir, f'training_history_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✓ Historial de entrenamiento guardado en: {path}")
        plt.close()
    
    def guardar_modelo(self, nombre="modelo"):
        # Se añade 'feature_names' al diccionario de guardado
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{nombre}_{timestamp}.pkl"
        ruta = os.path.join(self.config.models_dir, nombre_archivo)
        
        modelo_completo = {
            'model_state_dict': self.model.state_dict(),
            'model_architecture': {
                'input_size': self.model.network[0].in_features,
                'hidden_layers': self.config.hidden_layers,
                'output_size': self.model.network[-1].out_features,
                'dropout_rate': self.config.dropout_rate
            },
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'history': self.history,
            'config': self.config,
            'feature_names': self.feature_names # Nombres de features para predicciones futuras
        }
        
        joblib.dump(modelo_completo, ruta)
        print(f"\n💾 Modelo guardado exitosamente en: {ruta}")
        return ruta
    
    def cargar_modelo(self, ruta):
        # Se añade la carga de 'feature_names'
        print(f"📂 Cargando modelo desde: {ruta}")
        
        modelo_completo = joblib.load(ruta)
        
        arch = modelo_completo['model_architecture']
        self.model = FlexibleMLP(
            input_size=arch['input_size'],
            hidden_layers=arch['hidden_layers'],
            output_size=arch['output_size'],
            dropout_rate=arch['dropout_rate']
        ).to(self.device)
        
        self.model.load_state_dict(modelo_completo['model_state_dict'])
        self.model.eval()
        
        self.scaler = modelo_completo['scaler']
        self.label_encoder = modelo_completo['label_encoder']
        self.history = modelo_completo.get('history', self.history) # Uso .get para compatibilidad
        self.config = modelo_completo.get('config', self.config)
        self.feature_names = modelo_completo.get('feature_names')
        
        print("✓ Modelo cargado exitosamente")
        print(f"✓ Clases: {self.label_encoder.classes_}")
        
        # Re-inicializar config en el clasificador si se ha cargado uno nuevo
        self.__init__(config=self.config)
        
        print()
    
    def predecir(self, X_nuevo):
        """
        Hago predicciones sobre nuevos datos. Se requiere que X_nuevo sea un DataFrame
        para manejar la codificación One-Hot de forma consistente.
        """
        print("\n" + "=" * 60)
        print("🔮 INICIANDO PREDICCIÓN")
        print("=" * 60)
        
        if not isinstance(X_nuevo, pd.DataFrame):
            print("⚠ Advertencia: La predicción requiere un DataFrame para manejar las columnas categóricas.")
            # Asumo que es un array numpy de características numéricas si no es DataFrame
            X_proc = X_nuevo
        else:
            # Reaplicar One-Hot para ser consistente con el entrenamiento
            X_dummy = pd.get_dummies(X_nuevo, drop_first=True)
            
            # Alinear las columnas con las características originales del entrenamiento (importante!)
            if self.feature_names is None:
                raise RuntimeError("El modelo cargado no tiene nombres de características (feature_names).")
            
            X_proc = pd.DataFrame(columns=self.feature_names)
            # Rellenar con los valores que tenga y 0 en las columnas que falten
            for col in self.feature_names:
                if col in X_dummy.columns:
                    X_proc[col] = X_dummy[col]
                else:
                    X_proc[col] = 0.0 # Columna no presente en el ejemplo, se asume 0 (criterio de One-Hot)
            
            X_proc = X_proc.values

        # Normalizo los datos
        X_proc = self.scaler.transform(X_proc)
        
        # Convierto a tensor
        X_tensor = torch.FloatTensor(X_proc).to(self.device)
        
        # Hago la predicción
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = outputs.argmax(dim=1).cpu().numpy()
        
        # Decodifico las etiquetas
        predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions, probabilities


# ========================================
# FUNCIONES DE USO Y MENÚ
# ========================================

def entrenar_con_csv_real(ruta_csv, columna_objetivo, columnas_excluir=None):
    """
    Función para encapsular el flujo de entrenamiento con un CSV.
    """
    try:
        clasificador = ClasificadorProfesional()
        
        X, y = clasificador.cargar_datos_csv(
            ruta_csv=ruta_csv,
            columna_objetivo=columna_objetivo,
            columnas_excluir=columnas_excluir
        )
        
        # Preparar y crear el modelo
        X_train, X_val, y_train, y_val = clasificador.preparar_datos(X, y)
        clasificador.crear_modelo(
            input_size=X_train.shape[1],
            output_size=len(clasificador.label_encoder.classes_)
        )
        
        # Entrenar, evaluar y guardar
        clasificador.entrenar(X_train, y_train, X_val, y_val)
        clasificador.evaluar(X_val, y_val)
        clasificador.plot_historial()
        clasificador.guardar_modelo("modelo_entrenado_csv")
        
    except Exception as e:
        print(f"\n❌ ERROR FATAL DURANTE EL ENTRENAMIENTO: {e}")
        print("Asegúrate de que el CSV está limpio y las columnas son correctas.")

def menu_principal():
    """
    Función principal que presenta un menú interactivo al usuario.
    """
    while True:
        print("\n" + "=" * 60)
        print("          Sistema de Clasificación Profesional")
        print("=" * 60)
        print("1. 🚀 DEMO RÁPIDA (Entrenamiento con Datos Sintéticos)")
        print("2. 💾 ENTRENAR CON CSV REAL (Ej. datos_ejemplo.csv)")
        print("3. 🔮 CARGAR Y PREDECIR (Usar un modelo guardado)")
        print("4. 🚪 SALIR")
        print("-" * 60)
        
        eleccion = input("Elige una opción (1-4): ")
        print("-" * 60)
        
        if eleccion == '1':
            ejemplo_uso_datos_sinteticos()
            
        elif eleccion == '2':
            print("Entrenamiento con CSV. Necesitas especificar los parámetros:")
            ruta = input("Ruta del CSV (ej: datos_ejemplo.csv): ")
            objetivo = input("Nombre de la columna objetivo (ej: categoria_empleado): ")
            excluir_str = input("Columnas a excluir (separadas por coma, ej: id,nombre): ")
            excluir = [c.strip() for c in excluir_str.split(',')] if excluir_str else None
            
            if ruta and objetivo:
                entrenar_con_csv_real(ruta, objetivo, excluir)
            else:
                print("❌ Ruta de archivo y columna objetivo son obligatorias.")
                
        elif eleccion == '3':
            ruta_modelo = input("Ruta del archivo .pkl del modelo guardado: ")
            
            if not os.path.exists(ruta_modelo):
                print(f"❌ Archivo no encontrado: {ruta_modelo}")
                continue
                
            try:
                clasificador = ClasificadorProfesional()
                clasificador.cargar_modelo(ruta_modelo)
                
                print("\nDatos de predicción: Ingresa valores para el ejemplo (en formato CSV o DataFrame).")
                print("Para simular, introduce 'S' para crear 2 ejemplos sintéticos.")
                
                input_pred = input("Opción (S para sintético, o presiona Enter para cancelar): ").upper()
                
                if input_pred == 'S':
                    # Crea un DataFrame sintético con nombres de columnas
                    if clasificador.feature_names is None:
                        # Si no hay nombres, crea un array de la longitud correcta
                        num_features = clasificador.model.network[0].in_features
                        X_prueba = np.random.rand(2, num_features).astype(np.float32)
                        print("⚠ Usando datos sintéticos crudos (sin nombres de columna).")
                    else:
                        # Usar nombres de columnas para probar la consistencia del pre-procesamiento
                        data = {name: np.random.rand(2) for name in clasificador.feature_names}
                        X_prueba = pd.DataFrame(data)
                        # Elimina las columnas con nombres One-Hot para simular datos crudos
                        X_prueba = X_prueba.iloc[:, :5] # Asumimos las primeras 5 son las originales
                        
                    print(f"Prediciendo en {X_prueba.shape[0]} ejemplos:")
                    predicciones, probabilidades = clasificador.predecir(X_prueba)
                    
                    for i, (pred, probs) in enumerate(zip(predicciones, probabilidades)):
                        prob_max = probs.max()
                        clase_max = clasificador.label_encoder.classes_[np.argmax(probs)]
                        print(f"Ejemplo {i+1}: Predicción = {pred} (Clase: {clase_max}, Prob: {prob_max*100:.2f}%)")
                    
            except Exception as e:
                print(f"❌ ERROR AL CARGAR/PREDECIR: {e}")
                
        elif eleccion == '4':
            print("👋 Saliendo del sistema. ¡Hasta pronto!")
            sys.exit()
            
        else:
            print("Opción no válida. Inténtalo de nuevo.")

# Las funciones de ejemplo se mantienen, pero la lógica de 'if __name__ == "__main__":' cambia
def ejemplo_uso_datos_sinteticos():
    """
    Ejemplo de uso con datos sintéticos (para demostración)
    """
    from sklearn.datasets import make_classification
    
    print("\n" + "=" * 60)
    print("🎓 DEMO: ENTRENAMIENTO CON DATOS SINTÉTICOS")
    print("=" * 60)
    print()
    
    # Creo datos sintéticos
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    
    # Inicializo el clasificador
    clasificador = ClasificadorProfesional()
    
    # Preparo los datos
    X_train, X_val, y_train, y_val = clasificador.preparar_datos(X, y)
    
    # Creo el modelo
    clasificador.crear_modelo(
        input_size=X.shape[1],
        output_size=len(np.unique(y))
    )
    
    # Entreno
    clasificador.entrenar(X_train, y_train, X_val, y_val)
    
    # Evalúo
    clasificador.evaluar(X_val, y_val)
    
    # Grafico el historial
    clasificador.plot_historial()
    
    # Guardo el modelo
    ruta_modelo = clasificador.guardar_modelo("modelo_ejemplo")
    
    # Ejemplo de predicción
    print("\n" + "=" * 60)
    print("🔮 EJEMPLO DE PREDICCIÓN")
    print("=" * 60)
    
    X_prueba = X[:5]
    predicciones, probabilidades = clasificador.predecir(pd.DataFrame(X_prueba)) # Lo convierto a DF para ser consistente
    
    print("\nPredicciones en 5 ejemplos (Datos crudos sin nombres de columna):")
    for i, (pred, probs) in enumerate(zip(predicciones, probabilidades)):
        print(f"Ejemplo {i+1}: Predicción = {pred} (probabilidad: {probs.max()*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETADA")
    print("=" * 60)
    print()

# El nuevo punto de entrada ejecuta el menú interactivo
if __name__ == "__main__":
    menu_principal()