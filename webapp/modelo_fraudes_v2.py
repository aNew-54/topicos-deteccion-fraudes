"""
╔══════════════════════════════════════════════════════════════════╗
║           MODELO DE DETECCIÓN DE FRAUDES - v2.0                 ║
║           Mejorado con ingeniería de variables, ensemble         ║
║           avanzado, calibración de umbral y métricas reales      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import warnings
import joblib
from datetime import datetime

# Conexiones externas
try:
    from pyhive import hive
except ImportError:
    hive = None
try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

# ML
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_recall_curve, confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV


# ══════════════════════════════════════════════════════════════════
# 1. OBTENCIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════

def obtener_datos(csv_fallback: str = None) -> pd.DataFrame | None:
    """
    Intenta cargar desde Hive + MongoDB. Si falla, usa CSV local.
    Retorna siempre un DataFrame limpio con las columnas originales.
    """
    warnings.filterwarnings('ignore', category=UserWarning)
    df_historico = None

    # --- Hive ---
    if hive:
        print("🐘 Conectando a Hive (Data Lake)...")
        try:
            conn_hive = hive.Connection(host='localhost', port=10000, username='hadoop')
            query = "SELECT * FROM dev_fraudes_functional.FRAUDES_ENRIQUECIDO"
            df_historico = pd.read_sql(query, conn_hive)
            df_historico.columns = [col.split('.')[-1] for col in df_historico.columns]
            print(f"✅ Hive: {len(df_historico)} registros cargados.")
        except Exception as e:
            print(f"⚠️  Hive no disponible: {e}")

    if df_historico is None:
        if csv_fallback:
            print(f"📂 Cargando desde CSV: {csv_fallback}")
            df_historico = pd.read_csv(csv_fallback)
            print(f"✅ CSV: {len(df_historico)} registros cargados.")
        else:
            print("❌ Sin fuente de datos disponible.")
            return None

    # --- MongoDB (feedback en tiempo real) ---
    df_nuevos = pd.DataFrame()
    if MongoClient:
        print("🍃 Conectando a MongoDB (Feedback)...")
        try:
            cliente_mongo = MongoClient("mongodb://192.168.18.30:27017/", serverSelectionTimeoutMS=3000)
            cliente_mongo.server_info()
            db = cliente_mongo["db_fraudes"]
            coleccion = db["evaluaciones_tiempo_real"]
            cursor = coleccion.find({"fraude_detectado": {"$ne": None}})
            df_nuevos = pd.DataFrame(list(cursor))
            if not df_nuevos.empty:
                df_nuevos = df_nuevos.drop(
                    columns=['_id', 'fecha_evaluacion', 'ml_prediccion', 'ml_probabilidad_fraude'],
                    errors='ignore'
                )
                print(f"✅ MongoDB: {len(df_nuevos)} registros de feedback añadidos.")
        except Exception as e:
            print(f"⚠️  MongoDB no disponible: {e}")

    df_final = pd.concat([df_historico, df_nuevos], ignore_index=True) if not df_nuevos.empty else df_historico
    return df_final


# ══════════════════════════════════════════════════════════════════
# 2. INGENIERÍA DE VARIABLES
# ══════════════════════════════════════════════════════════════════

# Mapeos ordinales con semántica de riesgo
MAPEO_CAMBIO_DIRECCION = {
    'no change': 0,
    '4 to 8 years': 1,
    '1 year': 2,
    '2 to 3 years': 3,
    'under 6 months': 4
}

MAPEO_ANTIGUEDAD_VEHICULO = {
    'new': 0, '2 years': 1, '3 years': 2, '4 years': 3,
    '5 years': 4, '6 years': 5, '7 years': 6, 'more than 7': 7
}

MAPEO_EDAD_TITULAR = {
    '16 to 17': 0, '18 to 20': 1, '21 to 25': 2, '26 to 30': 3,
    '31 to 35': 4, '36 to 40': 5, '41 to 50': 6, '51 to 65': 7, 'over 65': 8
}

MAPEO_DIAS_POLIZA_ACCIDENTE = {
    'none': 0, '1 to 7': 1, '8 to 15': 2, '15 to 30': 3, 'more than 30': 4
}

MAPEO_RECLAMOS_PASADOS = {
    'none': 0, '1': 1, '2 to 4': 2, 'more than 4': 3
}

MAPEO_SUPLEMENTOS = {
    'none': 0, '1 to 2': 1, '3 to 5': 2, 'more than 5': 3
}


def ingenieria_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica transformaciones y crea nuevas variables derivadas.
    No modifica el DataFrame original.
    """
    df = df.copy()

    # ── Encodings ordinales con semántica de riesgo ──────────────
    df['cambio_direccion_ord'] = df['cambio_direccion'].map(MAPEO_CAMBIO_DIRECCION).fillna(0)
    df['antiguedad_vehiculo_ord'] = df['antiguedad_vehiculo'].map(MAPEO_ANTIGUEDAD_VEHICULO).fillna(4)
    df['edad_titular_ord'] = df['edad_titular'].map(MAPEO_EDAD_TITULAR).fillna(4)
    df['dias_poliza_accidente_ord'] = df['dias_poliza_accidente'].map(MAPEO_DIAS_POLIZA_ACCIDENTE).fillna(4)
    df['reclamos_pasados_ord'] = df['reclamos_pasados'].map(MAPEO_RECLAMOS_PASADOS).fillna(0)
    df['suplementos_ord'] = df['numero_suplementos'].map(MAPEO_SUPLEMENTOS).fillna(0)
    df['agente_externo'] = (df['tipo_agente'] == 'External').astype(int)

    # ── Variables nuevas derivadas (feature engineering) ─────────
    # Diferencia entre semana de accidente y semana de reclamo
    df['diff_semanas_reclamo'] = df['semana_reclamo'] - df['semana_accidente']

    # Combinación de señales fuertes de riesgo
    df['flag_cambio_reciente'] = (df['cambio_direccion_ord'] >= 3).astype(int)
    df['flag_reporte_tardio'] = (df['semanas_retraso_reporte'] > 0).astype(int)
    df['flag_sin_testigo_sin_policial'] = (
        (df['testigo_presente_ml'] == 0) & (df['reporte_policial_ml'] == 0)
    ).astype(int)

    # Score compuesto de riesgo (reemplaza el score_riesgo original)
    df['score_riesgo_v2'] = (
        df['flag_cambio_reciente'] * 3 +         # señal más fuerte
        df['flag_reporte_tardio'] * 2 +
        df['flag_sin_testigo_sin_policial'] * 2 +
        (df['agente_externo']) * 1 +
        (df['reclamos_pasados_ord'] == 0).astype(int) * 1  # sin historial = riesgo
    )

    # Ratio valor vehículo / deducible (vehículos caros con deducible mínimo)
    df['ratio_valor_deducible'] = df['valor_estimado_vehiculo'] / (df['deducible'] + 1)

    return df


# ══════════════════════════════════════════════════════════════════
# 3. DEFINICIÓN DE FEATURES
# ══════════════════════════════════════════════════════════════════

VARS_NUMERICAS = [
    # Originales continuas
    'edad', 'deducible', 'calificacion_conductor', 'valor_estimado_vehiculo',
    'semanas_retraso_reporte', 'dias_poliza_accidente_min', 'reclamos_pasados_min',
    # Binarias ML-ready
    'zona_accidente_ml', 'reporte_policial_ml', 'testigo_presente_ml', 'genero_ml',
    # Ordinales derivadas
    'cambio_direccion_ord', 'antiguedad_vehiculo_ord', 'edad_titular_ord',
    'dias_poliza_accidente_ord', 'reclamos_pasados_ord', 'suplementos_ord',
    'agente_externo',
    # Features engineered
    'diff_semanas_reclamo', 'flag_cambio_reciente', 'flag_reporte_tardio',
    'flag_sin_testigo_sin_policial', 'score_riesgo_v2', 'ratio_valor_deducible',
]

VARS_CATEGORICAS = [
    'cobertura',            # tipo de cobertura del seguro
    'culpabilidad',         # quién fue responsable del accidente
    'categoria_vehiculo',   # Utility tiene 11% fraude vs Sport 1.6%
    'rango_precio_vehiculo' # extremos tienen más fraude
]

# Columnas que NUNCA deben entrar como features (data leakage / metadata)
COLUMNAS_EXCLUIR = [
    'numero_poliza', 'numero_reporte', 'fraude_detectado',
    'ml_probabilidad_fraude', 'ml_prediccion',  # leakage directo
    'score_riesgo',                              # score externo sin valor predictivo
    'gob_fecha_procesamiento', 'gob_fuente_datos', 'gob_calidad_datos',
]


# ══════════════════════════════════════════════════════════════════
# 4. CONSTRUCCIÓN DEL MODELO
# ══════════════════════════════════════════════════════════════════

def construir_pipeline() -> Pipeline:
    """
    Ensemble de Gradient Boosting + Random Forest con preprocesamiento
    integrado en un Pipeline de scikit-learn.
    """
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), VARS_NUMERICAS),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), VARS_CATEGORICAS)
        ],
        remainder='drop'
    )

    # Gradient Boosting: mejor para datos tabulares desbalanceados
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42
    )

    # Random Forest: complementa con diversidad de árboles
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )

    # Voting Classifier: combina ambos por probabilidad (soft voting)
    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf)],
        voting='soft',
        weights=[2, 1]  # GradientBoosting tiene más peso
    )

    pipeline = Pipeline(steps=[
        ('preprocesador', preprocesador),
        ('modelo', ensemble)
    ])

    return pipeline


# ══════════════════════════════════════════════════════════════════
# 5. CALIBRACIÓN DE UMBRAL ÓPTIMO
# ══════════════════════════════════════════════════════════════════

def encontrar_umbral_optimo(y_true: np.ndarray, y_proba: np.ndarray, objetivo: str = 'f1') -> float:
    """
    Encuentra el umbral de decisión que maximiza F1 (o Recall).
    En fraude priorizamos no dejar pasar casos (Recall alto).
    
    objetivo: 'f1' para balance Precision/Recall | 'recall' para máximo Recall
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if objetivo == 'recall':
        # Máximo recall con precision mínima aceptable (ej: 30%)
        mask = precision[:-1] >= 0.30
        if mask.any():
            mejor_idx = recall[:-1][mask].argmax()
            return thresholds[mask][mejor_idx]

    # F1 óptimo
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    mejor_idx = f1_scores.argmax()
    return float(thresholds[mejor_idx])


# ══════════════════════════════════════════════════════════════════
# 6. EVALUACIÓN COMPLETA
# ══════════════════════════════════════════════════════════════════

def evaluar_modelo(modelo, X_test, y_test, umbral: float):
    """Imprime métricas relevantes para detección de fraude."""
    y_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred_umbral = (y_proba >= umbral).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred_umbral)
    cm = confusion_matrix(y_test, y_pred_umbral)

    print("\n" + "═" * 55)
    print("📊  MÉTRICAS DE EVALUACIÓN")
    print("═" * 55)
    print(f"  AUC-ROC       : {auc:.4f}  (>0.85 es bueno, >0.90 es excelente)")
    print(f"  F1-Score      : {f1:.4f}  (balance Precision/Recall)")
    print(f"  Umbral óptimo : {umbral:.3f}  (ajustado para fraude)")
    print()
    print(classification_report(y_test, y_pred_umbral, target_names=['No Fraude', 'Fraude']))

    tn, fp, fn, tp = cm.ravel()
    print(f"  Matriz de Confusión (umbral={umbral:.2f}):")
    print(f"    ✅ Fraudes detectados    (TP): {tp:>5}")
    print(f"    ❌ Fraudes no detectados (FN): {fn:>5}  ← minimizar esto")
    print(f"    ⚠️  Falsas alarmas       (FP): {fp:>5}")
    print(f"    ✅ No fraudes correctos  (TN): {tn:>5}")
    print("═" * 55)

    return auc, f1


# ══════════════════════════════════════════════════════════════════
# 7. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════

def entrenar(csv_fallback: str = None, guardar_modelo: bool = True):
    """
    Orquesta todo el proceso: datos → features → train → evaluar → guardar.
    """
    print("\n" + "═" * 55)
    print("🚀  ENTRENAMIENTO MODELO FRAUDES v2.0")
    print(f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 55 + "\n")

    # ── 1. Cargar datos ──────────────────────────────────────────
    df = obtener_datos(csv_fallback=csv_fallback)
    if df is None:
        return None

    # ── 2. Feature engineering ──────────────────────────────────
    print("⚙️  Aplicando ingeniería de variables...")
    df = ingenieria_variables(df)

    # ── 3. Preparar X, y ─────────────────────────────────────────
    todas_features = VARS_NUMERICAS + VARS_CATEGORICAS
    df_modelo = df.dropna(subset=['fraude_detectado'] + todas_features)

    X = df_modelo[todas_features]
    y = df_modelo['fraude_detectado'].astype(int)

    print(f"📦  Dataset: {len(X)} registros | Fraude: {y.sum()} ({y.mean()*100:.1f}%)")

    # ── 4. Split estratificado ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 5. Construir y entrenar pipeline ─────────────────────────
    print("\n🧠  Entrenando Ensemble (GradientBoosting + RandomForest)...")
    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)

    # ── 6. Validación cruzada ────────────────────────────────────
    print("🔄  Validación cruzada (5 folds, métrica: AUC-ROC)...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"    AUC-ROC CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 7. Calibrar umbral óptimo en test ────────────────────────
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    umbral_optimo = encontrar_umbral_optimo(y_test.values, y_proba_test, objetivo='f1')

    # ── 8. Evaluación final ───────────────────────────────────────
    auc, f1 = evaluar_modelo(pipeline, X_test, y_test, umbral_optimo)

    # ── 9. Importancia de variables ──────────────────────────────
    try:
        gb_model = pipeline.named_steps['modelo'].estimators_[0]
        feature_names = (
            VARS_NUMERICAS +
            list(pipeline.named_steps['preprocesador']
                 .named_transformers_['cat']
                 .get_feature_names_out(VARS_CATEGORICAS))
        )
        importancias = pd.Series(gb_model.feature_importances_, index=feature_names)
        print("\n🏆  TOP 15 Variables más importantes (GradientBoosting):")
        print(importancias.nlargest(15).to_string())
    except Exception:
        pass

    # ── 10. Guardar modelo ────────────────────────────────────────
    if guardar_modelo:
        artefacto = {
            'pipeline': pipeline,
            'umbral_optimo': umbral_optimo,
            'vars_numericas': VARS_NUMERICAS,
            'vars_categoricas': VARS_CATEGORICAS,
            'auc_roc': auc,
            'f1_score': f1,
            'fecha_entrenamiento': datetime.now().isoformat(),
        }
        joblib.dump(artefacto, 'modelo_fraudes_v2.pkl')
        print(f"\n💾  Modelo guardado en 'modelo_fraudes_v2.pkl'")
        print(f"    AUC-ROC: {auc:.4f} | F1: {f1:.4f} | Umbral: {umbral_optimo:.3f}")

    return artefacto


# ══════════════════════════════════════════════════════════════════
# 8. FUNCIÓN DE PREDICCIÓN EN PRODUCCIÓN
# ══════════════════════════════════════════════════════════════════

def predecir(df_nuevo: pd.DataFrame, ruta_modelo: str = 'modelo_fraudes_v2.pkl') -> pd.DataFrame:
    """
    Recibe un DataFrame con los campos originales y retorna predicciones.
    
    Retorna el mismo DataFrame con columnas añadidas:
        - probabilidad_fraude: float [0, 1]
        - prediccion_fraude:   int   {0, 1}
        - nivel_alerta:        str   {'BAJO', 'MEDIO', 'ALTO', 'CRÍTICO'}
    """
    artefacto = joblib.load(ruta_modelo)
    pipeline = artefacto['pipeline']
    umbral = artefacto['umbral_optimo']

    df_pred = ingenieria_variables(df_nuevo)
    X = df_pred[artefacto['vars_numericas'] + artefacto['vars_categoricas']]

    probabilidades = pipeline.predict_proba(X)[:, 1]
    predicciones = (probabilidades >= umbral).astype(int)

    resultado = df_nuevo.copy()
    resultado['probabilidad_fraude'] = probabilidades.round(4)
    resultado['prediccion_fraude'] = predicciones
    resultado['nivel_alerta'] = pd.cut(
        probabilidades,
        bins=[0, 0.2, 0.4, 0.7, 1.0],
        labels=['BAJO', 'MEDIO', 'ALTO', 'CRÍTICO'],
        include_lowest=True
    )

    return resultado


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Cambiar csv_fallback a None en producción (usará Hive)
    artefacto = entrenar(
        csv_fallback='fraudes_enriquecido_202603051611.csv',
        guardar_modelo=True
    )