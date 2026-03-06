"""
╔══════════════════════════════════════════════════════════════════╗
║           FLASK APP - DETECCIÓN DE FRAUDES v2.0                 ║
║           Integrado con modelo_fraudes_v2.py                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import joblib
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId
import modelo_fraudes_v2 as train_model

app = Flask(__name__)
app.secret_key = "clave_secreta_seguroia_v2"

# ── MongoDB ────────────────────────────────────────────────────────
mongo_client = MongoClient("mongodb://192.168.18.30:27017/")
db = mongo_client["db_fraudes"]
coleccion = db["evaluaciones_tiempo_real"]


# ── Carga del modelo ───────────────────────────────────────────────
def cargar_modelo():
    try:
        return joblib.load('modelo_fraudes_v2.pkl')
    except FileNotFoundError:
        return None

artefacto_modelo = cargar_modelo()


# ══════════════════════════════════════════════════════════════════
# RUTAS
# ══════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', resultado=None)


@app.route('/predecir', methods=['POST'])
def predecir():
    global artefacto_modelo

    if artefacto_modelo is None:
        artefacto_modelo = cargar_modelo()
    if artefacto_modelo is None:
        flash("⚠️ Modelo no disponible. Entrena primero el modelo.", "danger")
        return redirect(url_for('index'))

    try:
        # ── Helpers: calcular rangos automáticamente desde valores numéricos ──
        def rango_precio(v):
            if v < 20000:  return 'less than 20000'
            if v < 30000:  return '20000 to 29000'
            if v < 40000:  return '30000 to 39000'
            if v < 60000:  return '40000 to 59000'
            if v < 70000:  return '60000 to 69000'
            return 'more than 69000'

        def rango_dias(d):
            if d == 0:    return 'none'
            if d <= 7:    return '1 to 7'
            if d <= 15:   return '8 to 15'
            if d <= 30:   return '15 to 30'
            return 'more than 30'

        def rango_reclamos(n):
            if n == 0:  return 'none'
            if n == 1:  return '1'
            if n <= 4:  return '2 to 4'
            return 'more than 4'

        def semana_de_fecha(fecha_str):
            """Convierte 'YYYY-MM-DD' al número de semana del año (1-52)."""
            from datetime import datetime
            return datetime.strptime(fecha_str, '%Y-%m-%d').isocalendar()[1]

        # ── Valores que el usuario ingresa directamente ──────────────────
        valor_vehiculo       = float(request.form['valor_estimado_vehiculo'])
        dias_poliza_acc      = int(request.form['dias_poliza_accidente_min'])
        reclamos_n           = int(request.form['reclamos_pasados_min'])
        fecha_accidente_str  = request.form['fecha_accidente']
        fecha_reclamo_str    = request.form['fecha_reclamo']
        semana_acc           = semana_de_fecha(fecha_accidente_str)
        semana_rec           = semana_de_fecha(fecha_reclamo_str)

        # 1. Recoger datos del formulario — rangos se calculan en backend
        datos = {
            # Numéricos directos
            'edad':                      int(request.form['edad']),
            'deducible':                 int(request.form['deducible']),
            'calificacion_conductor':    int(request.form['calificacion_conductor']),
            'valor_estimado_vehiculo':   valor_vehiculo,
            'semanas_retraso_reporte':   int(request.form['semanas_retraso_reporte']),
            'dias_poliza_accidente_min': dias_poliza_acc,
            'reclamos_pasados_min':      reclamos_n,
            # Binarios ML-ready
            'zona_accidente_ml':         int(request.form['zona_accidente_ml']),
            'reporte_policial_ml':       int(request.form['reporte_policial_ml']),
            'testigo_presente_ml':       int(request.form['testigo_presente_ml']),
            'genero_ml':                 int(request.form['genero_ml']),
            # Categóricos principales
            'cobertura':                 request.form['cobertura'],
            'culpabilidad':              request.form['culpabilidad'],
            'categoria_vehiculo':        request.form['categoria_vehiculo'],
            # Rangos calculados automáticamente desde los valores numéricos
            'rango_precio_vehiculo':     rango_precio(valor_vehiculo),
            'dias_poliza_accidente':     rango_dias(dias_poliza_acc),
            'reclamos_pasados':          rango_reclamos(reclamos_n),
            # Feature engineering
            'cambio_direccion':          request.form['cambio_direccion'],
            'antiguedad_vehiculo':       request.form['antiguedad_vehiculo'],
            'edad_titular':              request.form['edad_titular'],
            'numero_suplementos':        request.form['numero_suplementos'],
            'tipo_agente':               request.form['tipo_agente'],
            # Semanas calculadas desde fechas
            'semana_accidente':          semana_acc,
            'semana_reclamo':            semana_rec,
        }

        # 2. Predecir usando el pipeline completo (incluye feature engineering)
        df_entrada = pd.DataFrame([datos])
        resultado_df = train_model.predecir(df_entrada, ruta_modelo='modelo_fraudes_v2.pkl')

        probabilidad = float(resultado_df['probabilidad_fraude'].iloc[0])
        prediccion   = int(resultado_df['prediccion_fraude'].iloc[0])
        nivel_alerta = str(resultado_df['nivel_alerta'].iloc[0])
        es_fraude    = prediccion == 1

        # 3. Guardar en MongoDB para feedback loop
        registro = datos.copy()
        registro['ml_prediccion']          = prediccion
        registro['ml_probabilidad_fraude'] = round(probabilidad * 100, 2)
        registro['nivel_alerta']           = nivel_alerta
        registro['fraude_detectado']       = None
        registro['fecha_evaluacion']       = datetime.now()
        coleccion.insert_one(registro)

        # 4. Preparar contexto para la vista
        resultado = {
            'es_fraude':         es_fraude,
            'nivel_alerta':      nivel_alerta,
            'probabilidad_pct':  round(probabilidad * 100, 2),
            'etiqueta':          '¡FRAUDE DETECTADO!' if es_fraude else 'Reclamo Legítimo',
            # Desglose de señales de riesgo para mostrar en UI
            'señales': {
                'Cambio de dirección reciente': datos['cambio_direccion'] in ['under 6 months', '2 to 3 years'],
                'Sin testigo ni reporte policial': datos['testigo_presente_ml'] == 0 and datos['reporte_policial_ml'] == 0,
                'Retraso en reporte': datos['semanas_retraso_reporte'] > 0,
                'Accidente temprano en póliza': datos['dias_poliza_accidente_min'] <= 15,
                'Agente externo': datos['tipo_agente'] == 'External',
            }
        }

        return render_template('index.html', resultado=resultado, form_data=datos)

    except Exception as e:
        flash(f"Error en predicción: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/admin', methods=['GET'])
def admin():
    # Estadísticas rápidas para el dashboard
    total    = coleccion.count_documents({})
    fraudes  = coleccion.count_documents({'ml_prediccion': 1})
    pendientes = coleccion.count_documents({'fraude_detectado': None})
    validados  = coleccion.count_documents({'fraude_detectado': {'$ne': None}})

    casos = list(
        coleccion.find({'fraude_detectado': None})
                 .sort('fecha_evaluacion', -1)
                 .limit(50)
    )

    # Métricas del modelo actual
    metricas_modelo = {}
    if artefacto_modelo:
        metricas_modelo = {
            'auc_roc':    round(artefacto_modelo.get('auc_roc', 0) * 100, 2),
            'f1_score':   round(artefacto_modelo.get('f1_score', 0) * 100, 2),
            'umbral':     round(artefacto_modelo.get('umbral_optimo', 0.5), 3),
            'fecha':      artefacto_modelo.get('fecha_entrenamiento', 'N/A'),
        }

    stats = {
        'total': total,
        'fraudes': fraudes,
        'pendientes': pendientes,
        'validados': validados,
        'tasa_fraude': round(fraudes / total * 100, 1) if total > 0 else 0,
    }

    return render_template('admin.html', casos=casos, stats=stats, metricas=metricas_modelo)


@app.route('/feedback/<id_caso>/<int:es_fraude_real>')
def feedback(id_caso, es_fraude_real):
    coleccion.update_one(
        {'_id': ObjectId(id_caso)},
        {'$set': {'fraude_detectado': es_fraude_real, 'fecha_validacion': datetime.now()}}
    )
    flash(f"✅ Caso validado como {'FRAUDE' if es_fraude_real else 'LEGÍTIMO'}. Datos guardados para reentrenamiento.", "success")
    return redirect(url_for('admin'))


@app.route('/reentrenar', methods=['POST'])
def reentrenar():
    global artefacto_modelo
    try:
        # Reentrenar con CSV base + feedback de MongoDB
        artefacto = train_model.entrenar(
            csv_fallback='fraudes_enriquecido_202603051611.csv',
            guardar_modelo=True
        )
        artefacto_modelo = cargar_modelo()
        auc  = round(artefacto['auc_roc'] * 100, 2)
        f1   = round(artefacto['f1_score'] * 100, 2)
        umbral = round(artefacto['umbral_optimo'], 3)
        flash(f"🎉 Modelo v2 reentrenado. AUC-ROC: {auc}% | F1: {f1}% | Umbral: {umbral}", "success")
    except Exception as e:
        flash(f"❌ Error al reentrenar: {str(e)}", "danger")
    return redirect(url_for('admin'))


# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)