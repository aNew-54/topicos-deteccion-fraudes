from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.task.trigger_rule import TriggerRule
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL DE RUTAS
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = "/home/hadoop/topicos-deteccion-fraudes"
DATALAKE_DIR    = f"{BASE_DIR}/datalake"
PROCESOS_DIR    = f"{DATALAKE_DIR}/procesos"
SCHEMA_DIR      = f"{DATALAKE_DIR}/schema"
DATASET_DIR     = f"{DATALAKE_DIR}/dataset"
REPORTS_TEMP    = f"{BASE_DIR}/reports/temp"
REPORTS_DIR     = f"{BASE_DIR}/reports"
WEBAPP_DIR      = f"{BASE_DIR}/webapp"
WAREHOUSE_DIR   = "/user/hadoop/topicos-deteccion-fraudes/warehouse"
HDFS_BASE       = "/user/hadoop/topicos-deteccion-fraudes"
HDFS_SCHEMA     = f"{HDFS_BASE}/schema/DEV_LANDING"

VENV_ML       = f"{WEBAPP_DIR}/venv_ml"
VENV_PYTHON   = f"{VENV_ML}/bin/python3"
VENV_ACTIVATE = f"source {VENV_ML}/bin/activate"

FRAUDES_DATA    = f"{DATASET_DIR}/fraudes.data"
CSV_ENRIQUECIDO = f"{REPORTS_DIR}/fraudes_enriquecido.csv"
MODELO_PKL      = f"{WEBAPP_DIR}/modelo_fraudes_v2.pkl"
WEBAPP_PORT     = 5000
WEBAPP_PID      = f"{BASE_DIR}/webapp.pid"

# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTOS POR DEFECTO
# ══════════════════════════════════════════════════════════════════════════════

default_args = {
    'owner':            'hadoop',
    'depends_on_past':  False,
    'start_date':       datetime(2026, 3, 1),
    'email_on_failure': False,
    'retries':          1,
    'retry_delay':      timedelta(minutes=3),
}

# ══════════════════════════════════════════════════════════════════════════════
# DAG
# ══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="pipeline_fraudes_completo",
    default_args=default_args,
    description="ETL Medallon + ML + Webapp — ejecucion completa con un clic",
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=['fraudes', 'medallon', 'pyspark', 'ml', 'flask'],
) as dag:

    # ══════════════════════════════════════════════════════════════
    # BLOQUE 0 — LEVANTAR SERVICIOS
    # ══════════════════════════════════════════════════════════════

    start_hdfs = BashOperator(
        task_id="start_hdfs",
        bash_command="""
            if hdfs dfs -ls / > /dev/null 2>&1; then
                echo "✅ HDFS ya esta corriendo."
            else
                echo "🚀 Iniciando HDFS..."
                start-dfs.sh && sleep 15
                hdfs dfs -ls / || { echo "❌ HDFS no pudo iniciarse."; exit 1; }
                echo "✅ HDFS iniciado."
            fi
        """,
        execution_timeout=timedelta(minutes=4),
    )

    start_yarn = BashOperator(
        task_id="start_yarn",
        bash_command="""
            if yarn node -list 2>/dev/null | grep -q "RUNNING"; then
                echo "✅ YARN ya esta corriendo."
            else
                echo "🚀 Iniciando YARN..."
                start-yarn.sh && sleep 15
                yarn node -list | grep -q "RUNNING" || { echo "❌ YARN no pudo iniciarse."; exit 1; }
                echo "✅ YARN iniciado."
            fi
        """,
        execution_timeout=timedelta(minutes=4),
    )

    start_hive_metastore = BashOperator(
        task_id="start_hive_metastore",
        bash_command="""
            if pgrep -f "metastore" > /dev/null; then
                echo "✅ Hive Metastore ya esta corriendo."
            else
                echo "🚀 Iniciando Hive Metastore..."
                nohup hive --service metastore > /tmp/hive_metastore.log 2>&1 &
                echo $! > /tmp/hive_metastore.pid
                sleep 20
                pgrep -f "metastore" || { echo "❌ Metastore fallo. Ver /tmp/hive_metastore.log"; exit 1; }
                echo "✅ Metastore iniciado."
            fi
        """,
        execution_timeout=timedelta(minutes=4),
    )

    start_hiveserver2 = BashOperator(
        task_id="start_hiveserver2",
        bash_command="""
            if pgrep -f "hiveserver2" > /dev/null; then
                echo "✅ HiveServer2 ya esta corriendo."
            else
                echo "🚀 Iniciando HiveServer2..."
                nohup hive --service hiveserver2 > /tmp/hiveserver2.log 2>&1 &
                echo $! > /tmp/hiveserver2.pid
                sleep 30
                nc -z localhost 10000 || { sleep 20; nc -z localhost 10000 || { echo "❌ Puerto 10000 no responde."; exit 1; }; }
                echo "✅ HiveServer2 listo en puerto 10000."
            fi
        """,
        execution_timeout=timedelta(minutes=6),
    )

    # ══════════════════════════════════════════════════════════════
    # BLOQUE 1 — PREPARAR HDFS
    # ══════════════════════════════════════════════════════════════

    prepare_hdfs = BashOperator(
        task_id="prepare_hdfs_dirs",
        bash_command=f"""
            echo "📂 Preparando estructura HDFS..."
            hdfs dfs -mkdir -p {HDFS_BASE}/dataset
            hdfs dfs -mkdir -p {HDFS_BASE}/schema/DEV_LANDING
            hdfs dfs -mkdir -p {WAREHOUSE_DIR}

            if hdfs dfs -test -e {HDFS_BASE}/dataset/fraudes.data; then
                echo "✅ fraudes.data ya existe en HDFS."
            else
                hdfs dfs -put {FRAUDES_DATA} {HDFS_BASE}/dataset/
                echo "✅ fraudes.data subido."
            fi

            hdfs dfs -put -f {SCHEMA_DIR}/fraudes.avsc {HDFS_SCHEMA}/
            echo "✅ Schema AVRO subido."
            hdfs dfs -ls {HDFS_BASE}/dataset
        """,
        execution_timeout=timedelta(minutes=5),
    )

    # ══════════════════════════════════════════════════════════════
    # BLOQUE 2 — ETL MEDALLÓN
    # ══════════════════════════════════════════════════════════════

    etl_workload = BashOperator(
        task_id="etl_workload",
        bash_command=f"""
            echo "🔄 [1/4] WORKLOAD"
            spark-submit \
              --master yarn --deploy-mode client \
              --conf spark.sql.warehouse.dir={WAREHOUSE_DIR} \
              --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
              {PROCESOS_DIR}/poblar_capa_workload.py \
              --env DEV --username hadoop \
              --base_path {HDFS_BASE} \
              --input_hdfs_path {HDFS_BASE}/dataset
            echo "✅ Workload OK."
        """,
        execution_timeout=timedelta(minutes=20),
    )

    etl_landing = BashOperator(
        task_id="etl_landing",
        bash_command=f"""
            echo "🔄 [2/4] LANDING"
            spark-submit \
              --master yarn --deploy-mode client \
              --conf spark.sql.warehouse.dir={WAREHOUSE_DIR} \
              --conf spark.sql.avro.compression.codec=snappy \
              --packages org.apache.spark:spark-avro_2.12:3.5.0 \
              {PROCESOS_DIR}/poblar_capa_landing.py \
              --env DEV --username hadoop \
              --base_path {HDFS_BASE} \
              --schema_path {HDFS_BASE}/schema \
              --source_db dev_fraudes_workload
            echo "✅ Landing OK."
        """,
        execution_timeout=timedelta(minutes=20),
    )

    etl_curated = BashOperator(
        task_id="etl_curated",
        bash_command=f"""
            echo "🔄 [3/4] CURATED"
            spark-submit \
              --master yarn --deploy-mode client \
              --conf spark.sql.warehouse.dir={WAREHOUSE_DIR} \
              --conf spark.sql.parquet.compression.codec=snappy \
              {PROCESOS_DIR}/poblar_capa_curated.py \
              --env DEV --username hadoop \
              --base_path {HDFS_BASE} \
              --source_db dev_fraudes_landing
            echo "✅ Curated OK."
        """,
        execution_timeout=timedelta(minutes=20),
    )

    etl_functional = BashOperator(
        task_id="etl_functional",
        bash_command=f"""
            echo "🔄 [4/4] FUNCTIONAL"
            spark-submit \
              --master yarn --deploy-mode client \
              --conf spark.sql.warehouse.dir={WAREHOUSE_DIR} \
              --conf spark.sql.parquet.compression.codec=snappy \
              {PROCESOS_DIR}/poblar_capa_functional.py \
              --env DEV --username hadoop \
              --base_path {HDFS_BASE} \
              --source_db curated
            echo "✅ Functional OK."
        """,
        execution_timeout=timedelta(minutes=20),
    )

    # ══════════════════════════════════════════════════════════════
    # BLOQUE 3 — EXPORTAR DATOS
    # ══════════════════════════════════════════════════════════════

    export_csv = BashOperator(
        task_id="export_gold_to_csv",
        bash_command=f"""
            echo "📊 Exportando Gold → CSV..."
            spark-submit \
              --master yarn --deploy-mode client \
              --conf spark.sql.warehouse.dir={WAREHOUSE_DIR} \
              {PROCESOS_DIR}/export_gold_to_csv.py
            echo "✅ Export CSV OK."
        """,
        execution_timeout=timedelta(minutes=10),
    )

    clean_csv = BashOperator(
        task_id="clean_and_rename_csv",
        bash_command=f"""
            echo "🗂️ Consolidando archivos part-*.csv..."
            mkdir -p {REPORTS_DIR}

            PARTS=( {REPORTS_TEMP}/fraudes_enriquecido/part-*.csv )
            if [ ! -f "${{PARTS[0]}}" ]; then
                echo "❌ No se encontraron part-*.csv en fraudes_enriquecido"; exit 1
            fi
            head -1 "${{PARTS[0]}}" > {CSV_ENRIQUECIDO}
            for f in "${{PARTS[@]}}"; do tail -n +2 "$f"; done >> {CSV_ENRIQUECIDO}
            echo "✅ fraudes_enriquecido.csv — $(wc -l < {CSV_ENRIQUECIDO}) lineas"

            PARTS_KPI=( {REPORTS_TEMP}/metricas_fraude/part-*.csv )
            if [ -f "${{PARTS_KPI[0]}}" ]; then
                head -1 "${{PARTS_KPI[0]}}" > {REPORTS_DIR}/metricas_fraude.csv
                for f in "${{PARTS_KPI[@]}}"; do tail -n +2 "$f"; done >> {REPORTS_DIR}/metricas_fraude.csv
                echo "✅ metricas_fraude.csv listo."
            fi
        """,
        execution_timeout=timedelta(minutes=5),
    )

    export_mongodb = BashOperator(
        task_id="export_gold_to_mongodb",
        bash_command=f"""
            echo "🍃 Exportando a MongoDB..."
            spark-submit \
              --master yarn --deploy-mode client \
              --packages org.mongodb.spark:mongo-spark-connector_2.12:10.4.0 \
              {PROCESOS_DIR}/export_gold_to_mongo.py
            echo "✅ Export MongoDB OK."
        """,
        execution_timeout=timedelta(minutes=10),
    )

    # ══════════════════════════════════════════════════════════════
    # BLOQUE 4 — ENTRENAR MODELO ML
    # ══════════════════════════════════════════════════════════════

    train_ml = BashOperator(
        task_id="entrenar_modelo_ml",
        bash_command=f"""
            echo "🧠 Activando entorno venv_ml..."
            if [ ! -f "{VENV_PYTHON}" ]; then
                echo "❌ No se encontro {VENV_PYTHON}"; exit 1
            fi

            {VENV_ACTIVATE} && pip install -q \
                scikit-learn pandas joblib pymongo flask numpy 2>&1 | tail -5

            echo "✅ venv_ml activo."
            echo "🚀 Iniciando entrenamiento..."
            cd {WEBAPP_DIR}

            {VENV_PYTHON} -c "
import sys
sys.path.insert(0, '{WEBAPP_DIR}')
import modelo_fraudes_v2 as train

artefacto = train.entrenar(
    csv_fallback='{CSV_ENRIQUECIDO}',
    guardar_modelo=True
)
if artefacto is None:
    print('ERROR: El entrenamiento retorno None.')
    sys.exit(1)

auc = round(artefacto['auc_roc'] * 100, 2)
f1  = round(artefacto['f1_score'] * 100, 2)
print(f'✅ Modelo guardado | AUC-ROC: {{auc}}% | F1: {{f1}}%')
"
            echo "✅ Entrenamiento completado."
        """,
        execution_timeout=timedelta(minutes=25),
    )

    # ══════════════════════════════════════════════════════════════
    # BLOQUE 5 — LANZAR WEBAPP FLASK
    # ══════════════════════════════════════════════════════════════

    launch_webapp = BashOperator(
        task_id="launch_webapp_flask",
        bash_command=f"""
            echo "🌐 Lanzando SeguroIA v2.0..."
            if [ ! -f "{VENV_PYTHON}" ]; then
                echo "❌ venv_ml no encontrado."; exit 1
            fi

            if [ -f {WEBAPP_PID} ]; then
                OLD_PID=$(cat {WEBAPP_PID})
                if kill -0 "$OLD_PID" 2>/dev/null; then
                    kill "$OLD_PID" && sleep 3
                fi
                rm -f {WEBAPP_PID}
            fi

            if nc -z localhost {WEBAPP_PORT} 2>/dev/null; then
                fuser -k {WEBAPP_PORT}/tcp 2>/dev/null || true
                sleep 2
            fi

            if [ ! -f "{MODELO_PKL}" ]; then
                echo "❌ Modelo no encontrado: {MODELO_PKL}"; exit 1
            fi

            cd {WEBAPP_DIR}
            nohup {VENV_PYTHON} app.py > /tmp/webapp_flask.log 2>&1 &
            NEW_PID=$!
            echo $NEW_PID > {WEBAPP_PID}
            sleep 6

            if kill -0 "$NEW_PID" 2>/dev/null; then
                echo "✅ Webapp corriendo (PID: $NEW_PID)"
                echo "🌐 URL: http://localhost:{WEBAPP_PORT}"
            else
                echo "❌ Webapp fallo. Logs:"; cat /tmp/webapp_flask.log; exit 1
            fi
        """,
        execution_timeout=timedelta(minutes=3),
    )

    # ══════════════════════════════════════════════════════════════
    # RESUMEN FINAL
    # ══════════════════════════════════════════════════════════════

    pipeline_summary = BashOperator(
        task_id="pipeline_summary",
        bash_command=f"""
            echo "╔══════════════════════════════════════════════════╗"
            echo "║     ✅  PIPELINE COMPLETADO EXITOSAMENTE        ║"
            echo "╠══════════════════════════════════════════════════╣"
            echo "║  ETL    : Workload→Landing→Curated→Functional   ║"
            echo "║  Datos  : CSV + MongoDB actualizados             ║"
            echo "║  Modelo : modelo_fraudes_v2.pkl reentrenado      ║"
            echo "║  Webapp : http://localhost:{WEBAPP_PORT}                 ║"
            echo "╚══════════════════════════════════════════════════╝"
            echo "📄 CSV: $(wc -l < {CSV_ENRIQUECIDO} 2>/dev/null || echo 'n/a') lineas"
            echo "🤖 Modelo: $(ls -lh {MODELO_PKL} 2>/dev/null | awk '{{print $5}}' || echo 'no encontrado')"
        """,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ══════════════════════════════════════════════════════════════
    # DEPENDENCIAS
    # ══════════════════════════════════════════════════════════════

    [start_hdfs, start_yarn] >> start_hive_metastore >> start_hiveserver2 >> prepare_hdfs
    prepare_hdfs >> etl_workload >> etl_landing >> etl_curated >> etl_functional
    etl_functional >> export_csv >> clean_csv
    clean_csv >> [export_mongodb, train_ml]
    [export_mongodb, train_ml] >> launch_webapp >> pipeline_summary
