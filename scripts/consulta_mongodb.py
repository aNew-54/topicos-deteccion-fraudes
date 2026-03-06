from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Variables globales para no repetir código
BASE_DIR = "/home/hadoop/topicos-deteccion-fraudes"
PROCESOS_DIR = f"{BASE_DIR}/datalake/procesos"
WAREHOUSE_DIR = f"/user/hadoop/topicos-deteccion-fraudes/warehouse"
HDFS_BASE = "/user/hadoop/topicos-deteccion-fraudes"

# Argumentos por defecto para el DAG
default_args = {
    'owner': 'hadoop',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 1), 
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id="etl_fraudes_medallon_completo",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    tags=['fraudes', 'medallon', 'pyspark']
) as dag:

    # 1. CAPA WORKLOAD (Lee CSV de HDFS y crea Tablas de texto)
    workload = BashOperator(
        task_id="capa_workload",
        bash_command=f"""
        spark-submit \
          --master yarn \
          --deploy-mode client \
          --conf spark.sql.warehouse.dir={WAREHOUSE_DIR} \
          --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
          {PROCESOS_DIR}/workload.py \
          --env DEV \
          --username hadoop \
          --base_path {HDFS_BASE} \
          --input_hdfs_path {HDFS_BASE}/dataset
        """
    )

    # 2. CAPA LANDING (Transforma a AVRO y particiona)
    landing = BashOperator(
        task_id="capa_landing",
        bash_command=f"""
        spark-submit \
          --master yarn \
          --deploy-mode client \
          --conf spark.sql.warehouse.dir={WAREHOUSE_DIR} \
          --conf spark.sql.avro.compression.codec=snappy \
          --packages org.apache.spark:spark-avro_2.12:3.5.0 \
          {PROCESOS_DIR}/landing.py \
          --env DEV \
          --username hadoop \
          --base_path {HDFS_BASE} \
          --schema_path {HDFS_BASE}/schema \
          --source_db dev_fraudes_workload
        """
    )

    # 3. CAPA CURATED (Limpia datos, Data Quality y pasa a Parquet)
    curated = BashOperator(
        task_id="capa_curated",
        bash_command=f"""
        spark-submit \
          --master yarn \
          --deploy-mode client \
          --conf spark.sql.warehouse.dir={WAREHOUSE_DIR} \
          --conf spark.sql.parquet.compression.codec=snappy \
          {PROCESOS_DIR}/curated.py \
          --env DEV \
          --username hadoop \
          --base_path {HDFS_BASE} \
          --source_db dev_fraudes_landing
        """
    )

    # 4. CAPA FUNCTIONAL (Feature Engineering para ML y KPIs)
    functional = BashOperator(
        task_id="capa_functional",
        bash_command=f"""
        spark-submit \
          --master yarn \
          --deploy-mode client \
          --conf spark.sql.warehouse.dir={WAREHOUSE_DIR} \
          --conf spark.sql.parquet.compression.codec=snappy \
          {PROCESOS_DIR}/functional.py \
          --env DEV \
          --username hadoop \
          --base_path {HDFS_BASE} \
          --source_db curated
        """
    )

    # 5. EXPORTAR A CSV (Saca los datos de Hive a Linux local)
    export_csv = BashOperator(
        task_id="export_gold_csv",
        bash_command=f"""
        spark-submit \
          --master yarn \
          --deploy-mode client \
          --conf spark.sql.warehouse.dir={WAREHOUSE_DIR} \
          {PROCESOS_DIR}/export_gold_to_csv.py
        """
    )

    # 5.5 LIMPIAR CSV (El paso extra necesario para renombrar los archivos part-*.csv)
    clean_csv = BashOperator(
        task_id="clean_rename_csv",
        bash_command=f"""
        cp {BASE_DIR}/datalake/temp_enriquecido/part-*.csv {BASE_DIR}/datalake/fraudes_enriquecido.csv && \
        cp {BASE_DIR}/datalake/temp_metricas/part-*.csv {BASE_DIR}/datalake/metricas_fraude.csv
        """
    )

    # 6. EXPORTAR A MONGODB (Lee los CSV limpios y los sube a la BD NoSQL)
    export_mongo = BashOperator(
        task_id="export_gold_mongo",
        bash_command=f"""
        spark-submit \
          --master yarn \
          --deploy-mode client \
          --packages org.mongodb.spark:mongo-spark-connector_2.12:10.4.0 \
          {PROCESOS_DIR}/export_gold_to_mongo.py
        """
    )

    # =========================================================================
    # DEFINICIÓN DEL ORDEN DE EJECUCIÓN (DEPENDENCIAS)
    # =========================================================================
    workload >> landing >> curated >> functional >> export_csv >> clean_csv >> export_mongo