#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script PySpark para despliegue de capa Workload - Detección de Fraudes
"""

import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# =============================================================================
# @section 1. Configuración de parámetros
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Proceso de carga - Capa Workload Fraudes')
    parser.add_argument('--env', type=str, default='DEV', help='Entorno: DEV, QA, PROD')
    parser.add_argument('--username', type=str, default='hadoop', help='Usuario HDFS')
    parser.add_argument('--base_path', type=str, default='/user/hadoop/topicos-deteccion-fraudes', help='Ruta base en HDFS')
    parser.add_argument('--input_hdfs_path', type=str, default='/user/hadoop/topicos-deteccion-fraudes/dataset', help='Ruta HDFS de los datos origen')
    return parser.parse_args()

# =============================================================================
# @section 2. Inicialización de SparkSession
# =============================================================================

def create_spark_session(app_name="Workload_Fraudes"):
    return SparkSession.builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false") \
        .config("spark.sql.legacy.charVarcharCodegen", "true") \
        .getOrCreate()

# =============================================================================
# @section 3. Funciones auxiliares
# =============================================================================

def crear_database(spark, env, base_path):
    """Crea la base de datos si no existe"""
    db_name = f"{env.lower()}_fraudes_workload"
    db_location = f"{base_path}/warehouse/{db_name}"
    
    spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name} LOCATION '{db_location}'")
    print(f"✅ Database '{db_name}' creada en: {db_location}")
    return db_name

def crear_tabla_external(spark, db_name, table_name, df, location, spark_schema):
    """Crea tabla externa en Hive con formato TEXTFILE"""
    df.createOrReplaceTempView(f"tmp_{table_name}")
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {db_name}.{table_name} (
        {', '.join([f'{field.name} STRING' for field in spark_schema.fields])}
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY '|'
    LINES TERMINATED BY '\\n'
    STORED AS TEXTFILE
    LOCATION '{location}'
    TBLPROPERTIES(
        'skip.header.line.count'='1',
        'store.charset'='ISO-8859-1',
        'retrieve.charset'='ISO-8859-1'
    )
    """
    spark.sql(create_table_sql)
    
    spark.sql(f"""
        INSERT OVERWRITE TABLE {db_name}.{table_name}
        SELECT * FROM tmp_{table_name}
    """)
    print(f"✅ Tabla '{db_name}.{table_name}' desplegada en: {location}")

# =============================================================================
# @section 4. Esquema de Datos
# =============================================================================

# Definimos todas las columnas como String para la capa Workload (Landing)
SCHEMA_FRAUDES = StructType([
    StructField("Month", StringType(), True),
    StructField("WeekOfMonth", StringType(), True),
    StructField("DayOfWeek", StringType(), True),
    StructField("Make", StringType(), True),
    StructField("AccidentArea", StringType(), True),
    StructField("DayOfWeekClaimed", StringType(), True),
    StructField("MonthClaimed", StringType(), True),
    StructField("WeekOfMonthClaimed", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("MaritalStatus", StringType(), True),
    StructField("Age", StringType(), True),
    StructField("Fault", StringType(), True),
    StructField("PolicyType", StringType(), True),
    StructField("VehicleCategory", StringType(), True),
    StructField("VehiclePrice", StringType(), True),
    StructField("FraudFound_P", StringType(), True),
    StructField("PolicyNumber", StringType(), True),
    StructField("RepNumber", StringType(), True),
    StructField("Deductible", StringType(), True),
    StructField("DriverRating", StringType(), True),
    StructField("Days_Policy_Accident", StringType(), True),
    StructField("Days_Policy_Claim", StringType(), True),
    StructField("PastNumberOfClaims", StringType(), True),
    StructField("AgeOfVehicle", StringType(), True),
    StructField("AgeOfPolicyHolder", StringType(), True),
    StructField("PoliceReportFiled", StringType(), True),
    StructField("WitnessPresent", StringType(), True),
    StructField("AgentType", StringType(), True),
    StructField("NumberOfSuppliments", StringType(), True),
    StructField("AddressChange_Claim", StringType(), True),
    StructField("NumberOfCars", StringType(), True),
    StructField("Year", StringType(), True),
    StructField("BasePolicy", StringType(), True)
])

# =============================================================================
# @section 5. Proceso principal
# =============================================================================

def main():
    args = parse_arguments()
    spark = create_spark_session()
    
    try:
        # 1. Crear base de datos
        db_name = crear_database(spark, args.env, args.base_path)
        
        table_name = "FRAUDES"
        archivo_datos = "fraudes.data"
        
        ruta_origen_hdfs = f"{args.input_hdfs_path}/{archivo_datos}"
        ruta_destino_hdfs = f"{args.base_path}/warehouse/{db_name}/{table_name.lower()}"
        
        print(f"📥 Procesando tabla: {table_name} desde {ruta_origen_hdfs}")
        
        # 2. Leer datos desde HDFS
        df = spark.read.csv(
            ruta_origen_hdfs,
            schema=SCHEMA_FRAUDES,
            sep='|',
            header=True,
            encoding='iso-8859-1',
            nullValue='\\N',
            emptyValue=''
        )
        
        # 3. Crear tabla y cargar
        crear_tabla_external(spark, db_name, table_name, df, ruta_destino_hdfs, SCHEMA_FRAUDES)
        
        # 4. Validar
        print("\n🔍 Muestra de datos:")
        spark.sql(f"SELECT * FROM {db_name}.{table_name} LIMIT 5").show(truncate=False)
        print("\n🎉 Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en el proceso: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main()