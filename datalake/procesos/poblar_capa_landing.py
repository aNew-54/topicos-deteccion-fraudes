#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='DEV')
    parser.add_argument('--username', type=str, default='hadoop')
    parser.add_argument('--base_path', type=str, default='/user/hadoop/topicos-deteccion-fraudes')
    parser.add_argument('--schema_path', type=str, default='/user/hadoop/topicos-deteccion-fraudes/schema')
    parser.add_argument('--source_db', type=str, default='dev_fraudes_workload')
    return parser.parse_args()

def create_spark_session():
    return SparkSession.builder \
        .appName("Landing_Fraudes") \
        .enableHiveSupport() \
        .config("spark.sql.avro.compression.codec", "snappy") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition", "true") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .getOrCreate()

def procesar_fraudes(spark, args):
    db_landing = f"{args.env.lower()}_fraudes_landing"
    db_source = args.source_db
    table_name = "FRAUDES"
    partition_col = "monthclaimed"
    
    # 1. Crear BD
    db_location = f"{args.base_path}/warehouse/{db_landing}"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_landing} LOCATION '{db_location}'")
    
    # 2. Crear Tabla Externa AVRO particionada
    location = f"{db_location}/{table_name.lower()}"
    schema_url = f"hdfs://{args.schema_path}/{args.env.upper()}_LANDING/fraudes.avsc"
    
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {db_landing}.{table_name}
    PARTITIONED BY ({partition_col} STRING)
    STORED AS AVRO
    LOCATION '{location}'
    TBLPROPERTIES ('avro.schema.url'='{schema_url}', 'avro.output.codec'='snappy')
    """
    spark.sql(create_sql)
    
    # 3. Leer de Workload y preparar columnas (pasando todo a minúsculas)
    df_source = spark.table(f"{db_source}.{table_name}")
    df_source = df_source.toDF(*[c.lower() for c in df_source.columns])
    
    # Mover columna de partición al final (Requisito de Hive)
    cols = [c for c in df_source.columns if c != partition_col]
    df_insert = df_source.select(*cols, col(partition_col))
    df_insert.createOrReplaceTempView("src_data")
    
    # 4. Insertar con particionamiento dinámico
    print(f"📥 Insertando datos particionados por '{partition_col}'...")
    spark.sql(f"""
        INSERT OVERWRITE TABLE {db_landing}.{table_name}
        PARTITION ({partition_col})
        SELECT * FROM src_data
    """)
    
    print("✅ Proceso completado. Particiones generadas:")
    spark.sql(f"SHOW PARTITIONS {db_landing}.{table_name}").show(truncate=False)

if __name__ == "__main__":
    args = parse_arguments()
    spark = create_spark_session()
    procesar_fraudes(spark, args)
    spark.stop()