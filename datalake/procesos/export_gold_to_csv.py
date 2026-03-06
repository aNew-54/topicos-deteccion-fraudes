#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession

def exportar_tabla(spark, db_name, table_name, output_path):
    print(f"📥 Exportando {db_name}.{table_name} a {output_path}...")
    df = spark.table(f"{db_name}.{table_name}")
    
    # coalesce(1) junta todos los nodos en un solo archivo
    df.coalesce(1).write \
      .mode("overwrite") \
      .option("header", "true") \
      .csv(output_path)
    print(f"✅ Exportación de {table_name} completada.")

def main():
    spark = SparkSession.builder \
        .appName("Export-Fraudes-To-CSV") \
        .enableHiveSupport() \
        .getOrCreate()

    db_functional = "dev_fraudes_functional"

    # 1. Exportar la tabla Principal
    exportar_tabla(
        spark=spark,
        db_name=db_functional,
        table_name="fraudes_enriquecido",
        output_path="file:/home/hadoop/topicos-deteccion-fraudes/reports/temp/fraudes_enriquecido"
    )

    # 2. Exportar la tabla de KPIs
    exportar_tabla(
        spark=spark,
        db_name=db_functional,
        table_name="metricas_fraude",
        output_path="file:/home/hadoop/topicos-deteccion-fraudes/reports/temp/metricas_fraude"
    )

    spark.stop()

if __name__ == "__main__":
    main()