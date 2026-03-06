#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
import logging
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cargar_a_mongo(spark, csv_path, uri, database, collection):
    try:
        logger.info(f"📥 Leyendo archivo local: {csv_path}")
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .option("encoding", "UTF-8") \
            .csv(csv_path)

        record_count = df.count()
        if record_count == 0:
            logger.warning(f"⚠️ El archivo {csv_path} está vacío.")
            return

        logger.info(f"📊 Registros a subir: {record_count}")
        
        logger.info(f"💾 Escribiendo en MongoDB: {database}.{collection} ...")
        df.write \
            .format("mongodb") \
            .mode("overwrite") \
            .option("spark.mongodb.connection.uri", uri) \
            .option("spark.mongodb.database", database) \
            .option("spark.mongodb.collection", collection) \
            .option("spark.mongodb.write.mode", "overwrite") \
            .save()
            
        logger.info(f"✅ Datos insertados exitosamente en {collection}\n")
    
    except Exception as e:
        logger.error(f"❌ Error subiendo {collection}: {str(e)}")
        raise

def main():
    # Tu conexión a MongoDB
    mongo_uri = "mongodb://192.168.18.30:27017/" # Cambiar por la IPv4 que sale al usar ipconfig en windows (donde tienen instalado MongoDB)
    database_name = "db_fraudes" # Nombre de tu base de datos en Mongo

    # Iniciar Spark con la configuración del conector de Mongo
    spark = SparkSession.builder \
        .appName("CSV_to_MongoDB_Fraudes") \
        .config("spark.mongodb.connection.uri", mongo_uri) \
        .getOrCreate()

    logger.info("✅ Sesión Spark conectada a MongoDB")

    try:
        # 1. Subir tabla principal
        ruta_enriquecido = "file:/home/hadoop/topicos-deteccion-fraudes/reports/temp/fraudes_enriquecido.csv"
        cargar_a_mongo(spark, ruta_enriquecido, mongo_uri, database_name, "fraudes_detalle")

        # 2. Subir tabla de KPIs
        ruta_metricas = "file:/home/hadoop/topicos-deteccion-fraudes/reports/temp/metricas_fraude.csv"
        cargar_a_mongo(spark, ruta_metricas, mongo_uri, database_name, "metricas_kpi")

    except Exception as e:
        logger.error("❌ El proceso de migración falló.")
        sys.exit(1)
    finally:
        spark.stop()
        logger.info("🛑 Sesión Spark cerrada")

if __name__ == "__main__":
    main()