#!/usr/bin/env python3
# -- coding: utf-8 --

import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, trim, regexp_replace, split

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='DEV')
    parser.add_argument('--username', type=str, default='hadoop')
    parser.add_argument('--base_path', type=str, default='/user/hadoop/topicos-deteccion-fraudes')
    parser.add_argument('--source_db', type=str, default='dev_fraudes_landing')
    return parser.parse_args()

def create_spark_session():
    return SparkSession.builder \
        .appName("Curated_Fraudes_Estandarizado_Espanol") \
        .enableHiveSupport() \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition", "true") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .getOrCreate()

def limpiar_traducir_y_ordenar(df):
    """Limpia espacios, traduce valores internos y reorganiza todo en español"""
    
    # 1. LIMPIEZA BÁSICA (Quitar espacios y saltos de línea ocultos)
    for columna, tipo in df.dtypes:
        if tipo == 'string':
            df = df.withColumn(columna, trim(regexp_replace(col(columna), r"[\r\n]", "")))

    # 2. FILTROS DE CALIDAD (Data Quality)
    df = df.filter(
        (col("policynumber").isNotNull()) & 
        (col("year") > 1900) &               
        (col("month") != "0") &              # Eliminamos accidentes sin mes
        (col("monthclaimed") != "0")         # Eliminamos reclamos sin mes
    )

    # 3. CASTEO DE NÚMEROS
    columnas_enteras = [
        "weekofmonth", "weekofmonthclaimed", "age", 
        "fraudfound_p", "repnumber", "deductible", 
        "driverrating", "year"
    ]
    for c in columnas_enteras:
        df = df.withColumn(c, col(c).cast(IntegerType()))
        
    df = df.filter(col("age") >= 0) # Validar edad lógica

    # 4. SEPARAR COLUMNAS COMPUESTAS
    df = df.withColumn("tipo_vehiculo_poliza", split(col("policytype"), " - ").getItem(0)) \
           .withColumn("cobertura", split(col("policytype"), " - ").getItem(1))
    df = df.drop("policytype")


    # 5. TRADUCCIÓN DE VALORES INTERNOS (CELDA POR CELDA)
    meses_map = {"Jan":"Enero", "Feb":"Febrero", "Mar":"Marzo", "Apr":"Abril", "May":"Mayo", "Jun":"Junio", 
                 "Jul":"Julio", "Aug":"Agosto", "Sep":"Septiembre", "Oct":"Octubre", "Nov":"Noviembre", "Dec":"Diciembre"}
    dias_map = {"Monday":"Lunes", "Tuesday":"Martes", "Wednesday":"Miércoles", "Thursday":"Jueves", 
                "Friday":"Viernes", "Saturday":"Sábado", "Sunday":"Domingo"}
    
    # Aplicar traducciones a los datos
    df = df.replace(meses_map, subset=["month", "monthclaimed"])
    df = df.replace(dias_map, subset=["dayofweek", "dayofweekclaimed"])
    df = df.replace({"Urban":"Urbano", "Rural":"Rural"}, subset=["accidentarea"])
    df = df.replace({"Male":"Masculino", "Female":"Femenino"}, subset=["sex"])
    df = df.replace({"Single":"Soltero", "Married":"Casado", "Widow":"Viudo", "Divorced":"Divorciado"}, subset=["maritalstatus"])
    df = df.replace({"Policy Holder":"Titular de Póliza", "Third Party":"Tercero"}, subset=["fault"])
    df = df.replace({"Yes":"Sí", "No":"No"}, subset=["policereportfiled", "witnesspresent"])
    
    # Traducción para la cobertura separada
    df = df.replace({"Liability":"Responsabilidad Civil", "Collision":"Colisión", "All Perils":"Todos los Riesgos"}, subset=["cobertura", "basepolicy"])


    # 6. TRADUCCIÓN DE NOMBRES DE COLUMNAS
    diccionario_traduccion = {
        "policynumber": "numero_poliza",
        "repnumber": "numero_reporte",
        "year": "anio",
        "month": "mes_accidente",
        "weekofmonth": "semana_accidente",
        "dayofweek": "dia_accidente",
        "monthclaimed": "mes_reclamo",
        "weekofmonthclaimed": "semana_reclamo",
        "dayofweekclaimed": "dia_reclamo",
        "age": "edad",
        "sex": "genero",
        "maritalstatus": "estado_civil",
        "driverrating": "calificacion_conductor",
        "make": "marca_auto",
        "vehiclecategory": "categoria_vehiculo",
        "vehicleprice": "rango_precio_vehiculo",
        "ageofvehicle": "antiguedad_vehiculo",
        "numberofcars": "numero_autos",
        "accidentarea": "zona_accidente",
        "fault": "culpabilidad",
        "policereportfiled": "reporte_policial",
        "witnesspresent": "testigo_presente",
        "numberofsuppliments": "numero_suplementos",
        "deductible": "deducible",
        "days_policy_accident": "dias_poliza_accidente",
        "days_policy_claim": "dias_poliza_reclamo",
        "pastnumberofclaims": "reclamos_pasados",
        "addresschange_claim": "cambio_direccion",
        "ageofpolicyholder": "edad_titular",
        "agenttype": "tipo_agente",
        "basepolicy": "poliza_base",
        "fraudfound_p": "fraude_detectado"
    }

    for ingles, espanol in diccionario_traduccion.items():
        df = df.withColumnRenamed(ingles, espanol)


    # 7. ORDENAMIENTO VISUAL DE LAS COLUMNAS LÓGICAMENTE AGRUPADAS
    orden_columnas = [
        "numero_poliza", "numero_reporte",                                  # Identificadores
        "anio", "mes_accidente", "semana_accidente", "dia_accidente",       # Fechas Accidente
        "semana_reclamo", "dia_reclamo",                                    # Fechas Reclamo (El mes va al final por partición)
        "edad", "genero", "estado_civil", "calificacion_conductor",         # Datos Persona
        "marca_auto", "categoria_vehiculo", "tipo_vehiculo_poliza",         # Datos Vehículo
        "rango_precio_vehiculo", "antiguedad_vehiculo", "numero_autos", 
        "zona_accidente", "culpabilidad", "reporte_policial",               # Detalles Accidente
        "testigo_presente", "numero_suplementos", 
        "cobertura", "poliza_base", "deducible", "dias_poliza_accidente",   # Detalles Póliza e Historial
        "dias_poliza_reclamo", "reclamos_pasados", "cambio_direccion", 
        "edad_titular", "tipo_agente",
        "fraude_detectado",                                                 # Target / Objetivo
        "mes_reclamo"                                                       # <-- Columna de partición SIEMPRE AL FINAL en Hive
    ]
    
    return df.select(*orden_columnas)

def main():
    args = parse_arguments()
    spark = create_spark_session()
    
    db_curated = f"{args.env.lower()}_fraudes_curated"
    db_source = args.source_db
    table_name = "FRAUDES"
    
    print(f"🚀 Iniciando Curated. Origen: {db_source}")
    
    # 1. Crear BD
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_curated} LOCATION '{args.base_path}/warehouse/{db_curated}'")
    
    # 2. Leer de Landing
    df_source = spark.table(f"{db_source}.{table_name}")
    total_origen = df_source.count()
    
    # 3. Aplicar Limpieza, Transformación, Traducción y Orden
    df_transformed = limpiar_traducir_y_ordenar(df_source)
    
    # 4. Guardar en Parquet (Particionado)
    print("📥 Escribiendo datos limpios, ordenados y traducidos en Parquet...")
    df_transformed.write \
        .mode("overwrite") \
        .format("parquet") \
        .option("compression", "snappy") \
        .partitionBy("mes_reclamo") \
        .saveAsTable(f"{db_curated}.{table_name}")
        
    # 5. Reporte
    total_curated = spark.sql(f"SELECT COUNT(*) FROM {db_curated}.{table_name}").first()[0]
    print(f"\n📈 REPORTE DE CALIDAD:")
    print(f"Registros crudos en Landing: {total_origen}")
    print(f"Registros limpios en Curated: {total_curated}")
    print(f"Datos descartados (Basura/Nulos): {total_origen - total_curated}")
    
    print("\n🔍 MUESTRA DE DATOS AGRUPADOS Y EN ESPAÑOL:")
    df_transformed.select("numero_poliza", "mes_accidente", "dia_accidente", "mes_reclamo", "dia_reclamo", "estado_civil", "zona_accidente", "fraude_detectado").show(5, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()