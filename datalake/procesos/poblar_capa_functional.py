#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg,
    round as spark_round, when, current_timestamp, lit,
    regexp_extract
)
from pyspark.sql.types import IntegerType

# =============================================================================
# ARGUMENTOS
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',       type=str, default='DEV')
    parser.add_argument('--username',  type=str, default='hadoop')
    parser.add_argument('--base_path', type=str, default='/user/hadoop/topicos-deteccion-fraudes')
    parser.add_argument('--source_db', type=str, default='curated')
    return parser.parse_args()

# =============================================================================
# SPARK SESSION
# =============================================================================

def create_spark_session():
    return SparkSession.builder \
        .appName("Functional_Fraudes_KPI_ML") \
        .enableHiveSupport() \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .getOrCreate()

# =============================================================================
# ENRIQUECIMIENTO + FEATURE ENGINEERING
# =============================================================================

def enriquecer_y_preparar_ml(df):
    """
    Se basa en las columnas REALES del Curated:
      - rango_precio_vehiculo  (string: "20000 to 29000", "less than 20000", "more than 69000")
      - dias_poliza_accidente  (string: "1 to 7", "none", "more than 30")
      - dias_poliza_reclamo    (string: "8 to 15", "none", "more than 30")
      - reclamos_pasados       (string: "none", "1", "2 to 4", "more than 4")
      - semana_accidente / semana_reclamo (integer)
      - reporte_policial / testigo_presente (string: "Sí" / "No")
      - zona_accidente         (string: "Urbano" / "Rural")
      - genero                 (string: "Masculino" / "Femenino")
      - fraude_detectado       (integer: 0 / 1)
      - deducible              (integer)
    """

    # ------------------------------------------------------------------
    # 1. VALOR ESTIMADO DEL VEHÍCULO (punto medio del rango de precio)
    # ------------------------------------------------------------------
    df = df.withColumn(
        "valor_estimado_vehiculo",
        when(col("rango_precio_vehiculo").rlike(r"(?i)less than"),   15000)
        .when(col("rango_precio_vehiculo").rlike(r"20000 to 29000"), 25000)
        .when(col("rango_precio_vehiculo").rlike(r"30000 to 39000"), 35000)
        .when(col("rango_precio_vehiculo").rlike(r"40000 to 59000"), 50000)
        .when(col("rango_precio_vehiculo").rlike(r"60000 to 69000"), 65000)
        .when(col("rango_precio_vehiculo").rlike(r"(?i)more than"),  80000)
        .otherwise(30000)
    )

    # ------------------------------------------------------------------
    # 2. PROXY DE RETRASO EN REPORTE
    #    semana_reclamo - semana_accidente (ambas Integer en Curated)
    # ------------------------------------------------------------------
    df = df.withColumn(
        "semanas_retraso_reporte",
        when(
            col("semana_reclamo").isNotNull() & col("semana_accidente").isNotNull(),
            when((col("semana_reclamo") - col("semana_accidente")) < 0, 0)
            .otherwise(col("semana_reclamo") - col("semana_accidente"))
        ).otherwise(None)
    )

    # ------------------------------------------------------------------
    # 3. DÍAS MÍNIMOS DE PÓLIZA ANTES DEL ACCIDENTE (extraído del string)
    #    "1 to 7" → 1 | "none" → 0 | "more than 30" → 31
    # ------------------------------------------------------------------
    df = df.withColumn(
        "dias_poliza_accidente_min",
        when(col("dias_poliza_accidente").rlike(r"(?i)^none$"), 0)
        .when(col("dias_poliza_accidente").rlike(r"(?i)more than (\d+)"),
              regexp_extract(col("dias_poliza_accidente"), r"more than (\d+)", 1)
              .cast(IntegerType()) + 1)
        .when(col("dias_poliza_accidente").rlike(r"(\d+) to (\d+)"),
              regexp_extract(col("dias_poliza_accidente"), r"(\d+) to", 1)
              .cast(IntegerType()))
        .otherwise(None)
    )

    # ------------------------------------------------------------------
    # 4. RECLAMOS PASADOS MÍNIMOS (extraído del string)
    #    "none" → 0 | "1" → 1 | "2 to 4" → 2 | "more than 4" → 5
    # ------------------------------------------------------------------
    df = df.withColumn(
        "reclamos_pasados_min",
        when(col("reclamos_pasados").rlike(r"(?i)^none$"), 0)
        .when(col("reclamos_pasados").rlike(r"(?i)more than (\d+)"),
              regexp_extract(col("reclamos_pasados"), r"more than (\d+)", 1)
              .cast(IntegerType()) + 1)
        .when(col("reclamos_pasados").rlike(r"(\d+) to (\d+)"),
              regexp_extract(col("reclamos_pasados"), r"(\d+) to", 1)
              .cast(IntegerType()))
        .when(col("reclamos_pasados").rlike(r"^\d+$"),
              col("reclamos_pasados").cast(IntegerType()))
        .otherwise(None)
    )

    # ------------------------------------------------------------------
    # 5. FEATURE ENGINEERING BINARIO (para modelos ML)
    # ------------------------------------------------------------------
    df = df.withColumn("zona_accidente_ml",
                       when(col("zona_accidente") == "Urbano", 1).otherwise(0)) \
           .withColumn("reporte_policial_ml",
                       when(col("reporte_policial") == "Sí", 1).otherwise(0)) \
           .withColumn("testigo_presente_ml",
                       when(col("testigo_presente") == "Sí", 1).otherwise(0)) \
           .withColumn("genero_ml",
                       when(col("genero") == "Femenino", 1).otherwise(0))

    # ------------------------------------------------------------------
    # 6. SCORE DE RIESGO HEURÍSTICO (0-5 señales de alerta por registro)
    # ------------------------------------------------------------------
    df = df.withColumn(
        "score_riesgo",
        (
            when(col("testigo_presente") == "No", 1).otherwise(0) +
            when(col("reporte_policial") == "No", 1).otherwise(0) +
            when(col("reclamos_pasados_min") > 0, 1).otherwise(0) +
            when(col("semanas_retraso_reporte") >= 2, 1).otherwise(0) +
            when(
                col("dias_poliza_accidente_min").isNotNull() &
                (col("dias_poliza_accidente_min") <= 7), 1
            ).otherwise(0)
        )
    )

    # ------------------------------------------------------------------
    # 7. GOBERNANZA DE DATOS (trazabilidad)
    # ------------------------------------------------------------------
    df = df.withColumn("gob_fecha_procesamiento", current_timestamp()) \
           .withColumn("gob_fuente_datos",  lit("Sistema_Seguros_Oracle")) \
           .withColumn("gob_calidad_datos", lit("Validado_Curated"))

    # ------------------------------------------------------------------
    # 8. PLACEHOLDERS ML (se actualizarán al entrenar el modelo)
    # ------------------------------------------------------------------
    df = df.withColumn("ml_probabilidad_fraude", lit(0.0)) \
           .withColumn("ml_prediccion",          lit(0))

    return df

# =============================================================================
# KPIs DE NEGOCIO
# =============================================================================

def generar_metricas_kpi(df_enriquecido):
    """
    Tabla METRICAS_FRAUDE con KPIs estratégicos agrupados por
    mes_reclamo + zona_accidente + tipo_vehiculo_poliza.
    """
    df_kpis = df_enriquecido.groupBy(
        "mes_reclamo", "zona_accidente", "tipo_vehiculo_poliza"
    ).agg(
        # Volumen base
        count("*").alias("total_reclamos"),
        spark_sum("fraude_detectado").alias("total_fraudes"),

        # KPI 1: Tasa de fraude (%)
        spark_round(
            (spark_sum("fraude_detectado") / count("*")) * 100, 2
        ).alias("kpi_tasa_fraude_pct"),

        # KPI 2: Pérdidas evitadas estimadas en USD
        spark_sum(
            when(col("fraude_detectado") == 1,
                 col("valor_estimado_vehiculo") - col("deducible"))
            .otherwise(0)
        ).alias("kpi_perdidas_evitadas_usd"),

        # KPI 3: Semanas promedio de retraso en reporte
        spark_round(avg("semanas_retraso_reporte"), 1)
          .alias("kpi_semanas_prom_retraso"),

        # KPI 4: Score de riesgo promedio del segmento
        spark_round(avg("score_riesgo"), 2)
          .alias("kpi_score_riesgo_promedio"),

        # KPI 5: Costo estimado de investigación (~$500 por caso)
        spark_round(
            spark_sum("fraude_detectado") * lit(500) / count("*"), 2
        ).alias("kpi_costo_investigacion_promedio_usd"),

        # KPI 6: % reclamos sin testigo
        spark_round(
            (spark_sum(when(col("testigo_presente") == "No", 1).otherwise(0)) / count("*")) * 100, 2
        ).alias("kpi_pct_sin_testigo"),

        # KPI 7: % reclamos sin reporte policial
        spark_round(
            (spark_sum(when(col("reporte_policial") == "No", 1).otherwise(0)) / count("*")) * 100, 2
        ).alias("kpi_pct_sin_reporte_policial"),


    )

    return df_kpis.filter(col("mes_reclamo").isNotNull())

# =============================================================================
# MAIN
# =============================================================================

def main():
    args  = parse_arguments()
    spark = create_spark_session()

    db_curated    = f"{args.env.lower()}_fraudes_{args.source_db}"
    db_functional = f"{args.env.lower()}_fraudes_functional"

    print(f"🚀 Iniciando capa Functional. Leyendo desde: {db_curated}")

    # 1. Crear BD Functional
    spark.sql(f"""
        CREATE DATABASE IF NOT EXISTS {db_functional}
        LOCATION '{args.base_path}/warehouse/{db_functional}'
    """)

    # 2. Leer Curated
    df_curated    = spark.table(f"{db_curated}.FRAUDES")
    total_curated = df_curated.count()
    print(f"📊 Registros leídos de Curated: {total_curated:,}")

    # 3. Enriquecer tabla principal
    print("📥 1. Generando FRAUDES_ENRIQUECIDO...")
    df_enriquecido = enriquecer_y_preparar_ml(df_curated)
    df_enriquecido.cache()

    df_enriquecido.write \
        .mode("overwrite") \
        .format("parquet") \
        .option("compression", "snappy") \
        .partitionBy("mes_reclamo") \
        .saveAsTable(f"{db_functional}.FRAUDES_ENRIQUECIDO")

    # 4. Generar KPIs
    print("📥 2. Generando METRICAS_FRAUDE (KPIs)...")
    df_kpis = generar_metricas_kpi(df_enriquecido)

    df_kpis.write \
        .mode("overwrite") \
        .format("parquet") \
        .option("compression", "snappy") \
        .partitionBy("mes_reclamo") \
        .saveAsTable(f"{db_functional}.METRICAS_FRAUDE")

    # 5. Reporte final
    total_enriquecido = spark.sql(f"SELECT COUNT(*) FROM {db_functional}.FRAUDES_ENRIQUECIDO").first()[0]
    total_kpis        = spark.sql(f"SELECT COUNT(*) FROM {db_functional}.METRICAS_FRAUDE").first()[0]

    print(f"\n✅ TABLAS GENERADAS EN FUNCTIONAL:")
    spark.sql(f"SHOW TABLES IN {db_functional}").show(truncate=False)
    print(f"  FRAUDES_ENRIQUECIDO : {total_enriquecido:,} registros")
    print(f"  METRICAS_FRAUDE     : {total_kpis:,} filas de KPIs\n")

    print("🔍 MUESTRA DE KPIs (top 10 por tasa de fraude):")
    spark.sql(f"""
        SELECT mes_reclamo, zona_accidente, tipo_vehiculo_poliza,
               total_reclamos, total_fraudes,
               kpi_tasa_fraude_pct,
               kpi_perdidas_evitadas_usd,
               kpi_score_riesgo_promedio
        FROM {db_functional}.METRICAS_FRAUDE
        ORDER BY kpi_tasa_fraude_pct DESC
        LIMIT 10
    """).show(truncate=False)

    df_enriquecido.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()