"""
ingest.py — Capa de ingesta (Bronze zone).

Responsabilidades:
  - Leer los CSV del NYT con esquemas explícitos (evita inferencia costosa)
  - Validar calidad básica post-carga (conteo de filas, tasa de nulos)
  - Consolidar los 4 archivos anuales de condados en un único DataFrame
  - Persistir en formato Parquet particionado (Bronze zone)

Decisiones de diseño:
  - Esquemas explícitos: la inferencia de esquemas en Spark lee el dataset
    completo dos veces (una para inferir, otra para procesar). Con 3.5M filas
    en condados, esto duplica el tiempo de lectura innecesariamente.
  - FIPS como StringType: los FIPS de condado tienen ceros a la izquierda
    significativos (ej. "01051" para Alabama). Leerlos como IntegerType los
    destruiría, rompiendo todos los joins posteriores.
  - mode=PERMISSIVE en lugar de FAILFAST para los CSVs del NYT: el dataset
    contiene filas deliberadamente incompletas (condados "Unknown" sin FIPS)
    que son datos válidos, no corrupción. FAILFAST abortaría la carga.
  - Parquet con Snappy: 10x más compacto que CSV, lectura columnar, predicate
    pushdown en date/state para que los stages posteriores lean solo las
    particiones necesarias.
"""

import logging
from pathlib import Path
from typing import Dict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, DateType, DoubleType, IntegerType,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PROCESSED_DIR, COUNTY_YEARS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# Definiciones de esquema
# =============================================================================
# Cada esquema refleja exactamente las columnas verificadas en los archivos
# fuente reales. Los nombres de campo son minúsculas independientemente de
# cómo estén en el CSV: Spark los mapea por posición cuando se usa header=true
# con esquema explícito (el header se salta automáticamente).

SCHEMA_NATIONAL = StructType([
    StructField("date",   DateType(),   nullable=False),
    StructField("cases",  LongType(),   nullable=True),
    StructField("deaths", LongType(),   nullable=True),
])

SCHEMA_STATES = StructType([
    StructField("date",   DateType(),   nullable=False),
    StructField("state",  StringType(), nullable=False),
    # FIPS estatal es de 2 dígitos con cero a la izquierda (ej. "01" = Alabama)
    StructField("fips",   StringType(), nullable=True),
    StructField("cases",  LongType(),   nullable=True),
    StructField("deaths", LongType(),   nullable=True),
])

SCHEMA_COUNTIES = StructType([
    StructField("date",   DateType(),   nullable=False),
    StructField("county", StringType(), nullable=False),
    StructField("state",  StringType(), nullable=False),
    # FIPS de condado es de 5 dígitos con cero a la izquierda (ej. "01051")
    # Nulo para las excepciones geográficas (NYC, Kansas City, Joplin)
    StructField("fips",   StringType(), nullable=True),
    StructField("cases",  LongType(),   nullable=True),
    StructField("deaths", LongType(),   nullable=True),
])

SCHEMA_ROLLING_STATES = StructType([
    StructField("date",                DateType(),   nullable=False),
    # geoid tiene formato "USA-NN" donde NN es el FIPS estatal (ej. "USA-53")
    StructField("geoid",               StringType(), nullable=True),
    StructField("state",               StringType(), nullable=False),
    StructField("cases",               LongType(),   nullable=True),
    StructField("cases_avg",           DoubleType(), nullable=True),
    StructField("cases_avg_per_100k",  DoubleType(), nullable=True),
    StructField("deaths",              LongType(),   nullable=True),
    StructField("deaths_avg",          DoubleType(), nullable=True),
    StructField("deaths_avg_per_100k", DoubleType(), nullable=True),
])

SCHEMA_ANOMALIES = StructType([
    StructField("date",       DateType(),   nullable=True),
    StructField("end_date",   DateType(),   nullable=True),
    StructField("county",     StringType(), nullable=True),
    StructField("state",      StringType(), nullable=True),
    StructField("geoid",      StringType(), nullable=True),
    StructField("type",       StringType(), nullable=True),
    StructField("omit_from_rolling_average",                   StringType(), nullable=True),
    StructField("omit_from_rolling_average_on_subgeographies", StringType(), nullable=True),
    StructField("adjusted_daily_count_for_avg",                LongType(),   nullable=True),
    StructField("description", StringType(), nullable=True),
])

# El CSV de mask-use tiene cabeceras en MAYÚSCULAS (COUNTYFP, NEVER, RARELY...).
# Spark con header=true y esquema explícito mapea por posición ordinal, no por nombre,
# por lo que los field names del esquema sobreescriben las cabeceras del CSV.
# Esto nos permite normalizar a minúsculas sin una transformación adicional.
SCHEMA_MASK = StructType([
    StructField("countyfp",   StringType(), nullable=False),
    StructField("never",      DoubleType(), nullable=True),
    StructField("rarely",     DoubleType(), nullable=True),
    StructField("sometimes",  DoubleType(), nullable=True),
    StructField("frequently", DoubleType(), nullable=True),
    StructField("always",     DoubleType(), nullable=True),
])


# =============================================================================
# SparkSession
# =============================================================================

def build_spark_session(config: Dict[str, str]) -> SparkSession:
    """
    Construye y devuelve una SparkSession con configuración optimizada.

    AQE (Adaptive Query Execution) permite a Spark re-optimizar joins y
    agregaciones en tiempo de ejecución basándose en estadísticas reales de
    particiones, en lugar de estimaciones del planificador. Esencial para datos
    con distribución asimétrica (unos pocos condados concentran la mayoría de casos).
    """
    builder = SparkSession.builder
    for key, value in config.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log.info(f"SparkSession iniciada — versión Spark: {spark.version}")
    return spark


# =============================================================================
# Helpers internos
# =============================================================================

def _read_csv(
    spark: SparkSession,
    path: str,
    schema: StructType,
) -> DataFrame:
    """
    Lee un CSV con esquema explícito.

    PERMISSIVE + columnNameOfCorruptRecord: las filas malformadas van a una
    columna de diagnóstico en lugar de abortar la carga. Permite auditar
    exactamente qué filas fallaron sin perder el resto del dataset.
    """
    return (
        spark.read
        .option("header", "true")
        .option("dateFormat", "yyyy-MM-dd")
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .option("nullValue", "")
        .schema(schema)
        .csv(path)
    )


def _validate(df: DataFrame, name: str, min_rows: int = 100) -> DataFrame:
    """
    Validación liviana post-carga — una única acción Spark.

    Calcula conteo y tasa de nulos en un solo .collect() para evitar
    lanzar dos jobs completos por archivo (count + select separados).
    Con 6 datasets en la ingesta, esto elimina 6 jobs Spark innecesarios.
    """
    data_cols = [c for c in df.columns if c != "_corrupt_record"]
    stats = df.select(
        F.count("*").alias("_total"),
        *[
            F.round(F.mean(F.col(c).isNull().cast("int")) * 100, 2).alias(c)
            for c in data_cols
        ],
    ).collect()[0]

    n = stats["_total"]
    if n < min_rows:
        raise ValueError(
            f"[{name}] Conteo de filas insuficiente: {n}. "
            "Verificar ruta del archivo o integridad del dataset."
        )
    log.info(f"[{name}] {n:,} filas cargadas.")

    for col in data_cols:
        rate = stats[col]
        if rate and rate > 0:
            log.warning(f"[{name}] '{col}': {rate:.1f}% nulos.")

    return df


# =============================================================================
# Funciones de ingesta por dataset
# =============================================================================

def ingest_national(spark: SparkSession, path: str) -> DataFrame:
    """Carga el dataset nacional diario (us.csv)."""
    log.info("Ingiriendo datos nacionales...")
    df = _read_csv(spark, path, SCHEMA_NATIONAL)
    return _validate(df, "national", min_rows=1_000)


def ingest_states(spark: SparkSession, path: str) -> DataFrame:
    """Carga el dataset estatal diario (us-states.csv)."""
    log.info("Ingiriendo datos por estado...")
    df = _read_csv(spark, path, SCHEMA_STATES)
    return _validate(df, "states", min_rows=50_000)


def ingest_counties(spark: SparkSession, paths: Dict[int, str]) -> DataFrame:
    """
    Carga los archivos de condado anuales y los consolida en un único DataFrame.

    El NYT divide los datos por año para mantener tamaños de archivo manejables
    (~880K-1.2M filas/año). Se consolidan aquí con una columna `source_year`
    que permite a los stages posteriores particionar o filtrar por año sin
    re-ingerir.

    UNION vs UNION DISTINCT: se usa UNION ALL (por defecto en Spark .union())
    porque la deduplicación estricta por clave (date, fips) ocurre en
    transform.py después de la normalización de FIPS, evitando eliminar filas
    legítimas por duplicados aparentes entre archivos que comparten la última
    fecha del año.
    """
    log.info("Ingiriendo datos de condados por año...")
    frames = []

    for year, path in sorted(paths.items()):
        try:
            df = _read_csv(spark, path, SCHEMA_COUNTIES)
            df = df.withColumn("source_year", F.lit(year).cast(IntegerType()))
            df = _validate(df, f"counties_{year}", min_rows=100_000)
            frames.append(df)
            log.info(f"[counties_{year}] Archivo cargado: {path}")
        except Exception as exc:
            log.error(f"Error al ingerir condados del año {year}: {exc}")
            raise

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.union(frame)

    total = combined.count()
    log.info(f"Condados consolidados: {total:,} filas totales ({len(frames)} años).")
    return combined


def ingest_rolling_averages(spark: SparkSession, path: str) -> DataFrame:
    """
    Carga promedios móviles precomputados por estado.

    Se usa como fuente de validación cruzada (comparar contra nuestro cálculo
    propio de rolling average) y como fuente de la métrica normalizada
    cases_avg_per_100k, que requiere datos de población del Censo que el
    dataset crudo de us-states.csv no contiene.
    """
    log.info("Ingiriendo promedios móviles (estados)...")
    df = _read_csv(spark, path, SCHEMA_ROLLING_STATES)
    return _validate(df, "rolling_avg_states", min_rows=50_000)


def ingest_anomalies(spark: SparkSession, path: str) -> DataFrame:
    """
    Carga el registro de anomalías del NYT (~2,510 registros).

    Esta tabla se hará broadcast en transform.py: es pequeña y se une
    contra DataFrames de millones de filas, por lo que broadcast evita
    un sort-merge join costoso.
    """
    log.info("Ingiriendo registro de anomalías...")
    df = _read_csv(spark, path, SCHEMA_ANOMALIES)
    count = df.count()
    log.info(f"[anomalies] {count} registros de anomalía cargados.")
    return df


def ingest_mask_use(spark: SparkSession, path: str) -> DataFrame:
    """
    Carga la encuesta de uso de mascarillas por condado (Julio 2020).

    Las cabeceras del CSV son mayúsculas (COUNTYFP, NEVER, etc.).
    El esquema SCHEMA_MASK normaliza los nombres a minúsculas en la lectura
    mediante mapeo posicional, sin necesidad de un paso de renombrado posterior.
    """
    log.info("Ingiriendo datos de uso de mascarillas...")
    df = _read_csv(spark, path, SCHEMA_MASK)
    return _validate(df, "mask_use", min_rows=3_000)


# =============================================================================
# Escritura a Bronze zone
# =============================================================================

def write_bronze(df: DataFrame, name: str, partition_cols: list = None) -> None:
    """
    Persiste un DataFrame en la zona Bronze como Parquet particionado.

    La estrategia de particionamiento está calibrada para los patrones de
    consulta típicos de cada dataset:
      - states:   particionado por 'state' → queries filtradas por estado
                  leen una sola partición en lugar del dataset completo.
      - counties: particionado por ('state', 'source_year') → el par más
                  frecuente en filtros de análisis (análisis estatal por período).
      - national: sin partición → dataset pequeño, un único archivo Parquet.
    """
    output_path = str(Path(PROCESSED_DIR) / "bronze" / name)
    writer = df.write.mode("overwrite").format("parquet")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.save(output_path)
    log.info(f"[bronze/{name}] Escrito en {output_path}")


# =============================================================================
# Orquestador de ingesta
# =============================================================================

def run_ingestion(spark: SparkSession, data_paths: Dict[str, str]) -> Dict[str, DataFrame]:
    """
    Orquesta la ingesta completa y devuelve un diccionario de DataFrames.

    Retorna los DataFrames en lugar de solo escribir a disco para que el
    pipeline principal pueda pasar directamente al stage de transformación
    sin re-leer desde Parquet (la re-lectura añade latencia de I/O en entornos
    locales donde el overhead de serialización/deserialización domina).
    """
    county_paths = {
        yr: data_paths[f"us_counties_{yr}"]
        for yr in COUNTY_YEARS
        if f"us_counties_{yr}" in data_paths
    }

    datasets: Dict[str, DataFrame] = {}
    try:
        datasets["national"]       = ingest_national(spark, data_paths["us_national"])
        datasets["states"]         = ingest_states(spark, data_paths["us_states"])
        datasets["counties"]       = ingest_counties(spark, county_paths)
        datasets["rolling_states"] = ingest_rolling_averages(spark, data_paths["rolling_avg_states"])
        datasets["anomalies"]      = ingest_anomalies(spark, data_paths["anomalies"])
        datasets["mask_use"]       = ingest_mask_use(spark, data_paths["mask_use"])
    except Exception as exc:
        log.critical(f"Stage de ingesta fallido: {exc}", exc_info=True)
        raise

    write_bronze(datasets["national"],       "national")
    write_bronze(datasets["states"],         "states",   partition_cols=["state"])
    write_bronze(datasets["counties"],       "counties", partition_cols=["state", "source_year"])
    write_bronze(datasets["rolling_states"], "rolling_states", partition_cols=["state"])

    log.info("Stage de ingesta completado.")
    return datasets
