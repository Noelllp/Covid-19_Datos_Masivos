"""
transform.py — Capa de transformación (Silver zone).

Responsabilidades:
  - Convertir la serie acumulativa en deltas diarios mediante Window lag()
  - Calcular promedios móviles de 7 días, tasa de crecimiento y CFR
  - Extraer componentes temporales (year, month, day_of_week, is_weekend)
  - Marcar filas anómalas cruzando con el registro de anomalías del NYT
  - Filtrar excepciones geográficas documentadas (NYC, Kansas City, Joplin)
  - Enriquecer condados con la encuesta de uso de mascarillas
  - Persistir los DataFrames procesados en Parquet (Silver zone)

Decisiones de diseño clave:
  - Window functions vs self-join para deltas: lag() sobre una Window es O(n)
    y no genera shuffle. Un self-join para calcular la diferencia de días
    consecutivos requeriría un sort-merge join O(n log n) con redistribución
    completa de datos entre nodos. Con 3.5M filas en condados, la diferencia
    es significativa.

  - Broadcast join para anomalías y mask-use: ambas tablas son pequeñas
    (~2,500 y ~3,200 filas respectivamente). Broadcast las distribuye a todos
    los ejecutores en memoria, eliminando el shuffle que un sort-merge join
    requeriría contra el DataFrame de condados (3.5M filas).

  - Orden de operaciones en transform_counties: las excepciones geográficas
    se filtran ANTES de calcular los deltas diarios. Si no, el lag() de un
    condado normal podría tomar el valor acumulado de una entidad de excepción
    que esté adyacente en el orden de filas, produciendo deltas incorrectos.
"""

import logging
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DoubleType, IntegerType

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PROCESSED_DIR, ROLLING_WINDOW_DAYS, GEOGRAPHIC_EXCEPTIONS

log = logging.getLogger(__name__)


# =============================================================================
# Especificaciones de Window reutilizables
# =============================================================================

def _national_window() -> Window:
    return Window.orderBy("date")

def _state_window() -> Window:
    return Window.partitionBy("state").orderBy("date")

# FIPS como clave de partición para condados: garantiza que lag() no cruce
# el límite entre dos condados distintos aunque estén adyacentes en los datos.
# Filas con FIPS nulo (condados "Unknown") forman su propia partición de nulo.
def _fips_window() -> Window:
    return Window.partitionBy("fips").orderBy("date")

# ROWS BETWEEN es preferible a RANGE BETWEEN para series con cadencia diaria
# uniforme. Con RANGE BETWEEN, si faltan fechas el rango se expande
# silenciosamente más allá de 7 días. ROWS garantiza exactamente 7 filas.
def _national_roll() -> Window:
    return (
        Window.orderBy("date")
        .rowsBetween(-(ROLLING_WINDOW_DAYS - 1), 0)
    )

def _state_roll() -> Window:
    return (
        Window.partitionBy("state").orderBy("date")
        .rowsBetween(-(ROLLING_WINDOW_DAYS - 1), 0)
    )

def _fips_roll() -> Window:
    return (
        Window.partitionBy("fips").orderBy("date")
        .rowsBetween(-(ROLLING_WINDOW_DAYS - 1), 0)
    )


# =============================================================================
# Transformaciones atómicas (funciones puras, sin efectos secundarios)
# =============================================================================

def compute_daily_deltas(df: DataFrame, geo_window: Window) -> DataFrame:
    """
    Convierte conteos acumulativos en incrementos diarios usando lag().

    El dataset del NYT almacena totales acumulados, no casos nuevos diarios.
    La diferencia cases[t] - cases[t-1] produce el incremento.

    Deltas negativos: el NYT corrige retroactivamente sus conteos (errores,
    reasignaciones entre condados). Esto produce valores negativos en la
    diferencia. Por diseño se imputan a 0 y se marcan con `delta_corrected=True`
    para que los analistas sepan que hubo una corrección sin eliminar el punto
    temporal de la serie.

    Para la primera fecha de cada geografía, lag() devuelve null → el delta
    es igual al acumulado (primer día reportado = todos los casos hasta ese día).
    """
    df = (
        df
        .withColumn("_prev_cases",  F.lag("cases",  1).over(geo_window))
        .withColumn("_prev_deaths", F.lag("deaths", 1).over(geo_window))
        .withColumn("_raw_daily_cases",  F.col("cases")  - F.col("_prev_cases"))
        .withColumn("_raw_daily_deaths", F.col("deaths") - F.col("_prev_deaths"))
    )

    df = df.withColumn(
        "delta_corrected",
        (F.col("_raw_daily_cases") < 0) | (F.col("_raw_daily_deaths") < 0)
    )

    # Primera fila de cada geografía: lag es null → daily = acumulado
    df = (
        df
        .withColumn(
            "daily_cases",
            F.when(F.col("_prev_cases").isNull(),  F.col("cases"))
             .otherwise(F.greatest(F.col("_raw_daily_cases"),  F.lit(0)))
        )
        .withColumn(
            "daily_deaths",
            F.when(F.col("_prev_deaths").isNull(), F.col("deaths"))
             .otherwise(F.greatest(F.col("_raw_daily_deaths"), F.lit(0)))
        )
    )

    return df.drop("_prev_cases", "_prev_deaths", "_raw_daily_cases", "_raw_daily_deaths")


def compute_rolling_averages(df: DataFrame, roll_window: Window) -> DataFrame:
    """
    Promedio móvil de 7 días para casos y muertes diarias.

    El promedio móvil es la métrica operativa estándar de vigilancia
    epidemiológica: suaviza el artefacto de reporte de fin de semana
    (acumulación Sab/Dom → pico el Lunes) revelando la tendencia real
    de transmisión.
    """
    return (
        df
        .withColumn(
            "rolling_avg_7d_cases",
            F.round(F.avg("daily_cases").over(roll_window), 2)
        )
        .withColumn(
            "rolling_avg_7d_deaths",
            F.round(F.avg("daily_deaths").over(roll_window), 2)
        )
    )


def compute_growth_rate(df: DataFrame, geo_window: Window) -> DataFrame:
    """
    Tasa de crecimiento de 7 días de los casos diarios.

    growth_rate = (casos_hoy - casos_hace_7_días) / casos_hace_7_días × 100

    Definida solo cuando el denominador > 0 para evitar divisiones por cero
    en los primeros días de la pandemia por estado. Valores positivos indican
    aceleración epidémica; negativos, desaceleración.
    """
    return (
        df
        .withColumn("_cases_7d_ago", F.lag("daily_cases", 7).over(geo_window))
        .withColumn(
            "growth_rate_7d_pct",
            F.when(
                F.col("_cases_7d_ago") > 0,
                F.round(
                    (F.col("daily_cases") - F.col("_cases_7d_ago"))
                    / F.col("_cases_7d_ago") * 100,
                    2
                )
            ).otherwise(F.lit(None).cast(DoubleType()))
        )
        .drop("_cases_7d_ago")
    )


def compute_cfr(df: DataFrame) -> DataFrame:
    """
    Case Fatality Rate (CFR) acumulado.

    CFR = muertes_acumuladas / casos_acumulados × 100

    Calculada sobre los acumulados (no los diarios) para reflejar la
    mortalidad real de todos los casos conocidos hasta cada fecha.
    Su evolución temporal documenta el impacto de las vacunas y la
    menor virulencia de las variantes tardías.
    """
    return df.withColumn(
        "cfr_pct",
        F.when(
            F.col("cases") > 0,
            F.round(F.col("deaths") / F.col("cases") * 100, 4)
        ).otherwise(F.lit(None).cast(DoubleType()))
    )


def extract_temporal_features(df: DataFrame) -> DataFrame:
    """
    Extrae componentes de fecha para análisis temporal y ML.

    day_of_week es crítico para modelar el artefacto de reporte de fin de
    semana: Spark usa 1=Domingo, 7=Sábado (convención Java Calendar).
    is_weekend captura Domingo (1) y Sábado (7) como booleano binario para
    uso directo como feature en modelos de clasificación/regresión.
    """
    return (
        df
        .withColumn("year",         F.year("date"))
        .withColumn("month",        F.month("date"))
        .withColumn("quarter",      F.quarter("date"))
        .withColumn("week_of_year", F.weekofyear("date"))
        .withColumn("day_of_week",  F.dayofweek("date"))
        .withColumn("is_weekend",   F.col("day_of_week").isin(1, 7))
    )


def flag_anomalies(df: DataFrame, anomalies_df: DataFrame) -> DataFrame:
    """
    Marca filas anómalas cruzando con el registro del NYT.

    El registro de anomalías (~2,510 registros) documenta rangos de fechas
    y geografías donde los datos son poco fiables (ej. el pico artificial
    del 14 de Abril 2020 cuando NY añadió ~3,700 muertes probables acumuladas
    en un solo día).

    Broadcast join: la tabla de anomalías es pequeña y se replica en todos
    los ejecutores. Alternativa (sort-merge join) redistribuiría el DataFrame
    grande innecesariamente.

    La columna `is_anomaly` es aditiva — no elimina datos. Los analistas
    pueden filtrar `WHERE NOT is_anomaly` para análisis de tendencias, o
    mantener todas las filas para análisis de completitud.
    """
    anomaly_lookup = F.broadcast(
        anomalies_df.select(
            F.col("geoid").alias("_anom_geoid"),
            F.col("date").alias("_anom_start"),
            F.col("end_date").alias("_anom_end"),
            "omit_from_rolling_average",
        )
    )

    joined = df.join(
        anomaly_lookup,
        on=(
            (df["geoid"] == anomaly_lookup["_anom_geoid"]) &
            (df["date"] >= anomaly_lookup["_anom_start"]) &
            (
                anomaly_lookup["_anom_end"].isNull() |
                (df["date"] <= anomaly_lookup["_anom_end"])
            )
        ),
        how="left",
    )

    return (
        joined
        .withColumn("is_anomaly", F.col("omit_from_rolling_average").isNotNull())
        .drop("_anom_geoid", "_anom_start", "_anom_end", "omit_from_rolling_average")
    )


def filter_geographic_exceptions(df: DataFrame, col_name: str = "county") -> DataFrame:
    """
    Excluye las excepciones geográficas documentadas del análisis de condados.

    NYC, Kansas City y Joplin se reportan como entidades independientes
    solapando los condados que las contienen. Incluirlas en agregaciones
    a nivel condado produce doble conteo.
    """
    exceptions = list(GEOGRAPHIC_EXCEPTIONS)
    log.info(f"Filtrando excepciones geográficas: {exceptions}")
    return df.filter(~F.col(col_name).isin(exceptions))


def join_mask_use(df: DataFrame, mask_df: DataFrame) -> DataFrame:
    """
    Enriquece el DataFrame de condados con la encuesta de uso de mascarillas.

    `mask_compliance_score`: puntaje compuesto ponderado que refleja el
    nivel de adopción de mascarillas. Ponderaciones: always=1.0,
    frequently=0.75, sometimes=0.5, rarely=0.25, never=0.0.
    El resultado es un índice en [0, 1] donde 1 = uso universal constante.

    Join por FIPS (5 dígitos, zero-padded en ambas tablas). Left join: los
    condados sin registro de encuesta conservan sus datos con score null.
    """
    mask_enriched = (
        mask_df
        .withColumn(
            "mask_compliance_score",
            F.round(
                F.col("always")     * 1.00 +
                F.col("frequently") * 0.75 +
                F.col("sometimes")  * 0.50 +
                F.col("rarely")     * 0.25 +
                F.col("never")      * 0.00,
                4
            )
        )
        .withColumnRenamed("countyfp", "fips")
        .select("fips", "never", "rarely", "sometimes", "frequently", "always",
                "mask_compliance_score")
    )

    # Broadcast: mask_enriched tiene ~3,200 filas vs 3.5M en el DataFrame de condados
    return df.join(F.broadcast(mask_enriched), on="fips", how="left")


# =============================================================================
# Pipelines de transformación por nivel geográfico
# =============================================================================

def transform_national(df: DataFrame) -> DataFrame:
    """Pipeline completo para el dataset nacional."""
    log.info("Transformando datos nacionales...")
    df = compute_daily_deltas(df, _national_window())
    df = compute_rolling_averages(df, _national_roll())
    df = compute_growth_rate(df, _national_window())
    df = compute_cfr(df)
    df = extract_temporal_features(df)
    return df


def transform_states(
    df: DataFrame,
    anomalies_df: Optional[DataFrame] = None,
) -> DataFrame:
    """
    Pipeline completo para el dataset estatal.

    Construye la columna `geoid` en formato "USA-NN" (donde NN es el FIPS
    estatal) antes del join de anomalías, ya que el registro de anomalías
    usa geoid como clave de unión — no el nombre del estado.
    """
    log.info("Transformando datos por estado...")

    # geoid estatal: "USA-" + FIPS de 2 dígitos (ej. "USA-01" para Alabama)
    df = df.withColumn(
        "geoid",
        F.when(F.col("fips").isNotNull(), F.concat(F.lit("USA-"), F.col("fips")))
         .otherwise(F.lit(None))
    )

    df = compute_daily_deltas(df, _state_window())
    df = compute_rolling_averages(df, _state_roll())
    df = compute_growth_rate(df, _state_window())
    df = compute_cfr(df)
    df = extract_temporal_features(df)

    if anomalies_df is not None:
        df = flag_anomalies(df, anomalies_df)

    return df


def transform_counties(
    df: DataFrame,
    anomalies_df: Optional[DataFrame] = None,
    mask_df: Optional[DataFrame] = None,
) -> DataFrame:
    """
    Pipeline completo para el dataset consolidado de condados.

    Orden de operaciones — el orden importa:
      1. filter_geographic_exceptions: antes de lag() para que las entidades
         especiales no contaminen los deltas de condados adyacentes.
      2. Etiquetar unknown_county: preserva filas sin FIPS con flag explícita.
      3. Construir geoid: FIPS de 5 dígitos zero-padded (lpad) para join con anomalías.
      4. compute_daily_deltas: requiere filas filtradas y ordenadas por FIPS.
      5. compute_rolling_averages, compute_growth_rate, compute_cfr.
      6. extract_temporal_features.
      7. join_mask_use: sobre filas ya validadas — no aportar mask score a
         filas que serán descartadas por unknown_county en análisis.
      8. flag_anomalies: al final porque es aditivo — no afecta cálculos anteriores.
    """
    log.info("Transformando datos de condados...")

    df = filter_geographic_exceptions(df, col_name="county")

    df = df.withColumn("unknown_county", F.col("fips").isNull())

    # geoid de condado: FIPS de 5 dígitos con cero a la izquierda
    df = df.withColumn(
        "geoid",
        F.when(F.col("fips").isNotNull(), F.lpad(F.col("fips"), 5, "0"))
         .otherwise(F.lit(None))
    )

    df = compute_daily_deltas(df, _fips_window())
    df = compute_rolling_averages(df, _fips_roll())
    df = compute_growth_rate(df, _fips_window())
    df = compute_cfr(df)
    df = extract_temporal_features(df)

    if mask_df is not None:
        df = join_mask_use(df, mask_df)

    if anomalies_df is not None:
        df = flag_anomalies(df, anomalies_df)

    return df


# =============================================================================
# Escritura a Silver zone y orquestador
# =============================================================================

def write_silver(df: DataFrame, name: str, partition_cols: list = None) -> None:
    """
    Persiste un DataFrame transformado en la Silver zone como Parquet.

    Silver zone contiene datos limpios y enriquecidos, listos para análisis.
    La misma estrategia de particionamiento que Bronze se aplica aquí para
    que las queries de analytics.py usen predicate pushdown.
    """
    output_path = str(Path(PROCESSED_DIR) / "silver" / name)
    writer = df.write.mode("overwrite").format("parquet")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.save(output_path)
    log.info(f"[silver/{name}] Escrito en {output_path}")


def run_transformations(datasets: dict) -> dict:
    """
    Orquesta el stage de transformación completo.

    Los DataFrames de states y counties se cachean después de su transformación
    porque analytics.py los consume múltiples veces (descriptivo + MapReduce + ML).
    Sin caché, Spark re-ejecutaría toda la cadena de transformaciones en cada acción.

    El caché se libera explícitamente después de escribir a Silver para evitar
    mantener 3.5M filas en memoria más tiempo del necesario.
    """
    try:
        national_t = transform_national(datasets["national"])

        states_t = transform_states(
            datasets["states"],
            anomalies_df=datasets.get("anomalies"),
        )

        counties_t = transform_counties(
            datasets["counties"],
            anomalies_df=datasets.get("anomalies"),
            mask_df=datasets.get("mask_use"),
        )

        # Caché: ambos DataFrames se usan en descriptivo + MapReduce + ML
        states_t.cache()
        counties_t.cache()

        write_silver(national_t, "national")
        write_silver(states_t,   "states",   partition_cols=["state"])
        write_silver(counties_t, "counties", partition_cols=["state", "source_year"])

        log.info("Stage de transformación completado.")

        return {
            "national": national_t,
            "states":   states_t,
            "counties": counties_t,
        }

    except Exception as exc:
        log.critical(f"Stage de transformación fallido: {exc}", exc_info=True)
        raise
