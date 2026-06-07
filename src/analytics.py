"""
analytics.py — Capa analítica descriptiva y predictiva (Gold zone).

Responsabilidades:
  - Estadísticas descriptivas: resúmenes temporales, rankings, análisis de olas
  - Jobs explícitos de MapReduce mediante la API RDD de Spark
  - Ingeniería de features y entrenamiento de modelos con MLlib
  - Escritura de resultados en la Gold zone (Parquet)

Sobre los jobs de MapReduce:
  Los jobs de esta sección usan la API de bajo nivel RDD (.map(), .reduceByKey(),
  .groupByKey()) para exponer el modelo de programación MapReduce de forma
  explícita, tal como requiere la especificación del proyecto universitario.
  Estas mismas operaciones podrían expresarse más concisamente con la API
  DataFrame — la implementación RDD es deliberada para demostrar el paradigma
  de computación distribuida subyacente.

  Paradigma Map → Shuffle/Sort → Reduce:
    Map:    transforma cada fila en pares (clave, valor)
    Shuffle: agrupa todos los pares por clave entre particiones/nodos
    Reduce: aplica una función de agregación a todos los valores de cada clave
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler,
    StringIndexer, OneHotEncoder,
)
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    PROCESSED_DIR, ROLLING_WINDOW_DAYS, ML_CONFIG,
    WAVE_DEFINITIONS, WAVE_LAST_LABEL, EXPECTED_DAYS,
)

log = logging.getLogger(__name__)


# =============================================================================
# SECCIÓN 1: ESTADÍSTICAS DESCRIPTIVAS (DataFrame API)
# =============================================================================

def describe_national_timeline(national_df: DataFrame) -> DataFrame:
    """
    Resumen mensual del timeline nacional.

    Agrupa por (year, month) y calcula: casos totales, muertes totales,
    CFR promedio, tasa de crecimiento promedio y pico del promedio móvil.
    Esta es la vista fundacional para entender el arco completo de la pandemia.
    """
    log.info("Calculando resumen mensual nacional...")
    return (
        national_df
        .groupBy("year", "month")
        .agg(
            F.sum("daily_cases").alias("monthly_cases"),
            F.sum("daily_deaths").alias("monthly_deaths"),
            F.round(F.avg("cfr_pct"), 4).alias("avg_cfr_pct"),
            F.round(F.avg("growth_rate_7d_pct"), 2).alias("avg_growth_rate_pct"),
            F.round(F.max("rolling_avg_7d_cases"), 1).alias("peak_rolling_avg_cases"),
        )
        .orderBy("year", "month")
    )


def describe_state_rankings(states_df: DataFrame) -> DataFrame:
    """
    Ranking nacional de estados por casos acumulados y CFR al cierre del dataset.

    Usa row_number() sobre una Window particionada por estado para extraer el
    snapshot del último día de cada estado sin un GroupBy adicional.
    Los rank() se aplican sobre el snapshot resultante (~56 filas) tras
    colectarlo como DataFrame independiente, garantizando ordenamiento global
    sin ambigüedad semántica de Window sin partitionBy.
    """
    log.info("Calculando rankings por estado...")
    last_row = Window.partitionBy("state").orderBy(F.desc("date"))

    snapshot = (
        states_df
        .withColumn("_rn", F.row_number().over(last_row))
        .filter(F.col("_rn") == 1)
        .select("state", "fips", "cases", "deaths", "cfr_pct")
        .drop("_rn")
    )

    # rank() sobre Window global es aceptable aquí: el snapshot tiene ~56 filas
    # (una por estado/territorio), por lo que no hay riesgo de OOM en el driver.
    rank_cases = Window.orderBy(F.desc("cases"))
    rank_cfr   = Window.orderBy(F.desc("cfr_pct"))

    return (
        snapshot
        .withColumn("rank_total_cases", F.rank().over(rank_cases))
        .withColumn("rank_cfr",         F.rank().over(rank_cfr))
        .orderBy("rank_total_cases")
    )


def assign_wave_labels(states_df: DataFrame) -> DataFrame:
    """
    Clasifica cada registro en una ola epidémica usando WAVE_DEFINITIONS de settings.

    Las fronteras calendario se centralizan en settings.py para que cualquier
    ajuste epidemiológico se propague automáticamente a todo el pipeline sin
    modificar lógica de negocio.
    """
    log.info("Asignando etiquetas de ola epidémica...")

    # Construir la expresión when/otherwise dinámicamente desde WAVE_DEFINITIONS
    expr = None
    for start, end, label in WAVE_DEFINITIONS:
        condition = (F.col("date") >= start) & (F.col("date") < end)
        expr = F.when(condition, label) if expr is None else expr.when(condition, label)

    expr = expr.otherwise(WAVE_LAST_LABEL)
    return states_df.withColumn("wave", expr)


def describe_wave_summary(states_df: DataFrame) -> DataFrame:
    """
    Estadísticas agregadas por ola epidémica.

    Para cada ola: casos totales, muertes totales, duración en días,
    peak del promedio móvil y CFR promedio. Permite comparar la magnitud
    y mortalidad relativa de cada ola — documentando la reducción de CFR
    con vacunación y la mayor infectividad de Omicron.
    """
    log.info("Calculando resumen por ola...")
    df_with_waves = assign_wave_labels(states_df)
    return (
        df_with_waves
        .groupBy("wave")
        .agg(
            F.sum("daily_cases").alias("wave_total_cases"),
            F.sum("daily_deaths").alias("wave_total_deaths"),
            F.countDistinct("date").alias("wave_duration_days"),
            F.round(F.max("rolling_avg_7d_cases"), 1).alias("wave_peak_rolling_avg"),
            F.round(F.avg("cfr_pct"), 4).alias("wave_avg_cfr_pct"),
        )
        .orderBy("wave")
    )


def describe_weekday_effect(states_df: DataFrame) -> DataFrame:
    """
    Cuantifica el artefacto de reporte de fin de semana.

    Esperado: Lunes (day_of_week=2 en Spark) tiene media y desviación
    estándar significativamente mayores que otros días. Este patrón
    es un artifact de infraestructura de reporte, no de transmisión real.
    Su cuantificación justifica el uso de `day_of_week` como feature en ML.
    """
    log.info("Analizando efecto día de la semana en el reporte...")
    day_labels = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
    mapping_expr = F.create_map([
        F.lit(k) for pair in day_labels.items() for k in pair
    ])
    return (
        states_df
        .groupBy("day_of_week")
        .agg(
            F.round(F.avg("daily_cases"),    1).alias("avg_daily_cases"),
            F.round(F.stddev("daily_cases"), 1).alias("std_daily_cases"),
            F.round(F.avg("daily_deaths"),   2).alias("avg_daily_deaths"),
        )
        .withColumn("day_name", mapping_expr[F.col("day_of_week")])
        .orderBy("day_of_week")
    )


def describe_county_top_n(counties_df: DataFrame, n: int = 20) -> DataFrame:
    """
    Top-N condados por total de casos acumulados a nivel nacional.

    Excluye filas con unknown_county=True (sin FIPS conocido).
    Calcula CFR por condado como muertes_totales / casos_totales para
    revelar heterogeneidad en la mortalidad entre jurisdicciones.
    """
    log.info(f"Calculando top {n} condados por casos totales...")
    return (
        counties_df
        .filter(F.col("unknown_county") == False)
        .groupBy("county", "state", "fips")
        .agg(
            F.sum("daily_cases").alias("total_cases"),
            F.sum("daily_deaths").alias("total_deaths"),
        )
        .withColumn(
            "county_cfr_pct",
            F.when(F.col("total_cases") > 0,
                   F.round(F.col("total_deaths") / F.col("total_cases") * 100, 4))
             .otherwise(F.lit(None).cast(DoubleType()))
        )
        .orderBy(F.desc("total_cases"))
        .limit(n)
    )


def describe_mask_vs_peak(counties_df: DataFrame) -> DataFrame:
    """
    Tabla de correlación entre cumplimiento de mascarillas y pico de casos.

    Compara el score de mascarillas (encuesta Julio 2020) contra el pico
    del promedio móvil durante la Wave 2 (Julio-Sep 2020), que inmediatamente
    sigue a la encuesta. Este es el par temporal más defensible metodológicamente:
    olas posteriores están confundidas por vacunación y cambio de variante.

    Solo se calcula si la columna mask_compliance_score existe en el DataFrame
    (es decir, si el join de mask_use se ejecutó en transform_counties).
    """
    if "mask_compliance_score" not in counties_df.columns:
        log.warning("Columna 'mask_compliance_score' no disponible. Saltando análisis de mascarillas.")
        return counties_df.limit(0)

    log.info("Calculando correlación mask_compliance vs pico de casos (Wave 2)...")
    return (
        counties_df
        .filter(
            (F.col("date") >= "2020-07-01") &
            (F.col("date") < "2020-10-01") &
            F.col("mask_compliance_score").isNotNull() &
            (F.col("unknown_county") == False)
        )
        .groupBy("county", "state", "fips", "mask_compliance_score")
        .agg(
            F.round(F.max("rolling_avg_7d_cases"), 2).alias("wave2_peak_rolling_avg"),
            F.sum("daily_cases").alias("wave2_total_cases"),
        )
        .filter(F.col("wave2_peak_rolling_avg").isNotNull())
        .orderBy(F.desc("wave2_peak_rolling_avg"))
    )


# =============================================================================
# SECCIÓN 2: JOBS EXPLÍCITOS DE MAPREDUCÉ (API RDD)
# =============================================================================

def mr_total_cases_by_state(states_df: DataFrame) -> List[Tuple]:
    """
    MapReduce Job 1 — Casos y muertes totales por estado.

    Map:    emit (state, (daily_cases, daily_deaths)) por cada fila
    Reduce: sum de ambas métricas por clave 'state'
    Output: lista ordenada por total de casos descendente

    Este es el patrón canónico de word-count aplicado a agregación geográfica.
    Cada estado es una "palabra"; casos y muertes son la "frecuencia".
    """
    log.info("[MR-1] Total de casos y muertes por estado...")

    rdd = (
        states_df
        .select("state", "daily_cases", "daily_deaths")
        .na.fill(0, ["daily_cases", "daily_deaths"])
        .rdd
        # Map: clave = estado, valor = tupla (casos, muertes)
        .map(lambda row: (
            row["state"],
            (int(row["daily_cases"]), int(row["daily_deaths"]))
        ))
    )

    # Reduce: suma elemento a elemento de la tupla por clave
    reduced = rdd.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )

    result = sorted(
        [(state, vals[0], vals[1]) for state, vals in reduced.collect()],
        key=lambda x: x[1],
        reverse=True,
    )
    log.info(f"[MR-1] Completado: {len(result)} estados procesados.")
    return result


def mr_peak_day_by_state(states_df: DataFrame) -> List[Tuple]:
    """
    MapReduce Job 2 — Día de máximo de casos diarios por estado.

    Map:    emit (state, (date_str, daily_cases)) por cada fila
    Reduce: retain el par (date, cases) con mayor número de casos

    Identifica la fecha exacta del pico epidémico por estado — input
    clave para el análisis de velocidad de propagación geográfica.
    Las fechas se convierten a string para serializabilidad en el RDD.
    """
    log.info("[MR-2] Día pico de casos por estado...")

    rdd = (
        states_df
        .select("state", "date", "daily_cases")
        .na.fill({"daily_cases": 0})
        .rdd
        .map(lambda row: (
            row["state"],
            (str(row["date"]), int(row["daily_cases"]))
        ))
    )

    # Reduce: conservar la tupla con mayor daily_cases
    reduced = rdd.reduceByKey(
        lambda a, b: a if a[1] >= b[1] else b
    )

    result = sorted(
        [(state, date, cases) for state, (date, cases) in reduced.collect()],
        key=lambda x: x[2],
        reverse=True,
    )
    log.info(f"[MR-2] Completado: pico identificado para {len(result)} estados.")
    return result


def mr_county_ranking_national(counties_df: DataFrame, top_n: int = 50) -> List[Tuple]:
    """
    MapReduce Job 3 — Ranking nacional de condados por total de casos.

    Map:    emit ((county, state, fips), daily_cases)
    Reduce: sum de daily_cases por condado (clave compuesta para unicidad)
    Sort:   ordenamiento global descendente en el driver (post-collect)

    El ordenamiento global ocurre en el driver (no distribuido) porque el
    resultado final es pequeño: ~3,200 condados → top_n filas. Distribuir
    un sort global de datos tan pequeños tiene overhead mayor que el cómputo.
    """
    log.info(f"[MR-3] Ranking nacional de condados (top {top_n})...")

    rdd = (
        counties_df
        .filter(F.col("unknown_county") == False)
        .select("county", "state", "fips", "daily_cases")
        .na.fill({"daily_cases": 0})
        .rdd
        .map(lambda row: (
            (row["county"], row["state"], row["fips"]),
            int(row["daily_cases"])
        ))
    )

    reduced = rdd.reduceByKey(lambda a, b: a + b)

    result = sorted(
        [((county, state, fips), total) for (county, state, fips), total in reduced.collect()],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    log.info(f"[MR-3] Completado: top {top_n} condados clasificados.")
    return result


def mr_reporting_gap_detection(states_df: DataFrame) -> List[Tuple]:
    """
    MapReduce Job 4 — Detección de gaps en la cobertura temporal por estado.

    Map:    emit (state, 1) por cada fila (presencia de un reporte)
    Reduce: COUNT de días reportados por estado
    Compare: contra los 1,157 días esperados (21-Ene-2020 → 23-Mar-2023)

    Los estados con significativamente menos días reportados tienen gaps
    de cobertura que deben documentarse como limitación del análisis.
    Este job es un control de calidad de datos, no un análisis epidemiológico.
    """
    log.info("[MR-4] Detección de gaps de reporte por estado...")
    # EXPECTED_DAYS centralizado en settings.py: 21-Ene-2020 → 23-Mar-2023

    rdd = (
        states_df
        .select("state", "date")
        .rdd
        # Map: cada fila reportada contribuye 1 al contador de su estado
        .map(lambda row: (row["state"], 1))
    )

    # Reduce: contar días reportados
    reduced = rdd.reduceByKey(lambda a, b: a + b)

    result = sorted(
        [
            (state, count, EXPECTED_DAYS - count, round(count / EXPECTED_DAYS * 100, 1))
            for state, count in reduced.collect()
        ],
        key=lambda x: x[2],  # Ordenar por gap descendente
        reverse=True,
    )

    gap_count = sum(1 for _, _, gap, _ in result if gap > 0)
    log.info(f"[MR-4] Completado: {gap_count} estados con gaps de cobertura.")
    return result


def mr_mask_compliance_histogram(counties_df: DataFrame) -> Dict[str, int]:
    """
    MapReduce Job 5 — Histograma de deciles de cumplimiento de mascarillas.

    Map:    emit (decile_bucket, 1) para cada condado con score válido
    Reduce: COUNT de condados por decile_bucket

    El patrón histograma MapReduce mapea cada valor a su bucket, luego
    cuenta la frecuencia por bucket. Demuestra el paradigma para análisis
    de distribuciones cuando el dominio del valor es conocido de antemano.
    """
    if "mask_compliance_score" not in counties_df.columns:
        log.warning("[MR-5] Columna 'mask_compliance_score' no disponible.")
        return {}

    log.info("[MR-5] Histograma de cumplimiento de mascarillas...")

    def to_decile_bucket(score) -> str:
        """Mapea un score [0,1] a su etiqueta de decil."""
        if score is None:
            return "unknown"
        idx = min(int(score * 10), 9)
        return f"{idx/10:.1f}-{(idx+1)/10:.1f}"

    rdd = (
        counties_df
        # Un registro por condado: filtrar por fecha única
        .filter(
            (F.col("date") == "2020-07-01") &
            F.col("mask_compliance_score").isNotNull() &
            (F.col("unknown_county") == False)
        )
        .select("fips", "mask_compliance_score")
        .rdd
        # Map: bucket → 1
        .map(lambda row: (to_decile_bucket(row["mask_compliance_score"]), 1))
    )

    # Reduce: contar condados por bucket
    histogram = dict(sorted(rdd.reduceByKey(lambda a, b: a + b).collect()))
    log.info(f"[MR-5] Completado: {histogram}")
    return histogram


# =============================================================================
# SECCIÓN 3: PIPELINE PREDICTIVO (MLlib)
# =============================================================================

def build_feature_set(states_df: DataFrame) -> DataFrame:
    """
    Construye el conjunto de features para forecasting de series temporales.

    Estrategia de ingeniería de features:

    Lag features:
      - lag_7:  período de incubación medio del COVID-19 (~5-7 días)
      - lag_14: ciclo completo síntoma-test-reporte
      - lag_21: ola de transmisión secundaria

    Codificación cíclica del mes (sin/cos):
      Convierte el mes (1-12) en dos coordenadas en el círculo unitario.
      Sin esta transformación, un modelo lineal ve Diciembre (12) y Enero (1)
      como extremos opuestos en lugar de meses adyacentes — error que
      destruye la captura de estacionalidad invernal.
      month_sin = sin(2π × month / 12)
      month_cos = cos(2π × month / 12)

    days_since_start:
      Captura la tendencia macro de la pandemia. Sin este feature, el modelo
      no puede distinguir que Wave 1 de 2020 y Wave 3 de 2020-21 a niveles
      similares de casos son epidemiológicamente distintos.

    Target:
      cases_7d_ahead = daily_cases del día t+7 (lead de 7 días).
      Predecir 7 días adelante es el horizonte operativo estándar para
      planificación hospitalaria.
    """
    log.info("Construyendo feature set para ML...")

    lag_window = Window.partitionBy("state").orderBy("date")

    df = states_df.select(
        "date", "state", "fips",
        "daily_cases", "daily_deaths",
        "rolling_avg_7d_cases", "growth_rate_7d_pct",
        "cfr_pct", "month", "day_of_week", "is_weekend",
        "week_of_year", "year",
    )

    # Features de lag temporal
    for lag in [7, 14, 21]:
        df = (
            df
            .withColumn(f"cases_lag_{lag}",
                        F.lag("daily_cases", lag).over(lag_window))
            .withColumn(f"deaths_lag_{lag}",
                        F.lag("daily_deaths", lag).over(lag_window))
            .withColumn(f"rolling_avg_lag_{lag}",
                        F.lag("rolling_avg_7d_cases", lag).over(lag_window))
        )

    # Codificación cíclica del mes
    TWO_PI = 2.0 * math.pi
    df = (
        df
        .withColumn("month_sin",
                    F.round(F.sin(F.col("month") * (TWO_PI / 12.0)), 6))
        .withColumn("month_cos",
                    F.round(F.cos(F.col("month") * (TWO_PI / 12.0)), 6))
    )

    # Tendencia macro de la pandemia
    df = df.withColumn(
        "days_since_start",
        F.datediff(F.col("date"), F.lit("2020-01-21")).cast(DoubleType())
    )

    # Variable objetivo: casos diarios 7 días en el futuro
    df = df.withColumn(
        "target_cases_7d",
        F.lead("daily_cases", ML_CONFIG["forecast_horizon_days"]).over(lag_window)
    )

    # Eliminar filas donde algún lag obligatorio sea nulo
    # (primeras N filas de cada estado y últimas 7 del dataset)
    required = [
        "cases_lag_7", "cases_lag_14", "cases_lag_21",
        "rolling_avg_7d_cases", "target_cases_7d",
    ]
    df = df.dropna(subset=required)

    count = df.count()
    log.info(f"Feature set construido: {count:,} filas.")
    return df


def temporal_train_test_split(
    df: DataFrame,
    train_end: str = None,
    val_end: str = None,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Split cronológico estricto: train / validation / test.

    PROHIBIDO el split aleatorio en series temporales: permite que el modelo
    vea datos futuros durante el entrenamiento (data leakage), produciendo
    métricas de evaluación artificialmente infladas y modelos inútiles en
    producción.

    Rationale de las fronteras:
      Train (Ene 2020 – Oct 2021): cubre Waves 1-5; el modelo aprende la
        dinámica base pre-Omicron con distintas variantes.
      Val (Nov 2021 – Jun 2022): cubre la ola Omicron; usado para ajuste
        de hiperparámetros sin contaminar el test set.
      Test (Jul 2022 – Mar 2023): datos completamente no vistos durante el
        entrenamiento y validación; métrica de performance reportada.
    """
    train_end = train_end or ML_CONFIG["train_end_date"]
    val_end   = val_end   or ML_CONFIG["val_end_date"]

    train = df.filter(F.col("date") <= train_end)
    val   = df.filter((F.col("date") > train_end) & (F.col("date") <= val_end))
    test  = df.filter(F.col("date") > val_end)

    log.info(
        f"Split temporal — "
        f"Train: {train.count():,} | Val: {val.count():,} | Test: {test.count():,}"
    )
    return train, val, test


def build_gbt_pipeline(feature_cols: List[str]) -> Pipeline:
    """
    Construye el Pipeline MLlib para regresión con GBT.

    Stages del Pipeline:
      1. StringIndexer: 'state' (string) → índice numérico ordinal
      2. OneHotEncoder: índice → vector binario (evita que el modelo asuma
         ordenamiento ordinal entre estados — California no es "mayor" que Alabama)
      3. VectorAssembler: combina todos los features numéricos + vector de estado
         en un único DenseVector requerido por los algoritmos de MLlib
      4. StandardScaler: normaliza a media=0, std=1. Requerido para la convergencia
         de regresión lineal y beneficioso para GBT (estandariza las escalas)
      5. GBTRegressor: árboles potenciados por gradiente — captura no-linealidades
         e interacciones sin expansión polinomial manual

    El Pipeline API garantiza que las mismas transformaciones del training
    se aplican idénticamente a val y test — previene leakage en preprocesamiento.
    """
    state_indexer = StringIndexer(
        inputCol="state", outputCol="state_idx", handleInvalid="keep"
    )
    state_encoder = OneHotEncoder(
        inputCols=["state_idx"], outputCols=["state_vec"]
    )

    numeric_features = [c for c in feature_cols if c != "state"]
    assembler = VectorAssembler(
        inputCols=numeric_features + ["state_vec"],
        outputCol="raw_features",
        handleInvalid="skip",
    )

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="target_cases_7d",
        maxIter=ML_CONFIG["gbt_max_iter"],
        maxDepth=ML_CONFIG["gbt_max_depth"],
        stepSize=ML_CONFIG["gbt_step_size"],
        subsamplingRate=ML_CONFIG["gbt_subsampling_rate"],
        seed=ML_CONFIG["random_seed"],
    )

    return Pipeline(stages=[state_indexer, state_encoder, assembler, scaler, gbt])


def train_and_evaluate(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    feature_cols: List[str],
) -> Dict:
    """
    Entrena el modelo GBT y reporta métricas de evaluación.

    Métricas:
      RMSE: penaliza errores grandes cuadráticamente — crítico en epidemiología
            donde subestimar un pico tiene consecuencias graves en planificación.
      MAE:  error medio en unidades originales (casos/día) — interpretable
            directamente por tomadores de decisión.
      R²:   proporción de varianza explicada — permite comparar modelos.

    El modelo se evalúa en validation para decisiones de hyperparámetros
    y en test (held-out) para el reporte final de performance.
    """
    log.info("Entrenando modelo GBT...")

    pipeline = build_gbt_pipeline(feature_cols)
    model = pipeline.fit(train_df)

    metrics_by_split = {}
    for split_name, split_df in [("validation", val_df), ("test", test_df)]:
        preds = model.transform(split_df)

        metrics = {}
        for metric in ["rmse", "mae", "r2"]:
            evaluator = RegressionEvaluator(
                labelCol="target_cases_7d",
                predictionCol="prediction",
                metricName=metric,
            )
            metrics[metric] = round(evaluator.evaluate(preds), 4)

        metrics_by_split[split_name] = metrics
        log.info(
            f"[{split_name.upper()}] "
            f"RMSE={metrics['rmse']:,.1f} | "
            f"MAE={metrics['mae']:,.1f} | "
            f"R²={metrics['r2']:.4f}"
        )

    return {"model": model, "metrics": metrics_by_split}


def run_state_clustering(states_df: DataFrame, k: int = None) -> DataFrame:
    """
    Clustering de estados por perfil epidémico con K-Means (MLlib).

    Vector de features por estado (agregado sobre todo el período):
      - total_cases, total_deaths: magnitud absoluta de la carga
      - peak_rolling_avg: severidad del peor momento
      - mean_cfr: mortalidad media sobre toda la pandemia
      - mean_growth_rate: velocidad promedio de propagación

    k=6 corresponde a las 6 olas identificadas — prior epidemiológico natural
    sobre cuántos perfiles distintos esperar. Debe validarse con el método
    del codo (inertia vs k) en la capa de visualización.

    StandardScaler es imprescindible: total_cases tiene magnitudes de millones
    mientras mean_cfr está en el rango [0, 10]. Sin normalización, K-Means
    ignoraría las métricas de escala pequeña completamente.
    """
    k = k or ML_CONFIG["kmeans_k"]
    log.info(f"Ejecutando clustering K-Means sobre estados (k={k})...")

    feature_cols = [
        "total_cases", "total_deaths",
        "peak_rolling_avg", "mean_cfr", "mean_growth_rate",
    ]

    state_profiles = (
        states_df
        .groupBy("state")
        .agg(
            F.sum("daily_cases").alias("total_cases"),
            F.sum("daily_deaths").alias("total_deaths"),
            F.round(F.max("rolling_avg_7d_cases"), 1).alias("peak_rolling_avg"),
            F.round(F.avg("cfr_pct"), 4).alias("mean_cfr"),
            F.round(F.avg("growth_rate_7d_pct"), 2).alias("mean_growth_rate"),
        )
        .na.fill(0)
    )

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler    = StandardScaler(inputCol="raw_features", outputCol="features",
                               withMean=True, withStd=True)
    kmeans    = KMeans(
        featuresCol="features", predictionCol="cluster",
        k=k, seed=ML_CONFIG["random_seed"], maxIter=50,
    )

    model     = Pipeline(stages=[assembler, scaler, kmeans]).fit(state_profiles)
    clustered = (
        model.transform(state_profiles)
        .select("state", "cluster", *feature_cols)
        .orderBy("cluster", "state")
    )

    log.info("Distribución de clusters:")
    clustered.groupBy("cluster").count().orderBy("cluster").show(truncate=False)
    return clustered


# =============================================================================
# SECCIÓN 4: ESCRITURA GOLD Y ORQUESTADOR
# =============================================================================

def write_gold(df: DataFrame, name: str, partition_cols: list = None) -> None:
    """
    Persiste outputs analíticos en la Gold zone.

    Gold zone: tablas listas para consumo final (visualizaciones, reportes,
    APIs). Son el producto terminal del pipeline — no se transforman más.
    """
    output_path = str(Path(PROCESSED_DIR) / "gold" / name)
    writer = df.write.mode("overwrite").format("parquet")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.save(output_path)
    log.info(f"[gold/{name}] Escrito en {output_path}")


def run_analytics(transformed: dict) -> dict:
    """
    Orquesta el stage analítico completo.

    Orden de ejecución:
      1. Descriptivo (agregaciones rápidas sobre DataFrames cacheados)
      2. MapReduce  (jobs RDD explícitos — resultados colectados al driver)
      3. ML pipeline (cómputo más pesado — ejecutado al final)

    Los resultados de MapReduce son listas Python colectadas al driver;
    son pequeñas (una entrada por estado/condado) y se loguean directamente.
    Los outputs de DataFrame se escriben a Gold zone para visualización.
    """
    national_df = transformed["national"]
    states_df   = transformed["states"]
    counties_df = transformed["counties"]

    log.info("=== ANALYTICS: DESCRIPTIVO ===")

    national_summary = describe_national_timeline(national_df)
    state_rankings   = describe_state_rankings(states_df)
    wave_summary     = describe_wave_summary(states_df)
    weekday_effect   = describe_weekday_effect(states_df)
    county_top20     = describe_county_top_n(counties_df, n=20)
    mask_vs_peak     = describe_mask_vs_peak(counties_df)

    write_gold(national_summary, "national_monthly_summary")
    write_gold(state_rankings,   "state_rankings")
    write_gold(wave_summary,     "wave_summary")
    write_gold(weekday_effect,   "weekday_reporting_effect")
    write_gold(county_top20,     "county_top_20")
    if mask_vs_peak.count() > 0:
        write_gold(mask_vs_peak, "mask_compliance_vs_peak")

    log.info("=== ANALYTICS: MAPREDUCÉ ===")

    mr1 = mr_total_cases_by_state(states_df)
    log.info(f"[MR-1] Top 5 estados: {mr1[:5]}")

    mr2 = mr_peak_day_by_state(states_df)
    log.info(f"[MR-2] Top 5 picos: {mr2[:5]}")

    mr3 = mr_county_ranking_national(counties_df, top_n=50)
    log.info(f"[MR-3] Top 5 condados: {mr3[:5]}")

    mr4 = mr_reporting_gap_detection(states_df)
    gaps = sum(1 for _, _, gap, _ in mr4 if gap > 0)
    log.info(f"[MR-4] Estados con gaps: {gaps}")

    mr5 = mr_mask_compliance_histogram(counties_df)

    log.info("=== ANALYTICS: PIPELINE ML ===")

    feature_df = build_feature_set(states_df)
    write_gold(feature_df, "ml_feature_set", partition_cols=["state"])

    train_df, val_df, test_df = temporal_train_test_split(feature_df)

    feature_cols = [
        "state",
        "cases_lag_7", "cases_lag_14", "cases_lag_21",
        "deaths_lag_7",
        "rolling_avg_7d_cases", "rolling_avg_lag_7", "rolling_avg_lag_14",
        "growth_rate_7d_pct",
        "cfr_pct",
        "month_sin", "month_cos",
        "day_of_week", "is_weekend",
        "days_since_start",
    ]

    ml_output = train_and_evaluate(train_df, val_df, test_df, feature_cols)

    test_preds = (
        ml_output["model"]
        .transform(test_df)
        .select("date", "state", "target_cases_7d", "prediction")
    )
    write_gold(test_preds, "ml_predictions_test")

    cluster_df = run_state_clustering(states_df)
    write_gold(cluster_df, "state_clusters")

    log.info("Stage analítico completado.")

    return {
        "descriptive": {
            "national_summary": national_summary,
            "state_rankings":   state_rankings,
            "wave_summary":     wave_summary,
            "weekday_effect":   weekday_effect,
            "county_top20":     county_top20,
        },
        "mapreduce": {
            "mr1_state_totals":       mr1,
            "mr2_peak_days":          mr2,
            "mr3_county_ranking":     mr3,
            "mr4_reporting_gaps":     mr4,
            "mr5_mask_histogram":     mr5,
        },
        "ml": {
            "model":   ml_output["model"],
            "metrics": ml_output["metrics"],
        },
    }
