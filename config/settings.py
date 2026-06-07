"""
settings.py — Configuración central del pipeline COVID-19.

Centralizar todos los parámetros aquí evita magic strings y números mágicos
dispersos entre módulos. Cualquier cambio de ruta, ventana temporal o umbral
se aplica desde un único lugar sin tocar lógica de negocio.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent
RAW_DATA_DIR   = BASE_DIR / "covid-19-data"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"

# ---------------------------------------------------------------------------
# Rutas absolutas a los archivos fuente (dataset NYT ya clonado localmente)
# ---------------------------------------------------------------------------
DATA_FILES = {
    "us_national":        str(RAW_DATA_DIR / "us.csv"),
    "us_states":          str(RAW_DATA_DIR / "us-states.csv"),
    "us_counties_2020":   str(RAW_DATA_DIR / "us-counties-2020.csv"),
    "us_counties_2021":   str(RAW_DATA_DIR / "us-counties-2021.csv"),
    "us_counties_2022":   str(RAW_DATA_DIR / "us-counties-2022.csv"),
    "us_counties_2023":   str(RAW_DATA_DIR / "us-counties-2023.csv"),
    "rolling_avg_states": str(RAW_DATA_DIR / "rolling-averages" / "us-states.csv"),
    "anomalies":          str(RAW_DATA_DIR / "rolling-averages" / "anomalies.csv"),
    "mask_use":           str(RAW_DATA_DIR / "mask-use" / "mask-use-by-county.csv"),
}

# ---------------------------------------------------------------------------
# Configuración de SparkSession
# ---------------------------------------------------------------------------
SPARK_CONFIG = {
    "spark.app.name":                               "COVID19-BigData-Pipeline",
    # 50 particiones calibradas para ~3.5M filas de condados en un entorno local.
    # En un cluster YARN con más ejecutores este valor subiría a 200-400.
    "spark.sql.shuffle.partitions":                 "50",
    # Adaptive Query Execution: re-optimiza joins y agregaciones en tiempo de
    # ejecución según estadísticas reales de particiones. Crítico para datos
    # asimétricos (condados muy poblados vs rurales).
    "spark.sql.adaptive.enabled":                   "true",
    "spark.sql.adaptive.coalescePartitions.enabled":"true",
    "spark.driver.memory":                          "4g",
    "spark.executor.memory":                        "4g",
    # Snappy: mejor balance compresión/velocidad de lectura para análisis iterativo.
    # Alternativa ZSTD si el storage es el cuello de botella.
    "spark.sql.parquet.compression.codec":          "snappy",
    # Reduce escrituras de shuffle a disco en operaciones de Window; beneficia
    # el cálculo masivo de lag() sobre 3.5M filas de condados.
    # Deshabilitado: Arrow + Java 21 produce UnsupportedOperationException en
    # sun.misc.Unsafe; applyInPandas funciona correctamente via pickle.
    "spark.sql.execution.arrow.pyspark.enabled":    "false",
    # En Windows el ejecutable Python se llama "python", no "python3".
    "spark.pyspark.python":                         "D:\\Programas\\python.exe",
    "spark.pyspark.driver.python":                  "D:\\Programas\\python.exe",
}

# ---------------------------------------------------------------------------
# Parámetros de análisis
# ---------------------------------------------------------------------------
ROLLING_WINDOW_DAYS = 7
COUNTY_YEARS        = [2020, 2021, 2022, 2023]

# Excepciones geográficas documentadas por el NYT.
# Estas entidades se reportan como jurisdicciones independientes solapando
# sus condados base → incluirlas en agregaciones a nivel condado produce
# doble conteo. Se excluyen del análisis de condados; pueden analizarse
# de forma independiente si se requiere.
GEOGRAPHIC_EXCEPTIONS = {
    "New York City",  # Cinco boroughs consolidados (no mapea a un único FIPS)
    "Kansas City",    # Reportado separado de sus 4 condados de Missouri
    "Joplin",         # Separado de Jasper/Newton desde el 25-Jun-2020
}

# ---------------------------------------------------------------------------
# Parámetros del pipeline de ML
# ---------------------------------------------------------------------------
ML_CONFIG = {
    "train_end_date": "2021-10-31",   # Incluye Waves 1-5
    "val_end_date":   "2022-06-30",   # Cubre ola Omicron para hyperparameter tuning
    # Test: Jul 2022 – Mar 2023 → datos completamente no vistos durante entrenamiento
    "forecast_horizon_days": 7,
    "gbt_max_iter":          100,
    "gbt_max_depth":         5,
    "gbt_step_size":         0.1,
    "gbt_subsampling_rate":  0.8,
    "kmeans_k":              6,       # Una por cada ola epidémica identificada
    "random_seed":           42,
}

# ---------------------------------------------------------------------------
# Definiciones de olas epidémicas
# ---------------------------------------------------------------------------
# Cada tupla: (inicio_inclusive, fin_exclusiva, etiqueta)
# Fronteras calendario derivadas de los picos documentados en vigilancia EE.UU.
# La séptima ola es el caso otherwise (todo lo que no cae en las seis anteriores).
WAVE_DEFINITIONS = [
    ("2020-01-21", "2020-07-01", "Wave 1 — Spring 2020"),
    ("2020-07-01", "2020-10-01", "Wave 2 — Summer 2020"),
    ("2020-10-01", "2021-04-01", "Wave 3 — Winter 2020-21"),
    ("2021-04-01", "2021-07-01", "Wave 4 — Spring 2021"),
    ("2021-07-01", "2021-12-01", "Wave 5 — Delta Summer 2021"),
    ("2021-12-01", "2022-04-01", "Wave 6 — Omicron Winter 2021-22"),
]
WAVE_LAST_LABEL = "Wave 7 — Omicron Subvariants 2022-23"

# ---------------------------------------------------------------------------
# Control de calidad de datos
# ---------------------------------------------------------------------------
# Días exactos en el dataset NYT: 21-Ene-2020 → 23-Mar-2023
EXPECTED_DAYS = 1157

# ---------------------------------------------------------------------------
# Parámetros de modelos de series temporales
# ---------------------------------------------------------------------------
# ARIMA(p,d,q): p=7 captura autocorrelación semanal, d=1 diferenciación
# estacionaria, q=1 suaviza el término de error.
ARIMA_ORDER = (7, 1, 1)

PROPHET_PARAMS = {
    "seasonality_mode":        "additive",
    "changepoint_prior_scale": 0.05,   # flexibilidad de changepoints (default documentado)
    "weekly_seasonality":      True,   # artefacto de reporte semanal
    "yearly_seasonality":      True,   # estacionalidad invernal
    "daily_seasonality":       False,
    "uncertainty_samples":     300,    # muestras para intervalos de predicción
}

# Regularización Ridge diferenciada:
#   LR usa λ=0.01  → colinealidad moderada entre lag features
#   Poly usa λ=0.1 → colinealidad severa adicional entre x, x², x³
LR_REG_PARAM   = 0.01
POLY_REG_PARAM = 0.1

# ---------------------------------------------------------------------------
# Directorios de salida
# ---------------------------------------------------------------------------
FIGURES_DIR = BASE_DIR / "data" / "figures"
