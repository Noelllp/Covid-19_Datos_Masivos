# COVID-19 Big Data Pipeline

Pipeline de análisis de datos a gran escala sobre el dataset COVID-19 del New York Times (Enero 2020 – Marzo 2023), implementado con **Apache Spark 3.5**, **MLlib** y el paradigma **MapReduce**.

---

## Objetivo

Análisis descriptivo y predictivo del comportamiento del COVID-19 en Estados Unidos durante la pandemia 2020–2023:

- **Descriptivo:** Identificar cuáles estados fueron más golpeados, cuándo ocurrieron los picos, cómo evolucionó la tasa de mortalidad, y comparativas regionales.
- **Predictivo:** Dado el comportamiento histórico de un estado, proyectar casos o muertes para semanas/meses siguientes.

---

## Descripción general

El proyecto implementa un pipeline ETL completo siguiendo la **arquitectura Medallion** (Bronze → Silver → Gold), procesando más de **3.5 millones de registros** de casos y muertes a nivel nacional, estatal y de condado.

El pipeline incluye:
- Ingesta y validación de datos con esquemas explícitos
- Transformaciones con Window functions distribuidas
- Ingeniería de 28 variables para modelado predictivo
- 3 jobs explícitos de MapReduce sobre la API RDD de Spark
- Comparación de 4 modelos predictivos (Regresión Lineal, Polinomial, ARIMA, Prophet)
- Clustering epidémico de estados con K-Means (k=6)
- 12 visualizaciones automáticas en PNG

---

## Requisitos del sistema

| Componente | Versión requerida |
|---|---|
| Python | 3.10+ |
| Java (JDK) | 11, 17 o 21 |
| Sistema operativo | Windows, macOS o Linux |

> **Windows:** Se requiere `winutils.exe` y `hadoop.dll`. Ver sección de configuración abajo.

---

## Estructura del proyecto

```
Analisis_Covid-19_Datos_Masivos-main/
│
├── main.py                  # Pipeline principal (Bronze → Gold + visualizaciones)
├── mejoras.py               # Visualizaciones adicionales (ejecutar tras main.py)
├── forecast_futuro.py       # Pronóstico independiente sin Spark (años futuros)
├── requirements.txt
│
├── config/
│   └── settings.py          # Configuración central: rutas, parámetros ML, olas epidémicas
│
├── src/
│   ├── ingest.py            # Stage 1: CSVs → Bronze zone (Parquet)
│   ├── transform.py         # Stage 2: deltas diarios, rolling avg, CFR → Silver zone
│   ├── features.py          # Stage 3A: ingeniería de 28 variables para ML
│   ├── models.py            # Stage 3B: Regresión Lineal, Polinomial, ARIMA, Prophet
│   ├── mapreduce_jobs.py    # Stage 3C: 3 jobs MapReduce explícitos (API RDD)
│   ├── analytics.py         # Stage 3D: estadísticas descriptivas + K-Means → Gold zone
│   └── visualizations.py   # Stage 4: visualizaciones PNG con matplotlib
│
├── covid-19-data/           # Dataset fuente del NYT (debe clonarse aparte — no incluido)
│   ├── us.csv
│   ├── us-states.csv
│   ├── us-counties-20XX.csv
│   └── ...
│
└── data/
    ├── figures/             # 12 gráficas PNG generadas por el pipeline
    └── processed/           # Bronze / Silver / Gold en Parquet (generado, no versionado)
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/covid-bigdata-project.git
cd covid-bigdata-project
```

### 2. Obtener los datos del NYT

```bash
git clone https://github.com/nytimes/covid-19-data.git
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar Java

```bash
java -version
```

### 5. Solo en Windows: configurar winutils

Spark en Windows requiere `winutils.exe` y `hadoop.dll`:

1. Descargar desde https://github.com/cdarlint/winutils (carpeta `hadoop-3.x.x/bin/`)
2. Copiar a `C:\hadoop\bin\`
3. Definir la variable de entorno antes de ejecutar:

```powershell
$env:HADOOP_HOME = "C:\hadoop"
```

---

## Ejecución

### Pipeline principal

```bash
python main.py
```

Genera en `data/processed/`: zonas Bronze, Silver y Gold (Parquet).
Genera en `data/figures/`: `viz1`, `viz4`, `viz9`, `viz14`, `viz15`.

### Visualizaciones adicionales (ejecutar después del pipeline)

```bash
python mejoras.py
```

Genera en `data/figures/`: `viz2`, `viz3`, `viz7`, `viz10`, `viz11`, `viz12`, `viz13`, `viz15`.

### Pronóstico de años futuros (sin necesidad de Spark)

```bash
# Todos los estados, 365 días al futuro
python forecast_futuro.py --dias 365

# Estados específicos
python forecast_futuro.py --dias 180 --estados California Texas Florida

# 2 años al futuro
python forecast_futuro.py --dias 730
```

Genera CSV y PNGs individuales en `data/forecast_futuro/`.

---

## Arquitectura del pipeline

```
CSVs NYT  →  [Stage 1: ingest.py]  →  Bronze (Parquet)
                                            ↓
                      [Stage 2: transform.py]  →  Silver (Parquet)
                                                        ↓
              ┌─────────────────────────────────────────┤
              ↓                  ↓                ↓                ↓
       features.py          models.py     mapreduce_jobs.py   analytics.py
       (28 variables)   (4 modelos ML)   (3 jobs RDD)     (descriptivo + KMeans)
              └─────────────────────────────────────────┤
                                                        ↓
                           [Stage 4: visualizations.py]
                                   Gold zone (Parquet)

+ mejoras.py        →  visualizaciones adicionales / corregidas
+ forecast_futuro.py  →  pronóstico autónomo (sin Spark)
```

---

## Datos fuente

Dataset público del **New York Times** — seguimiento diario de COVID-19 en EE.UU.:

| Archivo | Descripción | Filas aprox. |
|---|---|---|
| `us.csv` | Totales nacionales diarios | 1,158 |
| `us-states.csv` | Por estado, diario | 61,942 |
| `us-counties-20XX.csv` | Por condado, por año | 3,525,161 |
| `rolling-averages/us-states.csv` | Promedios móviles precomputados | 61,942 |
| `rolling-averages/anomalies.csv` | Registro de anomalías del NYT | 2,511 |
| `mask-use/mask-use-by-county.csv` | Encuesta de uso de mascarillas (Jul 2020) | 3,142 |

---

## Módulos principales

### `ingest.py` — Bronze zone
- Lee CSVs con esquemas explícitos (evita inferencia costosa)
- Consolida 4 archivos anuales de condados en un único DataFrame
- Valida calidad básica post-carga (conteo de filas, tasa de nulos)
- Persiste en Parquet particionado por estado y año

### `transform.py` — Silver zone
- Convierte totales acumulados → deltas diarios con Window `lag()`
- Calcula promedio móvil de 7 días, tasa de crecimiento y CFR
- Marca filas anómalas cruzando con el registro oficial del NYT
- Filtra excepciones geográficas (NYC, Kansas City, Joplin)
- Enriquece condados con la encuesta de uso de mascarillas

### `features.py` — Ingeniería de variables
28 variables organizadas en 8 familias para modelado predictivo:

| Familia | Variables |
|---|---|
| Señal bruta | `daily_cases`, `log_daily_cases` |
| Suavizado | `rolling_avg_7d`, `rolling_avg_14d`, `ma_ratio_7_14` |
| Momentum | `growth_rate_7d_pct`, `growth_rate_14d_pct`, `momentum_differential` |
| Aceleración | `acceleration_raw`, `acceleration_smooth`, `inflection_point` |
| Mortalidad | `cfr_rolling_14d`, `cfr_delta`, `cfr_lag_corrected_14d` |
| Posición temporal | `days_since_first_case_state`, `log_days_since_first_case` |
| Intensidad de ola | `wave_intensity`, `wave_intensity_zscore` |
| Volatilidad | `rolling_std_7d_cases`, `rolling_cv_7d` |

### `mapreduce_jobs.py` — Jobs MapReduce explícitos
Implementación del paradigma Map → Shuffle/Sort → Reduce sobre la API RDD de Spark:

- **Job 1:** Total de casos acumulados por estado
- **Job 2:** Total de muertes acumuladas por estado
- **Job 3:** Ranking nacional de estados por mortalidad (patrón Sort-by-Value)
- **Demo educativa:** comparación `reduceByKey` (con Combiner) vs `groupByKey` (sin Combiner)

### `models.py` — Modelos predictivos
Forecasting a 7 días con split cronológico estricto:

| Período | Fechas | Filas |
|---|---|---|
| Train | Ene 2020 – Oct 2021 | 87,753 |
| Validation | Nov 2021 – Jun 2022 | 71,412 |
| Test | Jul 2022 – Mar 2023 | 48,520 |

Modelos comparados:

| Modelo | Tipo | Implementación | MAE | RMSE | R² |
|---|---|---|---|---|---|
| Regresión Lineal (Ridge) | Global | MLlib Pipeline | 153.0 | 3902.7 | -1.065 |
| Regresión Polinomial (grado 2) | Global | MLlib + PolynomialExpansion | 479.6 | 3590.5 | -0.748 |
| ARIMA(7,1,1) | Por estado | statsmodels | 805.0 | 1409.1 | 0.410 |
| Prophet | Por estado | Meta Prophet | 3023.2 | 5437.7 | -7.790 |

> ARIMA obtiene el mejor R² (0.41) al modelar cada estado por separado. Las regresiones globales tienen R² negativo porque la heterogeneidad entre estados supera su capacidad de generalización con un único modelo.

### `analytics.py` — Gold zone
- Resumen mensual nacional, rankings de estados, análisis por ola epidémica
- Efecto día de la semana en el reporte (artefacto de infraestructura)
- Correlación uso de mascarillas vs pico de casos (Wave 2)
- Clustering K-Means (k=6) de estados por perfil epidémico

### `mejoras.py` — Visualizaciones adicionales
Script autónomo que lee la Silver zone y genera análisis complementarios sin necesidad de re-ejecutar el pipeline.

### `forecast_futuro.py` — Pronóstico autónomo
Script independiente que no requiere Spark. Lee directamente la Silver zone en Parquet,
entrena Prophet sobre la serie histórica completa de cada estado y proyecta N días al futuro.
Genera un CSV con las predicciones y PNGs individuales por estado.

---

## Visualizaciones generadas

| Archivo | Script | Descripción |
|---|---|---|
| `viz1_national_timeline.png` | main.py | Timeline nacional con bandas de olas epidémicas |
| `viz2_state_heatmap_fixed.png` | mejoras.py | Heatmap intensidad estado × mes (50 estados) |
| `viz3_wave_comparison_fixed.png` | mejoras.py | Casos, muertes y CFR por ola epidémica |
| `viz4_cfr_evolution.png` | main.py | Evolución de la tasa de mortalidad con hitos de vacunación |
| `viz7_model_comparison_fixed.png` | mejoras.py | Comparación MAE, RMSE y R² de los 4 modelos |
| `viz9_county_top20.png` | main.py | Top 20 condados por casos acumulados |
| `viz10_percapita.png` | mejoras.py | Casos y muertes por 100k habitantes por estado |
| `viz11_regional.png` | mejoras.py | Timeline por región censal de EE.UU. |
| `viz12_deaths_forecast.png` | mejoras.py | Pronóstico de muertes a 180 días (top 12 estados) |
| `viz13_weekly_forecast.png` | mejoras.py | Heatmap de intensidad semanal próximas 4 semanas |
| `viz14_clustering.png` | main.py | Clusters epidémicos de estados (K-Means k=6) |
| `viz15_forecast_resumen.png` | mejoras.py | Resumen comparativo de pronósticos por estado |

---

## Clustering epidémico (K-Means k=6)

Los estados se agruparon por perfil epidémico usando 7 características:
total de casos, total de muertes, pico máximo, CFR promedio, tasa de crecimiento media,
duración y número de olas.

| Cluster | Estados representativos | Perfil |
|---|---|---|
| A | AZ, GA, IN, LA, MA, MD, MS, NJ, NV, NM, OK, PA, SC, VA | Estados medianos con impacto moderado-alto |
| B | Missouri | Outlier por tasa de crecimiento extrema |
| C | California | Escala única (mayor población del país) |
| D | Florida, New York, Texas | Grandes estados con alto impacto absoluto |
| E | AK, CO, HI, ID, IA, KS, KY, ME, MN, MT, NE, NH, ND, OR, RI, SD, UT, VT, WV, WI, WY + territorios | Estados pequeños y territorios |
| F | AL, IL, MI, NC, OH, TN, WA | Cinturón industrial, impacto intermedio |

---

## Olas epidémicas definidas

| Ola | Período | Variante dominante |
|---|---|---|
| Wave 1 | Ene – Jun 2020 | Original |
| Wave 2 | Jul – Sep 2020 | Original (rebrote verano) |
| Wave 3 | Oct 2020 – Mar 2021 | Alpha |
| Wave 4 | Abr – Jun 2021 | Alpha/Beta |
| Wave 5 | Jul – Nov 2021 | Delta |
| Wave 6 | Dic 2021 – Mar 2022 | Omicron |
| Wave 7 | Abr 2022 – Mar 2023 | Subvariantes Omicron |

---

## Configuración de Spark

```python
"spark.sql.shuffle.partitions": "50"        # calibrado para ~3.5M filas local
"spark.sql.adaptive.enabled": "true"        # AQE: re-optimiza en tiempo de ejecución
"spark.driver.memory": "4g"
"spark.executor.memory": "4g"
"spark.sql.parquet.compression.codec": "snappy"
"spark.sql.execution.arrow.pyspark.enabled": "false"  # necesario con Java 21
```

Para cluster YARN: cambiar `shuffle.partitions` a 200–400 y ajustar memoria según los ejecutores disponibles.

---

## Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| Apache Spark 3.5 (PySpark) | Motor de procesamiento distribuido |
| PySpark MLlib | K-Means, Regresión Lineal/Polinomial |
| Spark RDD API | Jobs MapReduce explícitos |
| statsmodels | ARIMA(7,1,1) por estado |
| Prophet (Meta) | Forecasting con changepoints y estacionalidad |
| pandas / numpy | Procesamiento local post-collect |
| matplotlib | Generación de visualizaciones |
| Parquet + Snappy | Formato de almacenamiento columnar |

---

## Fuente de datos

New York Times COVID-19 Data:
> The New York Times. (2021). Coronavirus (Covid-19) Data in the United States. Retrieved from https://github.com/nytimes/covid-19-data
