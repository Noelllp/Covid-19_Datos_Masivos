"""
features.py — Ingeniería de variables para modelado predictivo (COVID-19).

FUNDAMENTO CIENTÍFICO
=====================
El modelado predictivo de series temporales epidemiológicas requiere transformar
las señales crudas del dataset en representaciones que capturen la dinámica del
fenómeno a distintas escalas temporales y con distintos niveles de abstracción.

Este módulo implementa ocho familias de variables organizadas por su rol
estadístico en el proceso de modelado:

  1. Señal bruta          → daily_cases, log_daily_cases
  2. Suavizado temporal   → rolling_avg_7d (Silver), rolling_avg_14d
  3. Momentum             → growth_rate_7d (Silver), growth_rate_14d
  4. Aceleración          → acceleration_raw, acceleration_smooth
  5. Mortalidad           → cfr_pct (Silver), cfr_rolling_14d
  6. Posición temporal    → days_since_first_case_state
  7. Intensidad relativa  → wave_intensity, wave_intensity_zscore
  8. Volatilidad          → rolling_std_7d, rolling_cv_7d, ma_crossover

PRINCIPIO DE DISEÑO — LOOK-AHEAD BIAS:
  En todas las features que usan ventanas expandibles (ej. máximo histórico para
  wave_intensity) se usa Window.unboundedPreceding para garantizar que en el
  tiempo t solo se usan datos disponibles hasta t. Usar el máximo total del
  período haría que la feature en t=100 "conociera" el pico de t=500.
  Este sesgo (look-ahead bias) produce modelos que funcionan en backtesting
  pero fallan en producción.

REFERENCIA EPIDEMIOLÓGICA:
  Cori, A. et al. (2013). A new framework and software to estimate time-varying
  reproduction numbers during epidemics. Am. J. Epidemiol., 178(9), 1505-1512.
"""

import logging
import math
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DoubleType

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PROCESSED_DIR

log = logging.getLogger(__name__)

# =============================================================================
# Especificaciones de Window
# =============================================================================

def _state_window() -> Window:
    return Window.partitionBy("state").orderBy("date")

def _state_roll(days: int) -> Window:
    return (
        Window.partitionBy("state")
        .orderBy("date")
        .rowsBetween(-(days - 1), 0)
    )

def _state_expanding() -> Window:
    """Ventana expandible: desde el primer registro del estado hasta hoy."""
    return (
        Window.partitionBy("state")
        .orderBy("date")
        .rowsBetween(Window.unboundedPreceding, 0)
    )


# =============================================================================
# VARIABLE 1: CASOS NUEVOS DIARIOS (señal bruta y transformada)
# =============================================================================

def add_log_daily_cases(df: DataFrame) -> DataFrame:
    """
    Transforma logarítmica de los casos diarios: log(daily_cases + 1).

    UTILIDAD ESTADÍSTICA:
    ---------------------
    La distribución de daily_cases durante las fases de crecimiento exponencial
    sigue aproximadamente una distribución log-normal (heavy right tail, sesgada
    hacia la derecha). Los modelos lineales (regresión, ARIMA) asumen residuos
    con distribución normal; aplicarlos directamente a daily_cases viola este
    supuesto, produciendo residuos heteroscedásticos y predicciones sesgadas.

    La transformación log(x + 1) — constante +1 para manejar zeros sin errores —
    comprime el rango dinámico de [0, 900.000] a [0, 13.7] y aproxima la
    distribución a la normal.

    INTERPRETACIÓN DE COEFICIENTES:
    Si el modelo predice Δlog_cases = 0.1, la variación en casos es:
      e^0.1 ≈ 1.105 → aumento del 10.5% en casos.
    Esto facilita la interpretación en términos de tasas de crecimiento relativo.

    CUÁNDO USAR:
    - Modelos lineales (OLS, Ridge, Lasso): usar log_daily_cases como target.
    - Modelos no lineales (GBT, Random Forest): el log es menos crítico
      porque los árboles no asumen linealidad, pero sigue ayudando con la
      escala de los errores de entrenamiento.
    - Visualización: escala logarítmica para comparar fases tempranas y tardías.

    CUÁNDO NO USAR:
    - Cuando el objetivo es predecir el número absoluto de casos (revertir
      con exp(pred) - 1). El error en escala log no es el mismo que en escala
      original — RMSE en log ≠ RMSE en casos reales.
    """
    return df.withColumn(
        "log_daily_cases",
        F.round(F.log(F.col("daily_cases").cast(DoubleType()) + 1.0), 6)
    ).withColumn(
        "log_daily_deaths",
        F.round(F.log(F.col("daily_deaths").cast(DoubleType()) + 1.0), 6)
    )


# =============================================================================
# VARIABLE 2: MEDIA MÓVIL DE 7 DÍAS (ya en Silver, documentada aquí)
# =============================================================================
# rolling_avg_7d_cases ya existe en el DataFrame Silver de transform.py.
# Su utilidad se documenta aquí para completar el marco analítico.
#
# UTILIDAD ESTADÍSTICA:
# La media móvil de orden q es un filtro de paso bajo: atenúa las frecuencias
# por encima de 1/q ciclos por período. Con q=7 filtra exactamente la
# componente semanal (el artefacto de reporte de fin de semana) y el ruido
# de alta frecuencia, revelando la tendencia epidémica subyacente.
#
# PROPIEDAD ESTADÍSTICA CLAVE — Introducción de autocorrelación:
# Si {X_t} es ruido blanco (no correlacionado), MA_7(X_t) tiene autocorrelación
# significativa hasta el lag 6. Esto debe tenerse en cuenta al aplicar tests
# de Durbin-Watson o Ljung-Box para detectar autocorrelación residual.
#
# TRADE-OFF SESGO-VARIANZA TEMPORAL:
# Ventana 7d: baja varianza (suavizado), lag de 3 días (sesgo temporal).
# Ventana 14d: menor varianza aún, lag de 7 días (mayor sesgo).
# Para modelos en tiempo real: usar MA7.
# Para detección de tendencia a largo plazo: usar MA14.


# =============================================================================
# VARIABLE 3: MEDIA MÓVIL DE 14 DÍAS
# =============================================================================

def add_rolling_14d(df: DataFrame) -> DataFrame:
    """
    Media móvil de 14 días y desviación estándar de 7 días.

    UTILIDAD ESTADÍSTICA:
    ---------------------
    La MA14 actúa como línea de tendencia "lenta" respecto a MA7.
    La relación entre ambas medias móviles genera dos métricas derivadas
    de alto valor predictivo:

    1. MA_RATIO (MA7 / MA14):
       > 1.0 → MA7 está por encima de MA14 → tendencia alcista (expansión).
       < 1.0 → MA7 está por debajo de MA14 → tendencia bajista (contracción).
       Este cociente es un indicador de momentum continuo, más informativo que
       la tasa de crecimiento binaria (positiva/negativa).

    2. MA_CROSSOVER (booleano):
       El cruce de MA7 por encima de MA14 ("golden cross") señala el inicio
       de una nueva ola. El cruce por debajo ("death cross") señala el pico.
       Usado en epidemiología computacional como heurístico de detección de olas.
       Para un modelo de ML: feature binaria que activa en el período de
       transición entre fases epidémicas.

    JUSTIFICACIÓN DE LA VENTANA DE 14 DÍAS:
    Epidemiológicamente, 14 días cubre dos períodos de incubación completos
    del COVID-19 (5-7 días cada uno). Una MA14 refleja la tendencia de
    transmisión con dos generaciones virales de contexto — estándar OMS.

    RELACIÓN CON OTRAS VARIABLES:
    - MA7 y MA14 tienen correlación muy alta (ρ > 0.95): no incluir ambas
      directamente en modelos lineales (multicolinealidad). Usar MA_RATIO
      en su lugar, que captura la información diferencial.
    - En modelos de árbol (GBT, Random Forest): la colinealidad no es un
      problema para la estimación pero sí para la interpretación de importancia
      de features (ambas "comparten" la importancia).
    """
    roll_14 = _state_roll(14)

    return (
        df
        .withColumn(
            "rolling_avg_14d_cases",
            F.round(F.avg("daily_cases").over(roll_14), 2)
        )
        .withColumn(
            "rolling_avg_14d_deaths",
            F.round(F.avg("daily_deaths").over(roll_14), 2)
        )
        .withColumn(
            # Ratio MA7/MA14: > 1 = tendencia positiva, < 1 = negativa
            "ma_ratio_7_14",
            F.when(
                F.col("rolling_avg_14d_cases") > 0,
                F.round(
                    F.col("rolling_avg_7d_cases") / F.col("rolling_avg_14d_cases"),
                    4
                )
            ).otherwise(F.lit(None).cast(DoubleType()))
        )
        .withColumn(
            # Golden cross: MA7 > MA14 → ola en expansión
            "ma_crossover_bullish",
            F.when(
                F.col("rolling_avg_14d_cases").isNotNull() &
                F.col("rolling_avg_7d_cases").isNotNull(),
                F.col("rolling_avg_7d_cases") > F.col("rolling_avg_14d_cases")
            ).otherwise(F.lit(None))
        )
    )


# =============================================================================
# VARIABLE 4: TASA DE CRECIMIENTO (ya en Silver, extendida aquí)
# =============================================================================

def add_growth_rate_14d(df: DataFrame) -> DataFrame:
    """
    Tasa de crecimiento quincenal (14 días) y comparación con la semanal.

    UTILIDAD ESTADÍSTICA:
    ---------------------
    La tasa de crecimiento semanal (ya en Silver) captura el momentum a corto
    plazo. La tasa de 14 días captura el momentum a escala de dos períodos de
    incubación — más estable, menos reactiva a fluctuaciones de un día.

    RELACIÓN CON EL NÚMERO DE REPRODUCCIÓN R(t):
    La tasa de crecimiento diaria r está relacionada con R(t) mediante:
      R(t) ≈ (1 + r × T_generación)^(T_serial / T_generación)
    donde T_generación ≈ 5 días para COVID-19.

    Aunque esta función no computa R(t) directamente (requiere la distribución
    del intervalo serial, no disponible en este dataset), growth_rate_7d_pct > 0
    es condición necesaria y suficiente para R(t) > 1 cuando T_generación es fijo.
    El modelo puede aprender esta relación implícitamente.

    DIFERENCIAL DE TASAS (momentum_differential):
    growth_rate_7d - growth_rate_14d > 0: la semana reciente creció más rápido
    que la quincena → la ola está acelerando EN ESTE PERÍODO.
    Valor negativo: la semana reciente creció menos → la ola está frenando.
    Este diferencial tiene mayor poder predictivo que cualquiera de las dos
    tasas individuales para identificar inflexiones.
    """
    w = _state_window()

    df = df.withColumn(
        "_cases_14d_ago",
        F.lag("daily_cases", 14).over(w)
    )

    df = df.withColumn(
        "growth_rate_14d_pct",
        F.when(
            F.col("_cases_14d_ago") > 0,
            F.round(
                (F.col("daily_cases") - F.col("_cases_14d_ago"))
                / F.col("_cases_14d_ago") * 100,
                2
            )
        ).otherwise(F.lit(None).cast(DoubleType()))
    )

    df = df.withColumn(
        # Diferencial: indica si la ola está acelerando (>0) o frenando (<0)
        # dentro de la quincena más reciente.
        "momentum_differential",
        F.when(
            F.col("growth_rate_7d_pct").isNotNull() &
            F.col("growth_rate_14d_pct").isNotNull(),
            F.round(F.col("growth_rate_7d_pct") - F.col("growth_rate_14d_pct"), 2)
        ).otherwise(F.lit(None).cast(DoubleType()))
    )

    return df.drop("_cases_14d_ago")


# =============================================================================
# VARIABLE 5: ACELERACIÓN DE CONTAGIOS (segunda derivada)
# =============================================================================

def add_acceleration(df: DataFrame) -> DataFrame:
    """
    Segunda derivada discreta de la serie de casos — aceleración epidémica.

    FUNDAMENTO MATEMÁTICO:
    ----------------------
    Análogamente a la mecánica clásica:
      Posición    → casos_acumulados
      Velocidad   → daily_cases         (primera diferencia)
      Aceleración → Δdaily_cases / Δt   (segunda diferencia)

    En discreto, con Δt = 1 día:
      acceleration_raw(t) = daily_cases(t) - 2·daily_cases(t-1) + daily_cases(t-2)

    UTILIDAD ESTADÍSTICA:
    ---------------------
    La aceleración resuelve una ambigüedad crítica del modelado de epidemias:
    dos estados con la MISMA tasa de crecimiento positiva pueden estar en
    fases opuestas de la ola:

    Estado A: daily_cases = [100, 200, 400]  → growth_rate = +100% (acelerando)
    Estado B: daily_cases = [400, 600, 800]  → growth_rate = +50%  (desacelerando)

    La tasa de crecimiento no distingue entre estos casos; la aceleración sí.

    INTERPRETACIÓN DE SIGNOS:
    ┌─────────────────────────────────────────────────────────────┐
    │  accel > 0 y growth > 0 → Expansión ACELERADA (peor caso)  │
    │  accel < 0 y growth > 0 → Expansión FRENANDO (cerca pico)  │
    │  accel < 0 y growth < 0 → Contracción ACELERADA (bajada)   │
    │  accel > 0 y growth < 0 → Contracción FRENANDO (rebote?)   │
    └─────────────────────────────────────────────────────────────┘
    Los cruces de cero de la aceleración con growth positivo identifican
    el punto de inflexión — el momento más cercano al pico epidémico.

    DOS VERSIONES:
    - acceleration_raw: segunda diferencia directa de daily_cases. Muy reactiva,
      útil para detección de inflexiones en tiempo real pero ruidosa.
    - acceleration_smooth: primera diferencia de growth_rate_7d (con lag 7).
      Suavizada, menos reactiva. Mejor como feature en modelos predictivos.

    PROPIEDAD ESTADÍSTICA — Estacionariedad:
    Si daily_cases es I(1) (una raíz unitaria), su primera diferencia es I(0)
    (estacionaria). La segunda diferencia (aceleración_raw) puede ser sobre-
    diferenciada y producir una serie I(-1) más variable de lo óptimo.
    Verificar con tests ADF/KPSS antes de usar en modelos ARIMA.
    """
    w = _state_window()

    # --- Versión cruda: segunda diferencia de daily_cases ---
    df = (
        df
        .withColumn("_cases_t1", F.lag("daily_cases", 1).over(w))
        .withColumn("_cases_t2", F.lag("daily_cases", 2).over(w))
        .withColumn(
            "acceleration_raw",
            F.when(
                F.col("_cases_t1").isNotNull() & F.col("_cases_t2").isNotNull(),
                # d²C/dt² ≈ C(t) - 2C(t-1) + C(t-2)
                F.col("daily_cases")
                - 2 * F.col("_cases_t1")
                + F.col("_cases_t2")
            ).otherwise(F.lit(None).cast(DoubleType()))
        )
        .drop("_cases_t1", "_cases_t2")
    )

    # --- Versión suavizada: primera diferencia de la tasa de crecimiento ---
    df = df.withColumn(
        "_prev_growth_rate",
        F.lag("growth_rate_7d_pct", 7).over(w)
    ).withColumn(
        "acceleration_smooth",
        F.when(
            F.col("_prev_growth_rate").isNotNull() &
            F.col("growth_rate_7d_pct").isNotNull(),
            # Δgrowth_rate / 7 días → unidades: puntos porcentuales por día
            F.round(
                (F.col("growth_rate_7d_pct") - F.col("_prev_growth_rate")) / 7.0,
                4
            )
        ).otherwise(F.lit(None).cast(DoubleType()))
    ).drop("_prev_growth_rate")

    # --- Señal de inflexión: cambio de signo en la aceleración ---
    df = df.withColumn(
        "_prev_accel_smooth",
        F.lag("acceleration_smooth", 1).over(w)
    ).withColumn(
        # True cuando la aceleración cambia de signo → posible pico o valle
        "inflection_point",
        F.when(
            F.col("_prev_accel_smooth").isNotNull() &
            F.col("acceleration_smooth").isNotNull(),
            (
                (F.col("acceleration_smooth") > 0) != (F.col("_prev_accel_smooth") > 0)
            )
        ).otherwise(F.lit(False))
    ).drop("_prev_accel_smooth")

    return df


# =============================================================================
# VARIABLE 6: TASA DE MORTALIDAD (CFR acumulada + rolling)
# =============================================================================

def add_rolling_cfr(df: DataFrame) -> DataFrame:
    """
    Case Fatality Rate (CFR) en ventana móvil de 14 días.

    La CFR acumulada (cfr_pct, ya en Silver) refleja la mortalidad histórica
    completa — estable pero lenta en capturar cambios recientes.
    La CFR rolling captura cambios en la severidad clínica reciente.

    UTILIDAD ESTADÍSTICA:
    ---------------------
    El gap entre CFR acumulada y CFR rolling es una señal informativa:

    1. cfr_rolling_14d > cfr_pct (acumulada):
       La mortalidad RECIENTE es mayor que la histórica.
       Posibles causas: nueva variante más letal, colapso de UCI,
       subregistro de casos (denominador subestimado).

    2. cfr_rolling_14d < cfr_pct (acumulada):
       La mortalidad RECIENTE es menor que la histórica.
       Posibles causas: vacunación efectiva, variante menos virulenta,
       mejor capacidad de testing (más casos detectados → denominador mayor).

    IMPORTANTE — Lag epidemiológico del denominador:
    Las muertes COVID-19 ocurren en promedio 14-21 días después del diagnóstico.
    La CFR calculada como deaths(t) / cases(t) mezcla casos recientes (aún vivos)
    con muertes de casos de hace 2-3 semanas. Una corrección más rigurosa sería:
      cfr_lag_corrected = deaths(t) / cases(t - 14)
    Esta versión se incluye como `cfr_lag_corrected_14d` para uso en modelos
    que requieren precisión epidemiológica.

    VARIABLES GENERADAS:
    - cfr_rolling_14d:      deaths/cases en ventana de 14 días (reactiva)
    - cfr_delta:            cfr_rolling_14d - cfr_pct (gap acumulada vs rolling)
    - cfr_lag_corrected_14d: deaths(t) / cases(t-14) (corrección de lag)
    """
    roll_14 = _state_roll(14)
    w = _state_window()

    df = df.withColumn(
        "_sum_deaths_14d",
        F.sum("daily_deaths").over(roll_14)
    ).withColumn(
        "_sum_cases_14d",
        F.sum("daily_cases").over(roll_14)
    ).withColumn(
        "cfr_rolling_14d",
        F.when(
            F.col("_sum_cases_14d") > 0,
            F.round(F.col("_sum_deaths_14d") / F.col("_sum_cases_14d") * 100, 4)
        ).otherwise(F.lit(None).cast(DoubleType()))
    )

    # Gap entre CFR rolling y acumulada: positivo = mortalidad reciente peor
    df = df.withColumn(
        "cfr_delta",
        F.when(
            F.col("cfr_rolling_14d").isNotNull() & F.col("cfr_pct").isNotNull(),
            F.round(F.col("cfr_rolling_14d") - F.col("cfr_pct"), 4)
        ).otherwise(F.lit(None).cast(DoubleType()))
    )

    # CFR con corrección de lag: muertes hoy / casos de hace 14 días
    df = df.withColumn(
        "_cases_14d_ago_cfr",
        F.lag("cases", 14).over(w)
    ).withColumn(
        "cfr_lag_corrected_14d",
        F.when(
            F.col("_cases_14d_ago_cfr") > 0,
            F.round(F.col("deaths") / F.col("_cases_14d_ago_cfr") * 100, 4)
        ).otherwise(F.lit(None).cast(DoubleType()))
    )

    return df.drop("_sum_deaths_14d", "_sum_cases_14d", "_cases_14d_ago_cfr")


# =============================================================================
# VARIABLE 7: DÍAS DESDE EL PRIMER CASO (por estado)
# =============================================================================

def add_days_since_first_case(df: DataFrame) -> DataFrame:
    """
    Días transcurridos desde el primer caso positivo registrado en cada estado.

    DISTINCIÓN CRÍTICA respecto a days_since_start (en Silver):
    - days_since_start: días desde el primer caso NACIONAL (21-Ene-2020).
      Es idéntico para todos los estados → captura el tiempo absoluto.
    - days_since_first_case_state: días desde el PRIMER CASO del ESTADO.
      Varía por estado → captura la posición en el ciclo epidémico estatal.

    Ejemplo: Washington tuvo su primer caso el 21-Ene-2020.
             Louisiana tuvo su primer caso el 09-Mar-2020.
             El 01-Abr-2020, Washington llevaba 71 días de epidemia y
             Louisiana solo 23 días — situaciones epidemiológicas muy distintas
             aunque la fecha calendario sea la misma.

    UTILIDAD ESTADÍSTICA:
    ---------------------
    El COVID-19 sigue una curva logística (S-curve) en cada estado:
      - Fase 1 (días 1-30): crecimiento exponencial, la pendiente aumenta.
      - Fase 2 (días 30-90): crecimiento lineal → aproxima al pico.
      - Fase 3 (días 90+): desaceleración, la curva se aplana.

    days_since_first_case_state es un predictor del RÉGIMEN de la curva,
    no de la magnitud. Permite al modelo aprender que en los primeros 30 días
    el crecimiento es exponencial y en los siguientes 60 días se frena.

    INTERPRETACIÓN EN CONTEXTO MULTI-OLA:
    Con múltiples olas, days_since_first_case_state crece monotónicamente
    hasta el final del período (1,157 días para los primeros estados).
    En olas tardías, el valor ya es grande para todos los estados → la feature
    captura la etapa macro de la pandemia, no la etapa de cada ola individual.
    Para el contexto intra-ola, usar days_since_wave_start (no implementado
    aquí — requiere detección automática de changepoints).

    IMPLEMENTACIÓN:
    Broadcast join de una tabla de primeros casos (una fila por estado) contra
    el DataFrame completo. Preferible al Window unboundedPreceding con MIN
    para evitar calcular MIN sobre 1,157 filas por estado en cada fila.
    """
    # Calcular la fecha del primer caso positivo por estado
    first_case_per_state = (
        df
        .filter(F.col("cases") > 0)
        .groupBy("state")
        .agg(F.min("date").alias("first_case_date"))
    )

    # Broadcast: ~56 filas (una por estado/territorio)
    df = df.join(F.broadcast(first_case_per_state), on="state", how="left")

    df = df.withColumn(
        "days_since_first_case_state",
        F.when(
            F.col("first_case_date").isNotNull(),
            F.datediff(F.col("date"), F.col("first_case_date"))
        ).otherwise(F.lit(None))
    )

    # Transformación logarítmica del tiempo para capturar la curvatura de la S-curve.
    # log(days + 1) comprime el rango de [0, 1157] a [0, 7.05] y linealiza
    # la relación entre tiempo y casos durante la fase de crecimiento exponencial.
    df = df.withColumn(
        "log_days_since_first_case",
        F.when(
            F.col("days_since_first_case_state").isNotNull(),
            F.round(
                F.log(F.col("days_since_first_case_state").cast(DoubleType()) + 1.0),
                4
            )
        ).otherwise(F.lit(None).cast(DoubleType()))
    )

    return df.drop("first_case_date")


# =============================================================================
# VARIABLE 8: INTENSIDAD DE OLA PANDÉMICA
# =============================================================================

def add_wave_intensity(df: DataFrame) -> DataFrame:
    """
    Índice de intensidad pandémica relativa al pico histórico de cada estado.

    PROBLEMA QUE RESUELVE:
    ----------------------
    Comparar la situación epidémica de California (40M hab.) con Wyoming
    (580K hab.) usando casos absolutos es engañoso: California puede tener
    50,000 casos/día mientras Wyoming tiene 1,000. ¿Cuál está peor?

    La intensidad de ola normaliza cada estado respecto a su PROPIO historial:
      wave_intensity = rolling_avg_7d(t) / max_histórico(rolling_avg_7d)

    Si California está a 25,000 (su pico fue 90,000) → intensity = 0.28
    Si Wyoming está a 800 (su pico fue 1,000) → intensity = 0.80
    Wyoming está epidemiológicamente mucho más comprometido.

    UTILIDAD ESTADÍSTICA:
    ---------------------
    1. NORMALIZACIÓN DE ESCALA:
       Rango [0, 1] por construcción (o >1 si el pico cambia — ver nota).
       Permite tratamiento uniforme de estados de tamaño muy distinto en
       modelos entrenados sobre el conjunto completo de estados.

    2. CAPTURA DE POSICIÓN EN LA OLA:
       intensity → 1.0: el estado está en su pico o cerca de él.
       intensity ↓ desde 1.0: post-pico, fase de recuperación.
       intensity ↑ hacia 1.0: aproximación al pico.
       Esta variable codifica información estructural sobre la dinámica de ola
       que ninguna de las otras variables captura directamente.

    3. COMPARABILIDAD INTER-OLA:
       En la Wave 6 (Omicron), el pico fue 3-5× mayor que en waves anteriores.
       Un estado en wave_intensity = 0.5 durante Wave 3 y durante Wave 6 vive
       realidades sanitarias muy distintas en términos absolutos, pero similares
       en términos relativos a su propia capacidad de respuesta.

    NOTA SOBRE LOOK-AHEAD BIAS:
    El denominador usa el MÁXIMO EXPANDIBLE (solo datos hasta el tiempo t),
    no el máximo total del período. Si el pico real es en el día 800 y estamos
    prediciendo el día 400, el denominador es el máximo hasta el día 400.
    Esto garantiza que la feature no "conoce" el futuro.

    Consecuencia: wave_intensity puede alcanzar valores = 1.0 en múltiples
    fechas (cada vez que se alcanza un nuevo máximo local). El valor nunca
    supera 1.0 por construcción.

    VERSIÓN Z-SCORE (wave_intensity_zscore):
    Alternativa estandarizada: (x - μ_estado) / σ_estado usando el expanding
    window para μ y σ. No acotada en [0,1] pero captura desviaciones respecto
    a la media histórica de cada estado, útil para detectar olas excepcionales.

    VARIABLES GENERADAS:
    - state_running_peak:    máximo rolling_avg_7d visto hasta hoy (expandible)
    - wave_intensity:        rolling_avg_7d / state_running_peak ∈ [0, 1]
    - state_running_mean:    media expandible de rolling_avg_7d
    - state_running_std:     desviación estándar expandible
    - wave_intensity_zscore: (rolling_avg_7d - running_mean) / running_std
    """
    expanding = _state_expanding()

    # Máximo histórico expandible (sin look-ahead)
    df = df.withColumn(
        "state_running_peak",
        F.max("rolling_avg_7d_cases").over(expanding)
    )

    df = df.withColumn(
        "wave_intensity",
        F.when(
            F.col("state_running_peak") > 0,
            F.round(
                F.col("rolling_avg_7d_cases") / F.col("state_running_peak"),
                4
            )
        ).otherwise(F.lit(0.0).cast(DoubleType()))
    )

    # Z-score expandible (versión no acotada de intensidad)
    df = df.withColumn(
        "state_running_mean",
        F.round(F.avg("rolling_avg_7d_cases").over(expanding), 2)
    ).withColumn(
        "state_running_std",
        F.round(F.stddev("rolling_avg_7d_cases").over(expanding), 2)
    )

    df = df.withColumn(
        "wave_intensity_zscore",
        F.when(
            F.col("state_running_std") > 0,
            F.round(
                (F.col("rolling_avg_7d_cases") - F.col("state_running_mean"))
                / F.col("state_running_std"),
                4
            )
        ).otherwise(F.lit(0.0).cast(DoubleType()))
    )

    return df


# =============================================================================
# VARIABLE AUXILIAR: VOLATILIDAD (rolling standard deviation)
# =============================================================================

def add_case_lags(df: DataFrame) -> DataFrame:
    """
    Lags autorregresivos de casos diarios: 7, 14 y 21 días.

    Cada lag captura un ciclo epidemiológico distinto:
      lag_7:  período de incubación medio del COVID-19 (~5-7 días)
      lag_14: ciclo completo síntoma → test → reporte
      lag_21: ola de transmisión secundaria (dos generaciones virales)

    Junto con rolling_avg_7d_cases, estos lags convierten el modelo en un
    proceso AR(21) que puede capturar la autocorrelación semanal y quincenal.
    """
    w = _state_window()
    for lag in [7, 14, 21]:
        df = df.withColumn(f"cases_lag_{lag}", F.lag("daily_cases", lag).over(w))
    return df


def add_cyclical_month(df: DataFrame) -> DataFrame:
    """
    Codificación cíclica del mes como par (sin, cos) en el círculo unitario.

    La codificación ordinal (month = 1..12) hace que un modelo lineal trate
    Diciembre (12) y Enero (1) como extremos opuestos, destruyendo la
    captura de estacionalidad invernal. La proyección sobre el círculo
    garantiza que la distancia entre meses adyacentes sea constante,
    incluyendo el salto Diciembre → Enero.

      month_sin = sin(2π × month / 12)
      month_cos = cos(2π × month / 12)
    """
    TWO_PI = 2.0 * math.pi
    return (
        df
        .withColumn("month_sin", F.round(F.sin(F.col("month") * (TWO_PI / 12.0)), 6))
        .withColumn("month_cos", F.round(F.cos(F.col("month") * (TWO_PI / 12.0)), 6))
    )


def add_volatility_features(df: DataFrame) -> DataFrame:
    """
    Volatilidad de la señal epidémica en ventana de 7 días.

    UTILIDAD ESTADÍSTICA:
    ---------------------
    La media móvil describe la TENDENCIA. La desviación estándar rolling
    describe la ESTABILIDAD de esa tendencia.

    Alta volatilidad (std_7d grande):
    - La señal diaria oscila mucho → el reporte es irregular (posibles gaps).
    - Mayor incertidumbre en las predicciones → modelos deben ampliar
      sus intervalos de confianza.
    - En epidemiología: puede indicar brotes heterogéneos (clusters localizados).

    Baja volatilidad (std_7d pequeña):
    - La transmisión es homogénea y el reporte es regular.
    - Mayor predictibilidad → errores de predicción sistemáticamente menores.

    COEFICIENTE DE VARIACIÓN (CV = std/mean):
    Normaliza la volatilidad por el nivel de la señal. Permite comparar la
    variabilidad relativa entre períodos de alta y baja transmisión:
    - Un std=500 cuando la media es 10,000 (CV=5%) es diferente a
      un std=500 cuando la media es 1,000 (CV=50%).

    APLICACIÓN EN ML:
    - Como feature: high std_7d → modelo debe ser más conservador.
    - Como diagnóstico: filas con CV > 50% pueden ser señal de datos
      problemáticos (correcciones masivas, gaps de reporte).
    """
    roll_7 = _state_roll(7)

    df = df.withColumn(
        "rolling_std_7d_cases",
        F.round(F.stddev("daily_cases").over(roll_7), 2)
    )

    df = df.withColumn(
        "rolling_cv_7d",
        F.when(
            F.col("rolling_avg_7d_cases") > 0,
            F.round(
                F.col("rolling_std_7d_cases") / F.col("rolling_avg_7d_cases") * 100,
                2
            )
        ).otherwise(F.lit(None).cast(DoubleType()))
    )

    return df


# =============================================================================
# ORQUESTADOR Y TABLA DE FEATURES FINAL
# =============================================================================

# Catálogo completo de features generadas, con metadatos para documentación
FEATURE_CATALOG = {
    # ── SEÑAL BRUTA ──────────────────────────────────────────────────────────
    "daily_cases": {
        "source": "Silver (transform.py)",
        "type": "continua, no-estacionaria",
        "rol_ml": "target principal y lag feature",
        "nota": "Primera diferencia del acumulado. Contiene artefacto semanal.",
    },
    "log_daily_cases": {
        "source": "features.py",
        "type": "continua, aproximadamente normal",
        "rol_ml": "target para modelos lineales; feature en todos los modelos",
        "nota": "Comprime heavy tail. Residuos más homoscedásticos.",
    },
    # ── SUAVIZADO ────────────────────────────────────────────────────────────
    "rolling_avg_7d_cases": {
        "source": "Silver (transform.py)",
        "type": "continua, fuertemente autocorrelacionada",
        "rol_ml": "feature de tendencia a corto plazo; lag feature estándar",
        "nota": "Filtro de paso bajo. Estándar OMS/CDC para vigilancia.",
    },
    "rolling_avg_14d_cases": {
        "source": "features.py",
        "type": "continua, fuertemente autocorrelacionada",
        "rol_ml": "feature de tendencia a largo plazo; base del MA crossover",
        "nota": "Cubre 2 períodos de incubación. Menos reactiva que MA7.",
    },
    "ma_ratio_7_14": {
        "source": "features.py",
        "type": "continua en (0, ∞), típicamente cerca de 1.0",
        "rol_ml": "feature de momentum; reemplaza a MA7 y MA14 para evitar colinealidad",
        "nota": ">1 = ola en expansión, <1 = contracción.",
    },
    "ma_crossover_bullish": {
        "source": "features.py",
        "type": "binaria",
        "rol_ml": "feature de régimen; señal de inicio/fin de ola",
        "nota": "True cuando MA7 > MA14 (golden cross).",
    },
    # ── MOMENTUM / TASA DE CRECIMIENTO ───────────────────────────────────────
    "growth_rate_7d_pct": {
        "source": "Silver (transform.py)",
        "type": "continua, aproximadamente estacionaria en rango [-100, +∞]",
        "rol_ml": "feature de velocidad epidémica; proxy de R(t)",
        "nota": ">0 ↔ R(t)>1 (epidemic expanding). Normaliza por escala.",
    },
    "growth_rate_14d_pct": {
        "source": "features.py",
        "type": "continua",
        "rol_ml": "feature de momentum a mediano plazo",
        "nota": "Más estable que la semanal. Menos reactiva a ruido diario.",
    },
    "momentum_differential": {
        "source": "features.py",
        "type": "continua, centrada en 0",
        "rol_ml": "feature de aceleración de momentum; detecta cambios de régimen",
        "nota": ">0 = ola acelerando esta semana vs quincena.",
    },
    # ── ACELERACIÓN ──────────────────────────────────────────────────────────
    "acceleration_raw": {
        "source": "features.py",
        "type": "continua, ruidosa, centrada en 0",
        "rol_ml": "feature de inflexión; señal temprana de pico",
        "nota": "Segunda diferencia de daily_cases. Muy sensible a outliers.",
    },
    "acceleration_smooth": {
        "source": "features.py",
        "type": "continua, suavizada, centrada en 0",
        "rol_ml": "feature de aceleración para modelos ML; más robusta que raw",
        "nota": "Primera diferencia de growth_rate_7d. Unidades: pp/día.",
    },
    "inflection_point": {
        "source": "features.py",
        "type": "binaria",
        "rol_ml": "señal de evento; marca posibles picos y valles",
        "nota": "True cuando acceleration_smooth cambia de signo.",
    },
    # ── MORTALIDAD ───────────────────────────────────────────────────────────
    "cfr_pct": {
        "source": "Silver (transform.py)",
        "type": "continua en [0, 100], lentamente cambiante",
        "rol_ml": "proxy de severidad de variante dominante y cobertura vacunal",
        "nota": "CFR acumulada. Lenta para capturar cambios recientes.",
    },
    "cfr_rolling_14d": {
        "source": "features.py",
        "type": "continua, más volátil que cfr_pct",
        "rol_ml": "proxy reactivo de severidad clínica reciente",
        "nota": "CFR en ventana 14d. Captura cambios de variante o saturación UCI.",
    },
    "cfr_delta": {
        "source": "features.py",
        "type": "continua, centrada en 0",
        "rol_ml": "señal de cambio de severidad; feature de tendencia mortalidad",
        "nota": ">0 = mortalidad reciente peor que histórica.",
    },
    "cfr_lag_corrected_14d": {
        "source": "features.py",
        "type": "continua",
        "rol_ml": "estimación de CFR epidemiológicamente más precisa",
        "nota": "deaths(t) / cases(t-14). Corrige el lag mortalidad/diagnóstico.",
    },
    # ── POSICIÓN TEMPORAL ────────────────────────────────────────────────────
    "days_since_first_case_state": {
        "source": "features.py",
        "type": "entera, monotónica por estado",
        "rol_ml": "captura posición en el ciclo epidémico estatal",
        "nota": "Distinto de days_since_start (nacional). Varía por estado.",
    },
    "log_days_since_first_case": {
        "source": "features.py",
        "type": "continua, cóncava",
        "rol_ml": "linealiza la relación tiempo-casos en fase exponencial",
        "nota": "log(days + 1). Mejor para modelos lineales.",
    },
    # ── INTENSIDAD RELATIVA ──────────────────────────────────────────────────
    "wave_intensity": {
        "source": "features.py",
        "type": "continua en [0, 1]",
        "rol_ml": "normalización por estado; captura posición en la ola",
        "nota": "Sin look-ahead: denominator = max histórico hasta hoy.",
    },
    "wave_intensity_zscore": {
        "source": "features.py",
        "type": "continua en (-∞, +∞), estandarizada",
        "rol_ml": "alternativa a wave_intensity; detecta olas excepcionales",
        "nota": ">2 = actividad epidémica muy por encima de la media del estado.",
    },
    # ── LAGS AUTORREGRESIVOS ─────────────────────────────────────────────────
    "cases_lag_7": {
        "source": "features.py",
        "type": "continua, no-estacionaria",
        "rol_ml": "señal AR(7); captura autocorrelación del período de incubación",
        "nota": "lag_7 ≈ período de incubación medio. Fuerte predictor a corto plazo.",
    },
    "cases_lag_14": {
        "source": "features.py",
        "type": "continua, no-estacionaria",
        "rol_ml": "señal AR(14); captura ciclo síntoma-test-reporte completo",
        "nota": "lag_14 = dos períodos de incubación. Estándar OMS para ventanas de análisis.",
    },
    "cases_lag_21": {
        "source": "features.py",
        "type": "continua, no-estacionaria",
        "rol_ml": "señal AR(21); captura efectos rezagados de segunda generación de contagios",
        "nota": "lag_21 = tres períodos de incubación. Captura rebotes de ola.",
    },
    # ── ESTACIONALIDAD CÍCLICA ───────────────────────────────────────────────
    "month_sin": {
        "source": "features.py",
        "type": "continua en [-1, 1]",
        "rol_ml": "componente sinusoidal de la estacionalidad mensual",
        "nota": "Codificación cíclica: Dic (12) y Ene (1) son adyacentes en el espacio.",
    },
    "month_cos": {
        "source": "features.py",
        "type": "continua en [-1, 1]",
        "rol_ml": "componente coseno de la estacionalidad mensual",
        "nota": "Par con month_sin. Juntos representan la posición en el ciclo anual.",
    },
    # ── VOLATILIDAD ──────────────────────────────────────────────────────────
    "rolling_std_7d_cases": {
        "source": "features.py",
        "type": "continua, no negativa",
        "rol_ml": "proxy de incertidumbre; peso de muestra en entrenamiento",
        "nota": "Alta std → mayor error de predicción esperado.",
    },
    "rolling_cv_7d": {
        "source": "features.py",
        "type": "continua en [0, ∞], típicamente [0, 100]%",
        "rol_ml": "normaliza la volatilidad por el nivel de transmisión",
        "nota": ">50% puede indicar datos problemáticos (gaps, correcciones).",
    },
}


def run_feature_engineering(states_df: DataFrame) -> DataFrame:
    """
    Aplica todas las transformaciones de features en secuencia.

    ORDEN DE APLICACIÓN:
    El orden importa porque algunas variables dependen de otras:
      1.  log_daily_cases:        depende de daily_cases (Silver)
      2.  rolling_14d + MA cross: depende de rolling_avg_7d_cases (Silver)
      3.  growth_rate_14d:        depende de growth_rate_7d_pct (Silver)
      4.  acceleration:           depende de daily_cases y growth_rate_7d_pct
      5.  rolling_cfr:            depende de cfr_pct (Silver) y daily_deaths
      6.  days_since_first_case:  depende de cases (cumulative)
      7.  wave_intensity:         depende de rolling_avg_7d_cases (Silver)
      8.  volatility:             depende de daily_cases
      9.  case_lags:              depende de daily_cases (Silver)
      10. cyclical_month:         depende de month (Silver)

    El DataFrame resultante tiene todas las variables del FEATURE_CATALOG
    más las variables Silver heredadas.

    Args:
        states_df: DataFrame Silver con las variables base ya calculadas.

    Returns:
        DataFrame enriquecido con todas las features de modelado.
    """
    log.info("Iniciando ingeniería de features para modelado predictivo...")

    df = states_df

    log.info("  [1/8] Transformación logarítmica...")
    df = add_log_daily_cases(df)

    log.info("  [2/8] Media móvil 14d + MA crossover...")
    df = add_rolling_14d(df)

    log.info("  [3/8] Tasa de crecimiento 14d + diferencial de momentum...")
    df = add_growth_rate_14d(df)

    log.info("  [4/8] Aceleración de contagios (raw + smooth)...")
    df = add_acceleration(df)

    log.info("  [5/8] CFR rolling 14d + corrección de lag...")
    df = add_rolling_cfr(df)

    log.info("  [6/8] Días desde primer caso por estado...")
    df = add_days_since_first_case(df)

    log.info("  [7/8] Intensidad de ola pandémica...")
    df = add_wave_intensity(df)

    log.info("  [8/10] Métricas de volatilidad...")
    df = add_volatility_features(df)

    log.info("  [9/10] Lags autorregresivos de casos (7, 14, 21 días)...")
    df = add_case_lags(df)

    log.info("  [10/10] Codificación cíclica del mes (sin/cos)...")
    df = add_cyclical_month(df)

    _log_feature_summary(df)
    return df


def _log_feature_summary(df: DataFrame) -> None:
    """
    Registra el catálogo de features generadas con estadísticas básicas.
    Útil para detectar features con alta tasa de nulos o rango inesperado.
    """
    feature_names = list(FEATURE_CATALOG.keys())
    existing = [c for c in feature_names if c in df.columns]

    log.info(f"Feature Engineering completado: {len(existing)} variables en el catálogo.")

    numeric_features = [
        c for c in existing
        if "binaria" not in FEATURE_CATALOG[c]["type"]
        and "Silver" not in FEATURE_CATALOG[c]["source"]
    ]

    if numeric_features:
        null_rates = df.select([
            F.round(F.mean(F.col(c).isNull().cast("int")) * 100, 1).alias(c)
            for c in numeric_features
        ]).collect()[0].asDict()

        problematic = {k: v for k, v in null_rates.items() if v and v > 30}
        if problematic:
            log.warning(f"Features con >30% nulos: {problematic}")
        else:
            log.info("Todas las features tienen <30% de nulos.")


def print_feature_catalog() -> None:
    """Imprime el catálogo de features en formato tabular para documentación."""
    print(f"\n{'='*90}")
    print("CATÁLOGO DE VARIABLES PARA MODELADO PREDICTIVO — COVID-19 NYT")
    print(f"{'='*90}")
    print(f"{'Variable':<35} {'Tipo':<30} {'Rol en ML'}")
    print(f"{'-'*90}")
    for name, meta in FEATURE_CATALOG.items():
        src = "* " if "Silver" not in meta["source"] else "  "
        print(f"{src}{name:<33} {meta['type'][:28]:<30} {meta['rol_ml'][:45]}")
    print(f"{'='*90}")
    print("* = Variable nueva (features.py)  |  Sin marca = heredada de Silver")
    print(f"Total: {len(FEATURE_CATALOG)} variables\n")
