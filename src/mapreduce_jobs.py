"""
mapreduce_jobs.py — Implementación explícita del paradigma MapReduce.

CONTEXTO ACADÉMICO
==================
Este módulo implementa tres procesos MapReduce reales sobre el dataset
COVID-19 del New York Times utilizando la API de bajo nivel RDD de Apache Spark.

El paradigma MapReduce fue introducido por Dean & Ghemawat (Google, 2004) como
modelo de programación para procesamiento paralelo de grandes volúmenes de datos
en clusters de servidores. El modelo abstrae la complejidad de la distribución,
tolerancia a fallos y sincronización en tres fases conceptuales:

  ┌─────────────────────────────────────────────────────────────┐
  │  INPUT → [MAP] → [SHUFFLE/SORT] → [REDUCE] → OUTPUT        │
  └─────────────────────────────────────────────────────────────┘

MAP:
  Función pura aplicada de forma independiente a cada registro de entrada.
  Emite pares intermedios (clave, valor). No hay comunicación entre mappers.
  Escalabilidad horizontal directa: el trabajo se divide entre tantos mappers
  como particiones del dataset.

SHUFFLE / SORT:
  Fase gestionada automáticamente por el framework (Hadoop/Spark).
  1. Particionado: cada clave intermedia se asigna a un reducer mediante
     una función hash: partición = hash(clave) mod num_reducers
  2. Transferencia de red: los pares se mueven entre nodos
  3. Ordenamiento: dentro de cada partición, las claves se ordenan
  Esta es la fase más costosa en tiempo de red y I/O.

REDUCE:
  Función aplicada a todos los valores asociados a una misma clave.
  Los reducers operan en paralelo sobre particiones distintas.
  Produce el output final del job.

COMBINER (optimización):
  Mini-reducer que se ejecuta en el nodo del mapper ANTES del shuffle.
  Agrega localmente los valores de la misma clave para reducir el volumen
  de datos transferidos por la red. Solo aplicable cuando la función de
  reducción es asociativa y conmutativa (ej. suma, máximo, mínimo).
  En Spark: `reduceByKey` incluye combiner implícito.
            `groupByKey` NO incluye combiner → evitar para agregaciones.

REFERENCIA: Dean, J., & Ghemawat, S. (2004). MapReduce: Simplified data
  processing on large clusters. OSDI'04, pp. 137-150.

IMPLEMENTACIÓN EN SPARK RDD:
  La API de RDD (Resilient Distributed Dataset) expone directamente el
  paradigma MapReduce:
    .map(f)            → fase Map
    .reduceByKey(f)    → Shuffle implícito + Reduce con Combiner
    .groupByKey()      → Shuffle implícito + agrupación (sin Combiner)
    .sortByKey()       → Shuffle con particionado por rango + Sort local
    .zipWithIndex()    → Enumeración secuencial post-ordenamiento
"""

import logging
from pathlib import Path
from typing import List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.rdd import RDD

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PROCESSED_DIR

log = logging.getLogger(__name__)


# =============================================================================
# JOB 1: TOTAL DE CASOS ACUMULADOS POR ESTADO
# =============================================================================
#
# JUSTIFICACIÓN DE USO:
#   El dataset almacena registros diarios por estado (~1,157 días × ~56 estados
#   = ~65,000 filas). Sumar los casos diarios de todos los registros de cada
#   estado es un problema de agregación por clave distribuida — el caso de uso
#   canónico de MapReduce. El modelo Map→Reduce particiona el trabajo entre
#   nodos sin que ningún nodo necesite ver todos los datos.
#
# PSEUDOCÓDIGO:
# ┌────────────────────────────────────────────────────────────────┐
# │  CLASS CasesMapper:                                            │
# │    FUNCTION map(offset, row):                                  │
# │      state      ← row.state                                    │
# │      daily_cases ← row.daily_cases ?? 0                        │
# │      EMIT(state, daily_cases)         ← par (clave, valor)     │
# │                                                                │
# │  CLASS CasesCombiner:   # mini-reducer local al mapper node    │
# │    FUNCTION combine(state, local_values[]):                    │
# │      EMIT(state, SUM(local_values))                            │
# │                                                                │
# │  CLASS CasesReducer:                                           │
# │    FUNCTION reduce(state, values[]):                           │
# │      total ← SUM(values)                                       │
# │      EMIT(state, total)                                        │
# └────────────────────────────────────────────────────────────────┘
#
# FLUJO DE DATOS (ejemplo simplificado):
#
#  Input (filas del dataset):
#   (2020-03-01, California, 01, daily_cases=120, ...)
#   (2020-03-02, California, 01, daily_cases=145, ...)
#   (2020-03-01, Texas,      48, daily_cases=80,  ...)
#   (2020-03-02, Texas,      48, daily_cases=95,  ...)
#
#  Después de MAP:
#   Mapper 1 emite: ("California", 120), ("California", 145)
#   Mapper 2 emite: ("Texas", 80),       ("Texas", 95)
#
#  Después de COMBINER (local en cada nodo):
#   Nodo 1: ("California", 265)    ← 120 + 145, reducción local
#   Nodo 2: ("Texas", 175)         ← 80 + 95, reducción local
#
#  SHUFFLE: todos los pares con key="California" van al mismo reducer
#           todos los pares con key="Texas"      van al mismo reducer
#
#  Después de REDUCE:
#   Reducer A: ("California", 9_500_000)  ← suma de todos los días
#   Reducer B: ("Texas",      6_200_000)
# =============================================================================

def mr1_total_casos_por_estado(states_df: DataFrame) -> List[Tuple[str, int]]:
    """
    Job 1 — Total de casos nuevos acumulados por estado (2020-2023).

    Patrón: Word Count extendido a suma numérica.
    Complejidad: O(n) en Map, O(k log k) en Shuffle donde k = num. estados (~56).

    Args:
        states_df: DataFrame Silver con columna `daily_cases` ya calculada.

    Returns:
        Lista de tuplas (estado, total_casos) ordenada descendentemente.
    """
    log.info("[MR-1] Iniciando: Total de casos por estado...")

    # -------------------------------------------------------------------------
    # FASE MAP
    # Transforma cada fila del dataset en un par (clave, valor).
    # La función lambda es el Mapper: recibe una Row y emite (state, daily_cases).
    # Cada partición del RDD ejecuta su mapper de forma completamente independiente.
    # Los nulos en daily_cases se imputan a 0 antes del map para evitar
    # que TypeErrors en el reducer corrompan la agregación.
    # -------------------------------------------------------------------------
    rdd_input: RDD = (
        states_df
        .select("state", "daily_cases")
        .na.fill({"daily_cases": 0})         # guardar contra nulos residuales
        .rdd
    )

    # MAP: (Row) → (state, daily_cases)
    rdd_mapped: RDD = rdd_input.map(
        lambda row: (
            row["state"],           # CLAVE: identificador del estado
            int(row["daily_cases"]) # VALOR: casos del día
        )
    )
    # En este punto cada partición contiene pares sin ninguna ordenación global.
    # Ejemplo de partición 0: [("California", 120), ("Texas", 80), ("California", 145)]
    # Ejemplo de partición 1: [("Texas", 95), ("Florida", 200)]

    # -------------------------------------------------------------------------
    # FASE SHUFFLE (implícita en reduceByKey)
    # reduceByKey aplica internamente:
    #   1. Combiner local: agrega valores de la misma clave DENTRO de cada partición
    #      antes de enviarlos por la red. Esto es la optimización del Combiner.
    #      Sin Combiner (groupByKey): se transfieren TODOS los valores individuales.
    #      Con Combiner (reduceByKey): se transfiere 1 suma parcial por partición.
    #
    #   2. Hash partitioner: hash("California") mod num_partitions → partición destino
    #      Garantiza que TODOS los registros de "California" lleguen al mismo reducer.
    #
    #   3. Sort: dentro de cada partición destino, las claves se ordenan lexicográficamente.
    #
    # FASE REDUCE
    # La función lambda (a, b) → a + b es el Reducer aplicado a todos los valores
    # de cada clave. Produce UN output por clave distinta.
    # -------------------------------------------------------------------------
    rdd_reduced: RDD = rdd_mapped.reduceByKey(
        lambda acumulado, valor_nuevo: acumulado + valor_nuevo
        # Esta función debe ser asociativa y conmutativa para que el
        # Combiner produzca el mismo resultado que si no existiera.
        # La suma cumple ambas propiedades: (a+b)+c == a+(b+c), a+b == b+a.
    )

    # -------------------------------------------------------------------------
    # RECOLECCIÓN Y ORDENAMIENTO FINAL
    # .collect() trae todos los pares (state, total) al nodo driver.
    # Seguro porque el output tiene exactamente ~56 entradas (una por estado).
    # El sorted() final es un ordenamiento local en el driver, no distribuido.
    # -------------------------------------------------------------------------
    resultado: List[Tuple[str, int]] = sorted(
        rdd_reduced.collect(),
        key=lambda par: par[1],  # ordenar por total de casos
        reverse=True             # descendente: mayor número de casos primero
    )

    log.info(f"[MR-1] Completado: {len(resultado)} estados procesados.")
    log.info(f"[MR-1] Top 3: {resultado[:3]}")
    return resultado


# =============================================================================
# JOB 2: TOTAL DE MUERTES ACUMULADAS POR ESTADO
# =============================================================================
#
# JUSTIFICACIÓN DE USO:
#   Mismo patrón que MR-1 pero sobre la variable `daily_deaths`. Se implementa
#   como job separado para demonstrar que el paradigma MapReduce permite procesar
#   diferentes variables del mismo dataset de forma independiente y paralelizable.
#   En un cluster real, MR-1 y MR-2 pueden ejecutarse simultáneamente sobre el
#   mismo dataset sin interferencia.
#
# PSEUDOCÓDIGO:
# ┌────────────────────────────────────────────────────────────────┐
# │  CLASS DeathsMapper:                                           │
# │    FUNCTION map(offset, row):                                  │
# │      state        ← row.state                                  │
# │      daily_deaths ← row.daily_deaths ?? 0                      │
# │      EMIT(state, daily_deaths)                                 │
# │                                                                │
# │  CLASS DeathsCombiner:                                         │
# │    FUNCTION combine(state, local_values[]):                    │
# │      EMIT(state, SUM(local_values))                            │
# │                                                                │
# │  CLASS DeathsReducer:                                          │
# │    FUNCTION reduce(state, values[]):                           │
# │      total ← SUM(values)                                       │
# │      EMIT(state, total)                                        │
# └────────────────────────────────────────────────────────────────┘
#
# DIFERENCIA CONCEPTUAL CON MR-1:
#   La única diferencia está en el Mapper: extrae `daily_deaths` en lugar de
#   `daily_cases`. El Shuffle y el Reducer son idénticos. Esto ilustra que
#   el paradigma MapReduce es composable: cambiar qué se mide no cambia
#   la arquitectura del sistema, solo la función de mapeo.
# =============================================================================

def mr2_total_muertes_por_estado(states_df: DataFrame) -> List[Tuple[str, int]]:
    """
    Job 2 — Total de muertes acumuladas por estado (2020-2023).

    Patrón: idéntico a MR-1, con daily_deaths como variable objetivo.
    El resultado de este job es el INPUT del Job 3 (ranking por mortalidad).

    Args:
        states_df: DataFrame Silver con columna `daily_deaths` calculada.

    Returns:
        Lista de tuplas (estado, total_muertes) ordenada descendentemente.
    """
    log.info("[MR-2] Iniciando: Total de muertes por estado...")

    rdd_input: RDD = (
        states_df
        .select("state", "daily_deaths")
        .na.fill({"daily_deaths": 0})
        .rdd
    )

    # -------------------------------------------------------------------------
    # FASE MAP: cada fila → (state, daily_deaths)
    # -------------------------------------------------------------------------
    rdd_mapped: RDD = rdd_input.map(
        lambda row: (
            row["state"],            # CLAVE
            int(row["daily_deaths"]) # VALOR
        )
    )

    # -------------------------------------------------------------------------
    # FASES SHUFFLE + REDUCE: idéntico a MR-1
    # El framework garantiza que todos los valores para un estado dado
    # llegan al mismo reducer, independientemente de en qué nodo físico
    # estaba cada fila del dataset original.
    # -------------------------------------------------------------------------
    rdd_reduced: RDD = rdd_mapped.reduceByKey(
        lambda acumulado, valor_nuevo: acumulado + valor_nuevo
    )

    resultado: List[Tuple[str, int]] = sorted(
        rdd_reduced.collect(),
        key=lambda par: par[1],
        reverse=True
    )

    log.info(f"[MR-2] Completado: {len(resultado)} estados procesados.")
    log.info(f"[MR-2] Top 3: {resultado[:3]}")
    return resultado


# =============================================================================
# JOB 3: RANKING NACIONAL DE ESTADOS POR MORTALIDAD
# =============================================================================
#
# JUSTIFICACIÓN DE USO:
#   Producir un ranking global ordenado requiere que TODOS los datos estén
#   disponibles en el mismo lugar antes de poder asignar posiciones. Esto
#   contrasta con la suma (MR-1, MR-2) donde cada reducer trabaja de forma
#   independiente. Para el ranking, se necesita coordinar entre todos los
#   estados, lo que hace necesario un diseño especial: el patrón "Sort by Value".
#
# PATRÓN: SORT BY VALUE (inversión de clave y valor)
#   En MapReduce, el Shuffle ordena automáticamente por CLAVE.
#   Para ordenar por VALOR (total_muertes), invertimos la relación:
#     input:  (state, total_muertes)
#     mapper: emite (total_muertes, state)  ← deaths es ahora la CLAVE
#   El shuffle ordena por la nueva clave (total_muertes), produciendo
#   los estados en orden de mortalidad. Luego asignamos posición secuencial.
#
# PSEUDOCÓDIGO:
# ┌────────────────────────────────────────────────────────────────┐
# │  # Input: resultado de MR-2 = [(state, total_muertes), ...]   │
# │                                                                │
# │  CLASS RankingMapper:                                          │
# │    FUNCTION map(state, total_muertes):                         │
# │      EMIT(total_muertes, state)    ← INVERSIÓN CLAVE↔VALOR     │
# │                                                                │
# │  # SHUFFLE: el framework ordena por la clave (total_muertes)   │
# │  # Resultado: pares llegan al reducer ordenados descendentemente│
# │                                                                │
# │  CLASS RankingReducer:                                         │
# │    posicion ← 1                                                │
# │    FUNCTION reduce(total_muertes, states[]):                   │
# │      FOR state IN states:                                      │
# │        EMIT(posicion, state, total_muertes)                    │
# │        posicion ← posicion + 1                                 │
# └────────────────────────────────────────────────────────────────┘
#
# DETALLE DEL SHUFFLE EN ORDENAMIENTO GLOBAL:
#   El ordenamiento distribuido (total sort) en MapReduce/Spark requiere
#   particionado por RANGO (no por hash). Con hash partitioning, los rangos
#   de valores quedan mezclados entre particiones — no hay orden global.
#   sortByKey() en Spark usa TeraSort-style range partitioning:
#     1. Muestrea el RDD para estimar la distribución de claves.
#     2. Crea fronteras de rango (ej. [0-1000], [1001-5000], [5001-∞]).
#     3. Asigna cada par a la partición cuyo rango cubre su clave.
#     4. Cada partición ordena localmente sus pares.
#   El resultado: partición_0 contiene los menores, partición_n los mayores.
#   Concatenando en orden, el dataset completo está ordenado globalmente.
#
# FUENTE: Zaharia, M. et al. (2012). Resilient Distributed Datasets: A
#   fault-tolerant abstraction for in-memory cluster computing. NSDI'12.
# =============================================================================

def mr3_ranking_mortalidad(
    mr2_resultado: List[Tuple[str, int]],
    spark_context,
) -> List[Tuple[int, str, int, float]]:
    """
    Job 3 — Ranking nacional de estados por total de muertes acumuladas.

    Recibe el output de MR-2 (lista de (estado, muertes)) y produce un
    ranking completo con posición, tasa de mortalidad relativa y percentil.

    El job demuestra el patrón "Sort by Value" de MapReduce y la técnica
    de ordenamiento global distribuido (total sort) mediante sortByKey().

    Args:
        mr2_resultado: Output de mr2_total_muertes_por_estado().
        spark_context: SparkContext activo (sc) para crear RDDs.

    Returns:
        Lista de tuplas (posición, estado, muertes, pct_del_total) ordenada.
    """
    log.info("[MR-3] Iniciando: Ranking nacional de mortalidad...")

    total_muertes_nacional = sum(muertes for _, muertes in mr2_resultado)
    log.info(f"[MR-3] Total de muertes registradas: {total_muertes_nacional:,}")

    # -------------------------------------------------------------------------
    # Crear RDD desde el resultado de MR-2.
    # En un pipeline Hadoop puro, el output de MR-2 estaría en HDFS y
    # este RDD se leería directamente desde disco. En Spark, reutilizamos
    # el resultado en memoria para evitar I/O innecesario entre jobs encadenados.
    # -------------------------------------------------------------------------
    rdd_input: RDD = spark_context.parallelize(mr2_resultado)
    # rdd_input contiene: [("California", 9500000), ("Texas", 6200000), ...]

    # -------------------------------------------------------------------------
    # FASE MAP: inversión clave↔valor para habilitar ordenamiento por muertes.
    #
    # INPUT:  (state,         total_muertes)
    # OUTPUT: (total_muertes, state)
    #
    # Al hacer de total_muertes la CLAVE, el Shuffle la usará como criterio
    # de ordenamiento automático. Sin esta inversión, el Shuffle ordenaría
    # por nombre de estado (orden alfabético), no por mortalidad.
    # -------------------------------------------------------------------------
    rdd_invertido: RDD = rdd_input.map(
        lambda par: (
            par[1],  # NUEVA CLAVE: total_muertes (numérica, sortable)
            par[0]   # NUEVO VALOR: nombre del estado
        )
    )
    # rdd_invertido: [(9500000, "California"), (6200000, "Texas"), ...]

    # -------------------------------------------------------------------------
    # FASE SHUFFLE + SORT: sortByKey() con ascending=False.
    #
    # Internamente, sortByKey() ejecuta:
    #   1. Muestreo: toma una muestra del RDD para estimar cuantiles
    #   2. Range partitioning: crea particiones para rangos de muertes
    #      Ej: partición_0 → [0, 10.000], partición_1 → [10.001, 100.000], etc.
    #   3. Shuffle: redistribuye pares a sus particiones de rango destino
    #   4. Sort local: cada partición ordena sus propios pares
    #
    # Con ascending=False: partición_0 contiene los MAYORES valores.
    # La concatenación de particiones en orden produce el ranking global.
    # -------------------------------------------------------------------------
    rdd_ordenado: RDD = rdd_invertido.sortByKey(ascending=False)
    # rdd_ordenado: [(9500000, "California"), (6200000, "Texas"), ...]

    # -------------------------------------------------------------------------
    # FASE REDUCE: asignación de posición secuencial con zipWithIndex().
    #
    # zipWithIndex() añade un índice secuencial (0-based) a cada elemento
    # del RDD YA ORDENADO, equivalente a la fase Reduce de ranking:
    #   reducer recibe pares en orden y emite (posición, estado, muertes).
    #
    # La transformación de (deaths, state), index → (rank, state, deaths)
    # es el post-procesamiento del Reducer que convierte el índice en posición.
    # -------------------------------------------------------------------------
    rdd_con_indice: RDD = rdd_ordenado.zipWithIndex()
    # rdd_con_indice: [((9500000, "California"), 0), ((6200000, "Texas"), 1), ...]

    rdd_ranking: RDD = rdd_con_indice.map(
        lambda item: (
            item[1] + 1,                             # posición (1-indexed)
            item[0][1],                              # nombre del estado
            item[0][0],                              # total de muertes
            round(item[0][0] / total_muertes_nacional * 100, 2)  # % del total
        )
    )
    # rdd_ranking: [(1, "California", 9500000, 18.3), (2, "Texas", 6200000, 12.0), ...]

    # -------------------------------------------------------------------------
    # RECOLECCIÓN FINAL AL DRIVER
    # ~56 entradas (una por estado/territorio) → collect() es seguro y eficiente.
    # En un contexto de producción con millones de categorías, se usaría
    # take(N) para evitar saturar la memoria del driver.
    # -------------------------------------------------------------------------
    resultado: List[Tuple[int, str, int, float]] = rdd_ranking.collect()

    log.info(f"[MR-3] Completado: {len(resultado)} estados en el ranking.")
    log.info(f"[MR-3] #1: {resultado[0]}")
    log.info(f"[MR-3] #2: {resultado[1]}")
    log.info(f"[MR-3] #3: {resultado[2]}")
    return resultado


# =============================================================================
# FUNCIÓN AUXILIAR: Comparación reduceByKey vs groupByKey
# (Sección educativa — ilustra por qué reduceByKey es superior)
# =============================================================================

def demo_combiner_optimization(states_df: DataFrame) -> dict:
    """
    Demuestra empíricamente la diferencia entre reduceByKey (con Combiner)
    y groupByKey (sin Combiner) para la misma operación de suma.

    COMBINER (reduceByKey):
      Cada partición pre-agrega sus valores localmente ANTES del shuffle.
      Si hay 1,000 filas de "California" distribuidas en 10 particiones:
        - Cada partición envía UNA suma parcial → 10 registros por la red
      Total datos en shuffle: 10 × (clave + valor_parcial)

    SIN COMBINER (groupByKey):
      TODOS los valores individuales se mueven por la red antes de agregar.
      Si hay 1,000 filas de "California":
        - Todas van al mismo reducer → 1,000 registros por la red
      Total datos en shuffle: 1,000 × (clave + valor)

    En el dataset COVID-19 (~65,000 filas de estados):
      reduceByKey reduce el tráfico de red en un factor de ~1,157
      (días del dataset) respecto a groupByKey.

    REGLA: Usar reduceByKey cuando la función de reducción es asociativa
    y conmutativa. Usar groupByKey solo cuando se necesita acceso a TODOS
    los valores antes de poder calcular el resultado (ej. mediana, moda).
    """
    rdd = (
        states_df
        .select("state", "daily_cases")
        .na.fill(0)
        .rdd
        .map(lambda row: (row["state"], int(row["daily_cases"])))
    )

    # Método 1: reduceByKey CON Combiner (recomendado)
    result_reduce = rdd.reduceByKey(lambda a, b: a + b)

    # Método 2: groupByKey SIN Combiner (ineficiente — solo para demostración)
    result_group = rdd.groupByKey().mapValues(sum)

    # Ambos producen el mismo resultado numérico
    reduce_count = result_reduce.count()
    group_count  = result_group.count()

    return {
        "reduceByKey_estados": reduce_count,
        "groupByKey_estados":  group_count,
        "resultados_identicos": reduce_count == group_count,
        "nota": (
            "groupByKey transfiere todos los valores individuales por la red "
            "antes de agregar. reduceByKey agrega localmente primero (Combiner). "
            "Para sumas sobre 65K filas: reduceByKey es ~1157x más eficiente "
            "en transferencia de red."
        ),
    }


# =============================================================================
# ORQUESTADOR DE LOS TRES JOBS
# =============================================================================

def run_mapreduce_pipeline(states_df: DataFrame) -> dict:
    """
    Ejecuta los tres jobs MapReduce en secuencia.

    MR-1 y MR-2 son independientes entre sí → en un cluster real podrían
    ejecutarse en paralelo. MR-3 depende del output de MR-2 → debe esperar.

    Grafo de dependencias:
        states_df ──► MR-1 ──► resultado_casos
                 └──► MR-2 ──► resultado_muertes ──► MR-3 ──► ranking

    Args:
        states_df: DataFrame Silver transformado (con daily_cases y daily_deaths).

    Returns:
        Diccionario con los resultados de los tres jobs.
    """
    spark_context = states_df.sparkSession.sparkContext

    log.info("=" * 60)
    log.info("PIPELINE MAPREDUCÉ — INICIO")
    log.info("=" * 60)

    # Job 1: Casos por estado
    mr1_resultado = mr1_total_casos_por_estado(states_df)

    # Job 2: Muertes por estado
    mr2_resultado = mr2_total_muertes_por_estado(states_df)

    # Job 3: Ranking (depende de MR-2)
    mr3_resultado = mr3_ranking_mortalidad(mr2_resultado, spark_context)

    # Demo educativa del Combiner
    combiner_demo = demo_combiner_optimization(states_df)
    log.info(f"[COMBINER DEMO] {combiner_demo['nota']}")

    log.info("=" * 60)
    log.info("PIPELINE MAPREDUCÉ — RESULTADOS FINALES")
    log.info("=" * 60)

    _imprimir_resultados(mr1_resultado, mr2_resultado, mr3_resultado)

    return {
        "mr1_casos_por_estado":   mr1_resultado,
        "mr2_muertes_por_estado": mr2_resultado,
        "mr3_ranking_mortalidad": mr3_resultado,
        "combiner_demo":          combiner_demo,
    }


def _imprimir_resultados(
    mr1: List[Tuple[str, int]],
    mr2: List[Tuple[str, int]],
    mr3: List[Tuple[int, str, int, float]],
) -> None:
    """Imprime los resultados de los tres jobs en formato tabular legible."""

    _sep = "-" * 70

    print(f"\n{'='*70}")
    print("JOB 1 — TOTAL DE CASOS POR ESTADO (Top 10)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Estado':<25} {'Total Casos':>15}")
    print(_sep)
    for i, (estado, casos) in enumerate(mr1[:10], start=1):
        print(f"{i:<4} {estado:<25} {casos:>15,}")

    print(f"\n{'='*70}")
    print("JOB 2 — TOTAL DE MUERTES POR ESTADO (Top 10)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Estado':<25} {'Total Muertes':>15}")
    print(_sep)
    for i, (estado, muertes) in enumerate(mr2[:10], start=1):
        print(f"{i:<4} {estado:<25} {muertes:>15,}")

    print(f"\n{'='*70}")
    print("JOB 3 — RANKING NACIONAL POR MORTALIDAD (Top 10)")
    print(f"{'='*70}")
    print(f"{'Pos':<5} {'Estado':<25} {'Muertes':>12} {'% Nacional':>12}")
    print(_sep)
    for pos, estado, muertes, pct in mr3[:10]:
        print(f"{pos:<5} {estado:<25} {muertes:>12,} {pct:>11.2f}%")
    print(f"{'='*70}\n")
