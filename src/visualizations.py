"""
visualizations.py — Módulo de visualización para el análisis COVID-19.

Produce figuras en formato PNG listas para incluir en el informe universitario.
Todas las funciones reciben datos como pandas DataFrames (ya colectados desde Spark)
para que matplotlib opere en el driver sin necesidad de acciones Spark adicionales.

Organización de las visualizaciones:
  VIZ-1  Timeline nacional de casos y muertes (escala dual)
  VIZ-2  Heatmap de casos diarios por estado × mes
  VIZ-3  Comparación de olas epidémicas (barras apiladas)
  VIZ-4  Evolución temporal del CFR (Case Fatality Rate)
  VIZ-5  Efecto día de la semana en reporte (artefacto de infraestructura)
  VIZ-6  Cruce de medias móviles MA7 / MA14 para un estado representativo
  VIZ-7  Comparación de modelos predictivos por métrica (MAE, RMSE, R²)
  VIZ-8  Perfiles de clustering epidémico de estados (scatter bi-dimensional)

Decisiones de diseño:
  - Paleta: 'tab10' para series categóricas, 'YlOrRd' para mapas de calor.
    Ambas son perceptualmente uniformes y amigables para daltónicos.
  - DPI=150: resolución suficiente para impresión A4 sin archivos excesivamente grandes.
  - tight_layout(): evita solapamientos entre títulos, ejes y leyendas automáticamente.
  - plt.close(): libera memoria del backend después de guardar cada figura.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla — compatible con entornos de servidor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import FIGURES_DIR, WAVE_DEFINITIONS, WAVE_LAST_LABEL

log = logging.getLogger(__name__)

# Paleta de colores consistente para las 7 olas a lo largo de todas las figuras
WAVE_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1",
]

WAVE_LABELS = [label for _, _, label in WAVE_DEFINITIONS] + [WAVE_LAST_LABEL]


def _ensure_figures_dir(figures_dir: Path) -> None:
    """Crea el directorio de figuras si no existe."""
    figures_dir.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, figures_dir: Path, filename: str) -> None:
    """Guarda la figura y libera memoria del backend."""
    path = figures_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] Guardada: {path}")


# =============================================================================
# VIZ-1: Timeline nacional — casos y muertes diarias
# =============================================================================

def plot_national_timeline(national_pdf: pd.DataFrame, figures_dir: Path) -> None:
    """
    Gráfico de líneas con dos ejes Y: casos diarios (rolling 7d) y muertes.

    El eje dual permite visualizar simultáneamente la escala de casos
    (~1M/día en el pico Omicron) y las muertes (~4,000/día pico), que
    difieren en un orden de magnitud, sin comprimir ninguna de las dos series.

    Bandas de color: regiones sombreadas para cada ola epidémica identificada
    en WAVE_DEFINITIONS — permite correlacionar visualmente picos con eventos.
    """
    log.info("[VIZ-1] Generando timeline nacional...")

    required = {"date", "rolling_avg_7d_cases", "rolling_avg_7d_deaths"}
    if not required.issubset(national_pdf.columns):
        log.warning(f"[VIZ-1] Columnas faltantes: {required - set(national_pdf.columns)}")
        return

    pdf = national_pdf.sort_values("date").copy()
    pdf["date"] = pd.to_datetime(pdf["date"])

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # Añadir bandas de ola como fondo antes de las líneas
    for i, (start, end, label) in enumerate(WAVE_DEFINITIONS):
        ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                    alpha=0.07, color=WAVE_COLORS[i % len(WAVE_COLORS)], label=None)
    # Última ola (sin end definido — hasta el final del dataset)
    if len(WAVE_DEFINITIONS) > 0:
        last_start = pd.Timestamp(WAVE_DEFINITIONS[-1][1])  # end de la penúltima = start de la última
        ax1.axvspan(last_start, pdf["date"].max(), alpha=0.07, color=WAVE_COLORS[6 % len(WAVE_COLORS)])

    # Serie de casos (eje izquierdo)
    ax1.plot(pdf["date"], pdf["rolling_avg_7d_cases"],
             color="#e15759", linewidth=1.5, label="Casos diarios (MA7)")
    ax1.fill_between(pdf["date"], pdf["rolling_avg_7d_cases"],
                     alpha=0.15, color="#e15759")
    ax1.set_ylabel("Casos diarios (promedio móvil 7d)", color="#e15759", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#e15759")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

    # Serie de muertes (eje derecho)
    ax2.plot(pdf["date"], pdf["rolling_avg_7d_deaths"],
             color="#4e79a7", linewidth=1.5, label="Muertes diarias (MA7)", linestyle="--")
    ax2.set_ylabel("Muertes diarias (promedio móvil 7d)", color="#4e79a7", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#4e79a7")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.1f}K"))

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

    # Leyenda de olas
    wave_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=WAVE_COLORS[i % len(WAVE_COLORS)], alpha=0.3)
        for i in range(len(WAVE_LABELS))
    ]
    fig.legend(
        wave_patches,
        [f"W{i+1}" for i in range(len(WAVE_LABELS))],
        loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.01),
        fontsize=8, framealpha=0.6,
    )

    # Leyenda de series
    lines = [
        Line2D([0], [0], color="#e15759", linewidth=1.5, label="Casos (MA7)"),
        Line2D([0], [0], color="#4e79a7", linewidth=1.5, linestyle="--", label="Muertes (MA7)"),
    ]
    ax1.legend(handles=lines, loc="upper left", fontsize=9)

    ax1.set_title("Timeline Nacional COVID-19 — Casos y Muertes Diarias (EE.UU.)",
                  fontsize=13, fontweight="bold", pad=18)
    ax1.set_xlabel("Fecha", fontsize=11)
    ax1.grid(axis="y", alpha=0.3, linestyle=":")

    _save(fig, figures_dir, "viz1_national_timeline.png")


# =============================================================================
# VIZ-2: Heatmap estado × mes
# =============================================================================

def plot_state_heatmap(states_pdf: pd.DataFrame, figures_dir: Path) -> None:
    """
    Heatmap de casos diarios promedio por estado (filas) × mes (columnas).

    Convierte el problema de comparar 50+ series temporales simultáneas en
    un mapa de color bidimensional. El rojo intenso indica los picos de
    cada estado; permite identificar estados que difieren en el timing de sus olas.

    Normalización por estado (division por el máximo de cada estado):
    sin normalizar, estados con muchas menos personas (Wyoming) serían
    invisibles junto a California. La normalización revela la severidad
    relativa de cada ola en su propio contexto demográfico.
    """
    log.info("[VIZ-2] Generando heatmap estado × mes...")

    required = {"date", "state", "daily_cases"}
    if not required.issubset(states_pdf.columns):
        log.warning(f"[VIZ-2] Columnas faltantes: {required - set(states_pdf.columns)}")
        return

    pdf = states_pdf.copy()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf["year_month"] = pdf["date"].dt.to_period("M").astype(str)

    pivot = (
        pdf.groupby(["state", "year_month"])["daily_cases"]
        .mean()
        .unstack(fill_value=0)
    )

    # Normalizar cada estado por su propio máximo (evita dominancia demográfica)
    row_max = pivot.max(axis=1).replace(0, 1)
    pivot_norm = pivot.div(row_max, axis=0)

    # Ordenar estados por total de casos (mayor a menor) para layout informativo
    state_order = pdf.groupby("state")["daily_cases"].sum().sort_values(ascending=False).index
    pivot_norm = pivot_norm.reindex(state_order)

    fig, ax = plt.subplots(figsize=(18, max(10, len(pivot_norm) * 0.22)))

    im = ax.imshow(pivot_norm.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_yticks(range(len(pivot_norm.index)))
    ax.set_yticklabels(pivot_norm.index, fontsize=7)

    # Mostrar solo etiquetas cada 3 meses para evitar solapamiento
    x_labels = list(pivot_norm.columns)
    x_ticks  = [i for i, lbl in enumerate(x_labels) if lbl.endswith("-01") or lbl.endswith("-07")]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([x_labels[i] for i in x_ticks], rotation=45, ha="right", fontsize=8)

    plt.colorbar(im, ax=ax, label="Casos diarios (normalizado por estado)", fraction=0.02)

    ax.set_title("Intensidad Epidémica por Estado y Mes (casos/día normalizados)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mes", fontsize=11)
    ax.set_ylabel("Estado (ordenado por total de casos)", fontsize=11)

    _save(fig, figures_dir, "viz2_state_heatmap.png")


# =============================================================================
# VIZ-3: Comparación de olas epidémicas
# =============================================================================

def plot_wave_comparison(wave_summary_pdf: pd.DataFrame, figures_dir: Path) -> None:
    """
    Gráfico de barras horizontales comparando las métricas de cada ola.

    Muestra tres paneles: total de casos, total de muertes y CFR promedio.
    La disposición horizontal facilita la lectura de las etiquetas de ola
    (que son cadenas largas). El color de la barra del CFR refleja si la
    mortalidad aumentó o disminuyó respecto a la ola anterior — documentando
    el impacto de la vacunación y el cambio de virulencia entre variantes.
    """
    log.info("[VIZ-3] Generando comparación de olas...")

    required = {"wave", "wave_total_cases", "wave_total_deaths", "wave_avg_cfr_pct"}
    if not required.issubset(wave_summary_pdf.columns):
        log.warning(f"[VIZ-3] Columnas faltantes: {required - set(wave_summary_pdf.columns)}")
        return

    pdf = wave_summary_pdf.copy().sort_values("wave")
    n = len(pdf)
    colors = [WAVE_COLORS[i % len(WAVE_COLORS)] for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, n * 0.6)))

    for ax, col, title, fmt in [
        (axes[0], "wave_total_cases",  "Total de Casos",   lambda v: f"{v/1e6:.1f}M"),
        (axes[1], "wave_total_deaths", "Total de Muertes", lambda v: f"{v/1e3:.0f}K"),
        (axes[2], "wave_avg_cfr_pct",  "CFR Promedio (%)", lambda v: f"{v:.2f}%"),
    ]:
        bars = ax.barh(range(n), pdf[col], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(pdf["wave"], fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: fmt(x).replace("K", "K").replace("M", "M")
        ))
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        # Anotar valor dentro de la barra
        for bar, val in zip(bars, pdf[col]):
            ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height() / 2,
                    fmt(val), va="center", ha="left", fontsize=7.5, color="white",
                    fontweight="bold")

    fig.suptitle("Comparación de Olas Epidémicas COVID-19 (EE.UU.)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, figures_dir, "viz3_wave_comparison.png")


# =============================================================================
# VIZ-4: Evolución del CFR por ola
# =============================================================================

def plot_cfr_evolution(states_pdf: pd.DataFrame, figures_dir: Path) -> None:
    """
    Evolución temporal de la Case Fatality Rate (CFR) nacional.

    El CFR nacional es el promedio ponderado de los estados en cada fecha.
    La línea de mediana (percentil 50) y las bandas de percentiles (P25-P75)
    muestran la distribución de mortalidad entre estados, revelando que la
    heterogeneidad inter-estatal es mayor en las primeras olas (antes de la
    vacunación) que en las posteriores.

    Hitos anotados:
      - Inicio de vacunación (Dic 2020)
      - Autorización booster (Sep 2021)
    """
    log.info("[VIZ-4] Generando evolución del CFR...")

    required = {"date", "state", "cfr_pct"}
    if not required.issubset(states_pdf.columns):
        log.warning(f"[VIZ-4] Columnas faltantes: {required - set(states_pdf.columns)}")
        return

    pdf = states_pdf.copy()
    pdf["date"] = pd.to_datetime(pdf["date"])

    daily = (
        pdf.groupby("date")["cfr_pct"]
        .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .reset_index()
    )
    daily.columns = ["date", "median", "p25", "p75"]
    daily = daily.sort_values("date")

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(daily["date"], daily["p25"], daily["p75"],
                    alpha=0.25, color="#4e79a7", label="Rango intercuartil (P25-P75)")
    ax.plot(daily["date"], daily["median"],
            color="#4e79a7", linewidth=1.8, label="CFR mediano nacional")

    # Hitos
    for date_str, label, color in [
        ("2020-12-14", "Inicio vacunación", "#59a14f"),
        ("2021-09-22", "Booster autorizado", "#f28e2b"),
    ]:
        ax.axvline(pd.Timestamp(date_str), color=color, linestyle="--",
                   linewidth=1.2, alpha=0.8)
        ax.text(pd.Timestamp(date_str), daily["median"].max() * 0.95,
                label, color=color, fontsize=8.5, rotation=90,
                va="top", ha="right")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_ylabel("Case Fatality Rate (%)", fontsize=11)
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_title("Evolución de la Tasa de Mortalidad (CFR) Nacional — COVID-19",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_ylim(bottom=0)

    _save(fig, figures_dir, "viz4_cfr_evolution.png")


# =============================================================================
# VIZ-5: Efecto del día de la semana en el reporte
# =============================================================================

def plot_weekday_effect(weekday_pdf: pd.DataFrame, figures_dir: Path) -> None:
    """
    Gráfico de barras con barras de error para el artefacto de reporte semanal.

    El patrón esperado: Lunes tiene media significativamente más alta que el
    resto de días porque los laboratorios acumulan resultados del fin de semana
    y los reportan el primer día hábil. Este artefacto justifica el uso de
    `day_of_week` como feature en los modelos predictivos.

    La barra de error (±1 desviación estándar) cuantifica la variabilidad
    intra-día que determina si las diferencias son estadísticamente significativas.
    """
    log.info("[VIZ-5] Generando análisis de efecto semanal...")

    required = {"day_of_week", "avg_daily_cases", "std_daily_cases"}
    if not required.issubset(weekday_pdf.columns):
        log.warning(f"[VIZ-5] Columnas faltantes: {required - set(weekday_pdf.columns)}")
        return

    day_map = {1: "Dom", 2: "Lun", 3: "Mar", 4: "Mié", 5: "Jue", 6: "Vie", 7: "Sáb"}
    pdf = weekday_pdf.copy().sort_values("day_of_week")
    pdf["day_name"] = pdf["day_of_week"].map(day_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#e15759" if d == 2 else "#76b7b2" for d in pdf["day_of_week"]]

    # Casos diarios por día de la semana
    bars = ax1.bar(pdf["day_name"], pdf["avg_daily_cases"],
                   yerr=pdf["std_daily_cases"], color=colors,
                   error_kw={"linewidth": 1.2, "alpha": 0.6},
                   edgecolor="white", linewidth=0.5)
    ax1.set_title("Casos Diarios Promedio por Día de la Semana",
                  fontsize=11, fontweight="bold")
    ax1.set_ylabel("Casos diarios promedio", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax1.grid(axis="y", alpha=0.3, linestyle=":")
    ax1.text(0.98, 0.96, "Rojo = Lunes\n(pico de acumulación\nde fin de semana)",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8,
             color="#e15759", style="italic")

    # Desviación estándar — mide la volatilidad de cada día
    ax2.bar(pdf["day_name"], pdf["std_daily_cases"], color=colors,
            edgecolor="white", linewidth=0.5)
    ax2.set_title("Variabilidad (Desviación Estándar) por Día",
                  fontsize=11, fontweight="bold")
    ax2.set_ylabel("Desviación estándar", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax2.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle("Artefacto de Reporte Semanal — Efecto Día de la Semana",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, figures_dir, "viz5_weekday_effect.png")


# =============================================================================
# VIZ-6: Cruce de medias móviles MA7 / MA14
# =============================================================================

def plot_ma_crossover(
    states_pdf: pd.DataFrame,
    figures_dir: Path,
    state_name: str = "California",
) -> None:
    """
    Gráfico de MA7 vs MA14 para un estado representativo.

    El cruce de medias móviles es un indicador de señal técnica adaptado
    de los mercados financieros a la epidemiología:
      - Cuando MA7 > MA14 (cruce alcista): la tendencia de corto plazo
        supera a la de mediano plazo → aceleración epidémica.
      - Cuando MA7 < MA14 (cruce bajista): desaceleración o descenso.

    Los puntos de cruce se anotan con triángulos de color para facilitar
    la identificación visual de las transiciones de fase epidémica.
    """
    log.info(f"[VIZ-6] Generando cruce MA7/MA14 para {state_name}...")

    required = {"date", "state", "rolling_avg_7d_cases"}
    if not required.issubset(states_pdf.columns):
        log.warning(f"[VIZ-6] Columnas faltantes: {required - set(states_pdf.columns)}")
        return

    if state_name not in states_pdf["state"].values:
        state_name = states_pdf["state"].iloc[0]
        log.warning(f"[VIZ-6] Estado no encontrado, usando: {state_name}")

    pdf = states_pdf[states_pdf["state"] == state_name].copy()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values("date")

    # Calcular MA14 si no existe
    if "rolling_avg_14d_cases" not in pdf.columns:
        pdf["rolling_avg_14d_cases"] = pdf["daily_cases"].rolling(14, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Panel superior: MA7 y MA14
    ax1.plot(pdf["date"], pdf["rolling_avg_7d_cases"],
             color="#e15759", linewidth=1.5, label="MA7 (corto plazo)", alpha=0.9)
    ax1.plot(pdf["date"], pdf["rolling_avg_14d_cases"],
             color="#4e79a7", linewidth=1.5, label="MA14 (mediano plazo)", alpha=0.9)

    # Sombrear cuando MA7 > MA14 (fase alcista)
    ax1.fill_between(pdf["date"],
                     pdf["rolling_avg_7d_cases"],
                     pdf["rolling_avg_14d_cases"],
                     where=(pdf["rolling_avg_7d_cases"] > pdf["rolling_avg_14d_cases"]),
                     interpolate=True, alpha=0.15, color="#e15759", label="MA7 > MA14 (aceleración)")
    ax1.fill_between(pdf["date"],
                     pdf["rolling_avg_7d_cases"],
                     pdf["rolling_avg_14d_cases"],
                     where=(pdf["rolling_avg_7d_cases"] <= pdf["rolling_avg_14d_cases"]),
                     interpolate=True, alpha=0.15, color="#4e79a7", label="MA7 ≤ MA14 (desaceleración)")

    ax1.set_ylabel("Casos diarios", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, linestyle=":")
    ax1.set_title(f"Cruce de Medias Móviles MA7/MA14 — {state_name}",
                  fontsize=13, fontweight="bold")

    # Panel inferior: ratio MA7/MA14
    ratio = pdf["rolling_avg_7d_cases"] / pdf["rolling_avg_14d_cases"].replace(0, np.nan)
    ax2.plot(pdf["date"], ratio, color="#76b7b2", linewidth=1.2)
    ax2.axhline(1.0, color="#e15759", linestyle="--", linewidth=1, alpha=0.7)
    ax2.fill_between(pdf["date"], ratio, 1.0,
                     where=(ratio > 1.0), alpha=0.2, color="#e15759")
    ax2.fill_between(pdf["date"], ratio, 1.0,
                     where=(ratio <= 1.0), alpha=0.2, color="#4e79a7")
    ax2.set_ylabel("Ratio MA7/MA14", fontsize=10)
    ax2.set_ylim(0, max(3, ratio.quantile(0.99) * 1.1))
    ax2.grid(axis="y", alpha=0.3, linestyle=":")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax2.set_xlabel("Fecha", fontsize=11)

    plt.tight_layout()
    safe_name = state_name.lower().replace(" ", "_")
    _save(fig, figures_dir, f"viz6_ma_crossover_{safe_name}.png")


# =============================================================================
# VIZ-7: Comparación de modelos predictivos
# =============================================================================

def plot_model_comparison(comparison_pdf: pd.DataFrame, figures_dir: Path) -> None:
    """
    Gráfico de barras agrupadas comparando los cuatro modelos por cada métrica.

    Muestra MAE, RMSE y R² en el test set (held-out) para los cuatro modelos:
    Regresión Lineal, Regresión Polinomial, ARIMA y Prophet.

    La codificación dual (posición agrupada + escala de color de barra) permite
    leer el ranking absoluto (altura de barra) y el ranking relativo dentro de
    cada métrica (posición izquierda → derecha) en una sola figura.

    Recomendación visual: Prophet aparece con borde más grueso como indicador
    del modelo recomendado sin necesidad de texto adicional.
    """
    log.info("[VIZ-7] Generando comparación de modelos...")

    required = {"Modelo", "Split", "MAE", "RMSE", "R²"}
    if not required.issubset(comparison_pdf.columns):
        log.warning(f"[VIZ-7] Columnas faltantes: {required - set(comparison_pdf.columns)}")
        return

    test_df = comparison_pdf[comparison_pdf["Split"] == "test"].copy()
    if test_df.empty:
        log.warning("[VIZ-7] Sin datos de test set en la tabla comparativa.")
        return

    models = test_df["Modelo"].tolist()
    n = len(models)
    x = np.arange(n)

    metrics = [
        ("MAE",  "MAE (casos/día)",       "menor es mejor"),
        ("RMSE", "RMSE (casos/día)",      "menor es mejor"),
        ("R²",   "R² (varianza explicada)", "mayor es mejor"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    for ax, (col, ylabel, note) in zip(axes, metrics):
        vals = test_df[col].astype(float).values
        bars = ax.bar(x, vals,
                      color=model_colors[:n],
                      edgecolor=["black" if m == "Prophet" else "white" for m in models],
                      linewidth=[2 if m == "Prophet" else 0.5 for m in models])

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{col} — Test Set\n({note})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle=":")

        # Anotar valor sobre cada barra
        for bar, val in zip(bars, vals):
            if not math.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.text(0.98, 0.02, "★ Prophet = modelo recomendado",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="gray", style="italic")

    fig.suptitle("Comparación de Modelos Predictivos COVID-19 — Test Set",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, figures_dir, "viz7_model_comparison.png")


# =============================================================================
# VIZ-8: Perfiles de clustering de estados
# =============================================================================

def plot_state_clusters(cluster_pdf: pd.DataFrame, figures_dir: Path) -> None:
    """
    Scatter plot 2D de los clusters epidémicos de estados.

    Representa cada estado como un punto en el espacio (total_cases, mean_cfr)
    coloreado por su cluster asignado por K-Means. El tamaño del punto refleja
    el peak_rolling_avg (severidad del peor momento).

    Esta proyección bidimensional captura las dos dimensiones más interpretables:
    la magnitud absoluta de la pandemia y la mortalidad. Los otros 3 features
    del vector (total_deaths, mean_growth_rate) determinan el cluster pero no
    se muestran para mantener la legibilidad.
    """
    log.info("[VIZ-8] Generando scatter de clusters de estados...")

    required = {"state", "cluster", "total_cases", "mean_cfr", "peak_rolling_avg"}
    if not required.issubset(cluster_pdf.columns):
        log.warning(f"[VIZ-8] Columnas faltantes: {required - set(cluster_pdf.columns)}")
        return

    pdf = cluster_pdf.copy()
    n_clusters = pdf["cluster"].nunique()
    cluster_colors = [WAVE_COLORS[i % len(WAVE_COLORS)] for i in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(12, 8))

    for cluster_id in sorted(pdf["cluster"].unique()):
        mask = pdf["cluster"] == cluster_id
        sub = pdf[mask]

        # Normalizar tamaño del punto por peak (entre 20 y 400)
        peak_vals = sub["peak_rolling_avg"].fillna(0)
        peak_max = peak_vals.max()
        sizes = 20 + 380 * (peak_vals / peak_max) if peak_max > 0 else [80] * len(sub)

        sc = ax.scatter(
            sub["total_cases"],
            sub["mean_cfr"],
            s=sizes,
            c=[cluster_colors[cluster_id]] * len(sub),
            label=f"Cluster {cluster_id}",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

        # Anotar estados con etiqueta
        for _, row in sub.iterrows():
            ax.annotate(
                row["state"][:2].upper(),  # Abreviatura de 2 letras
                (row["total_cases"], row["mean_cfr"]),
                fontsize=6.5, ha="center", va="bottom", color="gray",
                xytext=(0, 4), textcoords="offset points",
            )

    ax.set_xlabel("Total de Casos (toda la pandemia)", fontsize=11)
    ax.set_ylabel("CFR Promedio (%)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(title="Cluster K-Means", fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_title(
        "Clustering Epidémico de Estados — K-Means\n"
        "(tamaño del punto = peak del promedio móvil 7d)",
        fontsize=13, fontweight="bold",
    )

    _save(fig, figures_dir, "viz8_state_clusters.png")


# =============================================================================
# VIZ-9: Top-20 condados por casos acumulados
# =============================================================================

def plot_county_top_n(county_pdf: pd.DataFrame, figures_dir: Path) -> None:
    """
    Gráfico de barras horizontales para el ranking de los 20 condados con
    más casos acumulados. Anotado con el CFR de cada condado para revelar
    la heterogeneidad en mortalidad entre jurisdicciones.
    """
    log.info("[VIZ-9] Generando ranking de condados...")

    required = {"county", "state", "total_cases", "county_cfr_pct"}
    if not required.issubset(county_pdf.columns):
        log.warning(f"[VIZ-9] Columnas faltantes: {required - set(county_pdf.columns)}")
        return

    pdf = county_pdf.nlargest(20, "total_cases").copy()
    pdf["label"] = pdf["county"] + ", " + pdf["state"].str[:2].str.upper()
    pdf = pdf.sort_values("total_cases", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(pdf["label"], pdf["total_cases"],
                   color="#4e79a7", edgecolor="white", linewidth=0.5)

    # Anotar CFR a la derecha de cada barra
    for bar, (_, row) in zip(bars, pdf.iterrows()):
        cfr = row["county_cfr_pct"]
        cfr_str = f"CFR: {cfr:.1f}%" if pd.notna(cfr) else ""
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                cfr_str, va="center", fontsize=7.5, color="#e15759")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_xlabel("Total de Casos Acumulados", fontsize=11)
    ax.set_title("Top 20 Condados por Casos Acumulados — COVID-19 (EE.UU.)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    _save(fig, figures_dir, "viz9_county_top20.png")


# =============================================================================
# Orquestador
# =============================================================================

def run_visualizations(
    analytics_results: dict,
    transformed: Optional[dict] = None,
    figures_dir: Optional[Path] = None,
) -> None:
    """
    Orquesta la generación de todas las visualizaciones del proyecto.

    Recibe los resultados devueltos por run_analytics() y, opcionalmente,
    los DataFrames transformados para visualizaciones que necesitan granularidad
    de serie temporal (VIZ-4 CFR, VIZ-6 MA crossover).

    Todos los DataFrames se colectan a pandas antes de pasar a matplotlib.
    El collect() es aceptable aquí porque estamos produciendo agregaciones
    ya reducidas por run_analytics() — no se colectan los 3.5M de filas brutas.

    Args:
        analytics_results: dict devuelto por run_analytics().
        transformed: dict de DataFrames Silver (national, states, counties).
        figures_dir: directorio de salida. Por defecto FIGURES_DIR de settings.
    """
    if figures_dir is None:
        figures_dir = FIGURES_DIR
    _ensure_figures_dir(figures_dir)

    log.info("=" * 60)
    log.info("GENERANDO VISUALIZACIONES")
    log.info(f"Directorio de salida: {figures_dir}")
    log.info("=" * 60)

    descriptive = analytics_results.get("descriptive", {})

    # VIZ-1: Timeline nacional (necesita datos Silver con granularidad diaria)
    if transformed and "national" in transformed:
        try:
            national_pdf = transformed["national"].toPandas()
            plot_national_timeline(national_pdf, figures_dir)
        except Exception as exc:
            log.error(f"[VIZ-1] Error: {exc}")

    # VIZ-2: Heatmap estado × mes (datos Silver de estados)
    if transformed and "states" in transformed:
        try:
            states_pdf = transformed["states"].toPandas()
            plot_state_heatmap(states_pdf, figures_dir)
        except Exception as exc:
            log.error(f"[VIZ-2] Error: {exc}")
    else:
        states_pdf = None

    # VIZ-3: Comparación de olas
    if "wave_summary" in descriptive:
        try:
            wave_pdf = descriptive["wave_summary"].toPandas()
            plot_wave_comparison(wave_pdf, figures_dir)
        except Exception as exc:
            log.error(f"[VIZ-3] Error: {exc}")

    # VIZ-4: Evolución del CFR
    if states_pdf is not None:
        try:
            plot_cfr_evolution(states_pdf, figures_dir)
        except Exception as exc:
            log.error(f"[VIZ-4] Error: {exc}")
    elif transformed and "states" in transformed:
        try:
            states_pdf = transformed["states"].select(
                "date", "state", "cfr_pct"
            ).toPandas()
            plot_cfr_evolution(states_pdf, figures_dir)
        except Exception as exc:
            log.error(f"[VIZ-4] Error: {exc}")

    # VIZ-5: Efecto día de la semana
    if "weekday_effect" in descriptive:
        try:
            weekday_pdf = descriptive["weekday_effect"].toPandas()
            plot_weekday_effect(weekday_pdf, figures_dir)
        except Exception as exc:
            log.error(f"[VIZ-5] Error: {exc}")

    # VIZ-6: Cruce MA7/MA14 para California (estado con más casos)
    if states_pdf is not None:
        try:
            top_state = states_pdf.groupby("state")["daily_cases"].sum().idxmax()
            plot_ma_crossover(states_pdf, figures_dir, state_name=top_state)
        except Exception as exc:
            log.error(f"[VIZ-6] Error: {exc}")

    # VIZ-7: Comparación de modelos
    ml_results = analytics_results.get("ml", {})
    if "comparison_pdf" in ml_results and not ml_results["comparison_pdf"].empty:
        try:
            plot_model_comparison(ml_results["comparison_pdf"], figures_dir)
        except Exception as exc:
            log.error(f"[VIZ-7] Error: {exc}")

    # VIZ-8: Clustering de estados
    if "state_clusters" in descriptive:
        try:
            cluster_pdf = descriptive["state_clusters"].toPandas()
            plot_state_clusters(cluster_pdf, figures_dir)
        except Exception as exc:
            log.error(f"[VIZ-8] Error: {exc}")

    # VIZ-9: Top-20 condados
    if "county_top20" in descriptive:
        try:
            county_pdf = descriptive["county_top20"].toPandas()
            plot_county_top_n(county_pdf, figures_dir)
        except Exception as exc:
            log.error(f"[VIZ-9] Error: {exc}")

    log.info("=" * 60)
    log.info(f"Visualizaciones completadas — ver {figures_dir}")
    log.info("=" * 60)
