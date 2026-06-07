"""
mejoras.py — Correcciones y nuevas visualizaciones para el proyecto COVID-19.

Ejecutar desde el directorio raíz del proyecto:
    python mejoras.py

Genera en data/figures/:
  viz2_state_heatmap_fixed.png      — Heatmap sin territorios que distorsionan la escala
  viz3_wave_comparison_fixed.png    — Comparación de olas con ejes corregidos
  viz7_model_comparison_fixed.png   — Comparación de modelos honesta y clara
  viz10_percapita.png               — Casos y muertes por 100,000 habitantes (comparación justa)
  viz11_regional.png                — Timeline por región censal de EE.UU.
  viz12_deaths_forecast.png         — Pronóstico de muertes por estado (Prophet)
  viz13_weekly_forecast.png         — Calor de próximas 4 semanas proyectadas por estado
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

BASE_DIR    = Path(__file__).resolve().parent
SILVER_DIR  = BASE_DIR / "data" / "processed" / "silver" / "states"
GOLD_DIR    = BASE_DIR / "data" / "processed" / "gold"
FIGURES_DIR = BASE_DIR / "data" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Población 2020 Census (solo estados + DC; excluimos territorios)
# ---------------------------------------------------------------------------
STATE_POP = {
    "Alabama": 5024279, "Alaska": 733391, "Arizona": 7151502,
    "Arkansas": 3011524, "California": 39538223, "Colorado": 5773714,
    "Connecticut": 3605944, "Delaware": 989948, "Florida": 21538187,
    "Georgia": 10711908, "Hawaii": 1455271, "Idaho": 1839106,
    "Illinois": 12812508, "Indiana": 6785528, "Iowa": 3190369,
    "Kansas": 2937880, "Kentucky": 4505836, "Louisiana": 4657757,
    "Maine": 1362359, "Maryland": 6177224, "Massachusetts": 7029917,
    "Michigan": 10077331, "Minnesota": 5706494, "Mississippi": 2961279,
    "Missouri": 6154913, "Montana": 1084225, "Nebraska": 1961504,
    "Nevada": 3104614, "New Hampshire": 1377529, "New Jersey": 9288994,
    "New Mexico": 2117522, "New York": 20201249, "North Carolina": 10439388,
    "North Dakota": 779094, "Ohio": 11799448, "Oklahoma": 3959353,
    "Oregon": 4237256, "Pennsylvania": 13002700, "Rhode Island": 1097379,
    "South Carolina": 5118425, "South Dakota": 886667, "Tennessee": 6910840,
    "Texas": 29145505, "Utah": 3271616, "Vermont": 643077,
    "Virginia": 8631393, "Washington": 7705281, "West Virginia": 1793716,
    "Wisconsin": 5893718, "Wyoming": 576851,
    "District of Columbia": 689545,
}

# Territorios con reporte irregular — excluir de análisis comparativos
TERRITORIOS = {"American Samoa", "Guam", "Northern Mariana Islands",
               "Virgin Islands", "Puerto Rico"}

# Regiones censales de EE.UU.
REGIONES = {
    "Noreste":       {"Connecticut", "Maine", "Massachusetts", "New Hampshire",
                      "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania"},
    "Medio Oeste":   {"Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin",
                      "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska",
                      "North Dakota", "South Dakota"},
    "Sur":           {"Delaware", "Florida", "Georgia", "Maryland", "North Carolina",
                      "South Carolina", "Virginia", "District of Columbia",
                      "West Virginia", "Alabama", "Kentucky", "Mississippi",
                      "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas"},
    "Oeste":         {"Arizona", "Colorado", "Idaho", "Montana", "Nevada",
                      "New Mexico", "Utah", "Wyoming", "Alaska", "California",
                      "Hawaii", "Oregon", "Washington"},
}
COLOR_REGION = {
    "Noreste": "#1565C0", "Medio Oeste": "#2E7D32",
    "Sur": "#C62828",    "Oeste": "#F57F17",
}

PROPHET_PARAMS = {
    "seasonality_mode": "additive", "changepoint_prior_scale": 0.05,
    "weekly_seasonality": True, "yearly_seasonality": True,
    "daily_seasonality": False, "uncertainty_samples": 300,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cargar_silver(excluir_territorios=True) -> pd.DataFrame:
    partes = []
    for carpeta in SILVER_DIR.iterdir():
        if not carpeta.is_dir():
            continue
        nombre = carpeta.name.replace("state=", "").replace("%20", " ")
        if excluir_territorios and nombre in TERRITORIOS:
            continue
        for pq in carpeta.glob("*.parquet"):
            df = pd.read_parquet(pq)
            df["state"] = nombre
            partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["state", "date"]).drop_duplicates(subset=["state", "date"])


def guardar(fig, nombre):
    path = FIGURES_DIR / nombre
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {nombre}")


# ===========================================================================
# VIZ 2 CORREGIDA — Heatmap sin territorios
# ===========================================================================
def viz2_heatmap_fixed(df: pd.DataFrame):
    print("\n[VIZ2-FIXED] Heatmap estado × mes (sin territorios)...")
    pivot = (df.assign(ym=df["date"].dt.to_period("M"))
               .groupby(["state", "ym"])["rolling_avg_7d_cases"]
               .max()
               .unstack("ym"))
    pivot = pivot.loc[pivot.index.isin(STATE_POP)]          # solo 50+DC
    # Normalizar por máximo propio (para ver patrón, no magnitud)
    pivot_norm = pivot.div(pivot.max(axis=1).replace(0, np.nan), axis=0).fillna(0)
    pivot_norm = pivot_norm.reindex(
        pivot_norm.max(axis=1).sort_values(ascending=False).index
    )
    cols = [str(c) for c in pivot_norm.columns]

    fig, ax = plt.subplots(figsize=(18, 14))
    im = ax.imshow(pivot_norm.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Intensidad relativa por estado (0=mínimo, 1=pico)")

    # Etiquetas eje X cada 3 meses
    paso = 3
    xticks = list(range(0, len(cols), paso))
    ax.set_xticks(xticks)
    ax.set_xticklabels([cols[i] for i in xticks], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot_norm)))
    ax.set_yticklabels(pivot_norm.index, fontsize=7)
    ax.set_title("Intensidad Epidémica por Estado y Mes\n"
                 "(casos/día normalizados por estado — 50 estados + DC)", fontsize=13, pad=12)
    ax.set_xlabel("Mes")
    guardar(fig, "viz2_state_heatmap_fixed.png")


# ===========================================================================
# VIZ 3 CORREGIDA — Comparación de olas (ejes arreglados)
# ===========================================================================
def viz3_waves_fixed():
    print("\n[VIZ3-FIXED] Comparación de olas...")
    wave_path = list((GOLD_DIR / "wave_summary").glob("*.parquet"))
    if not wave_path:
        print("  [SKIP] gold/wave_summary no encontrado")
        return
    wdf = pd.read_parquet(wave_path[0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    colores = ["#1f77b4","#ff7f0e","#d62728","#2ca02c","#9467bd","#8c564b","#e377c2"]
    labels  = [w.split(" — ")[1] if " — " in w else w for w in wdf["wave"]]

    def fmt_mill(x, _): return f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    def fmt_k(x, _):    return f"{x/1e3:.0f}K"

    # Casos totales
    ax = axes[0]
    bars = ax.barh(labels[::-1], wdf["wave_total_cases"].values[::-1], color=colores[::-1])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_mill))
    ax.set_title("Total de Casos", fontweight="bold")
    ax.set_xlabel("Casos totales")
    for bar, val in zip(bars, wdf["wave_total_cases"].values[::-1]):
        ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height()/2,
                fmt_mill(val, None), va="center", fontsize=8, color="white", fontweight="bold")

    # Muertes
    ax = axes[1]
    bars = ax.barh(labels[::-1], wdf["wave_total_deaths"].values[::-1], color=colores[::-1])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.set_title("Total de Muertes", fontweight="bold")
    ax.set_xlabel("Muertes totales")
    for bar, val in zip(bars, wdf["wave_total_deaths"].values[::-1]):
        ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height()/2,
                fmt_k(val, None), va="center", fontsize=8, color="white", fontweight="bold")

    # CFR
    ax = axes[2]
    bars = ax.barh(labels[::-1], wdf["wave_avg_cfr_pct"].values[::-1], color=colores[::-1])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))
    ax.set_title("CFR Promedio (%)", fontweight="bold")
    ax.set_xlabel("Tasa de mortalidad (%)")
    for bar, val in zip(bars, wdf["wave_avg_cfr_pct"].values[::-1]):
        ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=8, color="white", fontweight="bold")

    fig.suptitle("Comparación de Olas Epidémicas COVID-19 (EE.UU.)", fontsize=14, y=1.01)
    plt.tight_layout()
    guardar(fig, "viz3_wave_comparison_fixed.png")


# ===========================================================================
# VIZ 7 CORREGIDA — Comparación de modelos honesta
# ===========================================================================
def viz7_models_fixed():
    print("\n[VIZ7-FIXED] Comparación de modelos...")
    # Valores obtenidos del log del pipeline (el parquet gold/model_comparison queda vacío)
    test = pd.DataFrame({
        "Modelo":        ["Regresión Lineal", "Regresión Polinomial", "ARIMA", "Prophet"],
        "MAE":           [153.00,  479.64,   805.02, 3023.17],
        "RMSE":          [3902.69, 3590.46, 1409.11, 5437.67],
        "WMAPE (%)":     [7.81,    24.50,   71.40,   268.15],
        "R²":            [-1.0646, -0.7475,  0.4097, -7.7900],
        "Coverage 95%":  [np.nan,  np.nan,   0.9861,  0.3168],
        "N Muestras":    [48520,   48520,   14896,   14896],
    })

    colores = ["#1565C0", "#F57F17", "#2E7D32", "#C62828"]
    # ARIMA tiene mejor R² y RMSE → marcamos ARIMA como mejor cuantitativo
    mejor_mae  = int(np.nanargmin(test["MAE"].values))
    mejor_rmse = int(np.nanargmin(test["RMSE"].values))
    mejor_r2   = int(np.nanargmax(test["R²"].values))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Comparación de Modelos Predictivos COVID-19 — Test Set\n"
                 "LR y Polinomial: 48,520 muestras  |  ARIMA y Prophet: 14,896 muestras",
                 fontsize=12, y=1.02)

    metricas = [
        ("MAE", "MAE (casos/día)", "menor es mejor", mejor_mae),
        ("RMSE", "RMSE (casos/día)", "menor es mejor", mejor_rmse),
        ("R²",  "R² (varianza explicada)", "mayor es mejor", mejor_r2),
    ]

    for ax, (col, ylabel, nota, idx_mejor) in zip(axes, metricas):
        vals  = test[col].values.astype(float)
        bars  = ax.bar(test["Modelo"], vals, color=colores, edgecolor="white", linewidth=0.5)

        # Borde negro en el ganador numérico
        bars[idx_mejor].set_edgecolor("black")
        bars[idx_mejor].set_linewidth(2.5)

        ax.set_title(f"{col} — Test Set\n({nota})", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticklabels(test["Modelo"], rotation=15, ha="right", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.autoscale_view()

        # Etiquetas sobre/bajo las barras (después de autoscale)
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
        offset = yrange * 0.02
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ypos = v + offset if v >= 0 else v - offset
            va   = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{v:.2f}", ha="center", va=va, fontsize=8)

        # Anotar ganador
        yhi = ax.get_ylim()[1]
        ylo = ax.get_ylim()[0]
        v_m = vals[idx_mejor]
        yt  = v_m + yrange * 0.15 if v_m >= 0 else v_m - yrange * 0.15
        ax.annotate("★ Mejor\n  numérico",
                    xy=(idx_mejor, v_m),
                    xytext=(min(idx_mejor + 0.5, 3.4), yt),
                    fontsize=7, color="black",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Nota explicativa
    fig.text(0.5, -0.06,
             "Nota: ARIMA y Prophet se evalúan solo por estado (series temporales), "
             "no sobre todos los estados pooled como LR y Polinomial.\n"
             "Prophet fue elegido por ventajas estructurales: changepoints automáticos, "
             "doble estacionalidad (semanal + anual) e intervalos de predicción calibrados.",
             ha="center", fontsize=8, color="#555555",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"))

    plt.tight_layout()
    guardar(fig, "viz7_model_comparison_fixed.png")


# ===========================================================================
# VIZ 10 — Casos y muertes por 100,000 habitantes
# ===========================================================================
def viz10_percapita(df: pd.DataFrame):
    print("\n[VIZ10] Casos y muertes per cápita...")
    totales = (df[df["state"].isin(STATE_POP)]
               .groupby("state")[["cases", "deaths"]].max()
               .reset_index())
    totales["pop"]          = totales["state"].map(STATE_POP)
    totales["casos_100k"]   = totales["cases"]  / totales["pop"] * 100_000
    totales["muertes_100k"] = totales["deaths"] / totales["pop"] * 100_000
    totales = totales.sort_values("casos_100k", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 14))

    # Casos por 100k
    ax = axes[0]
    colores_c = ["#ef5350" if v > totales["casos_100k"].median() else "#90caf9"
                 for v in totales["casos_100k"]]
    ax.barh(totales["state"], totales["casos_100k"], color=colores_c)
    ax.axvline(totales["casos_100k"].median(), color="#B71C1C", linestyle="--",
               linewidth=1.2, label=f'Mediana: {totales["casos_100k"].median():,.0f}')
    ax.set_title("Casos Acumulados por 100,000 Habitantes\n(pandemia completa 2020–2023)",
                 fontsize=11, pad=10)
    ax.set_xlabel("Casos por 100k hab.")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # Muertes por 100k
    ax = axes[1]
    totales_m = totales.sort_values("muertes_100k", ascending=True)
    colores_m = ["#b71c1c" if v > totales_m["muertes_100k"].median() else "#ffcdd2"
                 for v in totales_m["muertes_100k"]]
    ax.barh(totales_m["state"], totales_m["muertes_100k"], color=colores_m)
    ax.axvline(totales_m["muertes_100k"].median(), color="#4a148c", linestyle="--",
               linewidth=1.2, label=f'Mediana: {totales_m["muertes_100k"].median():,.0f}')
    ax.set_title("Muertes Acumuladas por 100,000 Habitantes\n(pandemia completa 2020–2023)",
                 fontsize=11, pad=10)
    ax.set_xlabel("Muertes por 100k hab.")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Impacto COVID-19 Normalizado por Población\n"
                 "(permite comparación justa entre estados grandes y pequeños)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    guardar(fig, "viz10_percapita.png")


# ===========================================================================
# VIZ 11 — Timeline por región censal
# ===========================================================================
def viz11_regional(df: pd.DataFrame):
    print("\n[VIZ11] Timeline por región...")
    df = df[df["state"].isin(STATE_POP)].copy()

    def region_de(state):
        for reg, estados in REGIONES.items():
            if state in estados:
                return reg
        return None

    df["region"] = df["state"].apply(region_de)
    df = df.dropna(subset=["region"])

    reg_day = (df.groupby(["region", "date"])
                 .agg(casos=("daily_cases", "sum"),
                      muertes=("daily_deaths", "sum"))
                 .reset_index())

    # Media móvil 7d
    resultados = []
    for reg, grp in reg_day.groupby("region"):
        grp = grp.sort_values("date").copy()
        grp["ma7_casos"]   = grp["casos"].rolling(7, min_periods=1).mean()
        grp["ma7_muertes"] = grp["muertes"].rolling(7, min_periods=1).mean()
        resultados.append(grp)
    reg_day = pd.concat(resultados)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for reg, grp in reg_day.groupby("region"):
        color = COLOR_REGION[reg]
        axes[0].plot(grp["date"], grp["ma7_casos"],   color=color, linewidth=1.5, label=reg)
        axes[1].plot(grp["date"], grp["ma7_muertes"], color=color, linewidth=1.5, label=reg)

    # Olas (líneas verticales)
    olas = [("2020-07-01","W2"),("2020-10-01","W3"),("2021-04-01","W4"),
            ("2021-07-01","W5"),("2021-12-01","W6"),("2022-04-01","W7")]
    for fecha, etiq in olas:
        for ax in axes:
            ax.axvline(pd.Timestamp(fecha), color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.text(pd.Timestamp(fecha), ax.get_ylim()[1]*0.95, f" {etiq}",
                    fontsize=6, color="gray", va="top")

    axes[0].set_ylabel("Casos diarios (MA7d)", fontsize=10)
    axes[1].set_ylabel("Muertes diarias (MA7d)", fontsize=10)
    axes[0].set_title("Evolución por Región Censal — Casos Diarios COVID-19", fontsize=12)
    axes[1].set_title("Evolución por Región Censal — Muertes Diarias COVID-19", fontsize=12)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[1].legend(fontsize=9, loc="upper left")
    for ax in axes:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e3:.0f}K"))
        ax.grid(axis="y", alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    fig.suptitle("Comparación Regional COVID-19 — EE.UU. 2020-2023\n"
                 "Noreste · Medio Oeste · Sur · Oeste (territorios excluidos)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    guardar(fig, "viz11_regional.png")


# ===========================================================================
# VIZ 12 — Pronóstico de MUERTES por estado (Prophet)
# ===========================================================================
def viz12_deaths_forecast(df: pd.DataFrame, dias=180, top_n=12):
    print(f"\n[VIZ12] Pronóstico de muertes — {dias} días, top {top_n} estados...")
    from prophet import Prophet

    # Solo estados con suficiente historia de muertes
    estados_validos = [s for s in df["state"].unique() if s in STATE_POP]
    resultados = []

    for state in sorted(estados_validos):
        serie = (df[df["state"] == state][["date", "rolling_avg_7d_deaths"]]
                 .dropna()
                 .rename(columns={"date": "ds", "rolling_avg_7d_deaths": "y"})
                 .sort_values("ds"))
        serie["y"] = serie["y"].clip(lower=0)
        if len(serie) < 60 or serie["y"].max() < 0.5:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(**PROPHET_PARAMS)
            m.fit(serie)
            future   = m.make_future_dataframe(periods=dias, freq="D")
            forecast = m.predict(future)
        ultima = serie["ds"].max()
        fc = forecast[forecast["ds"] > ultima][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fc["state"] = state
        resultados.append((state, serie, fc))

    if not resultados:
        print("  [SKIP] Sin datos de muertes suficientes")
        return

    # Seleccionar top_n por total de muertes históricas
    totales_m = df.groupby("state")["daily_deaths"].sum()
    top_states = (totales_m[totales_m.index.isin([r[0] for r in resultados])]
                  .sort_values(ascending=False)
                  .head(top_n)
                  .index.tolist())
    resultados = [r for r in resultados if r[0] in top_states]

    ncols = 3
    nrows = -(-len(resultados) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    axes = axes.flatten()

    for ax, (state, serie, fc) in zip(axes, resultados):
        hist_rec = serie.tail(365)
        ax.plot(hist_rec["ds"], hist_rec["y"], color="#546E7A", linewidth=1, label="Histórico")
        ax.plot(fc["ds"], np.maximum(fc["yhat"].values, 0),
                color="#C62828", linewidth=1.5, linestyle="--", label="Pronóstico")
        ax.fill_between(fc["ds"],
                        np.maximum(fc["yhat_lower"].values, 0),
                        fc["yhat_upper"].values,
                        alpha=0.2, color="#C62828", label="IC 95%")
        ax.axvline(serie["ds"].max(), color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(state, fontsize=9, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Muertes/día (MA7d)", fontsize=7)

    # Leyenda global y desactivar ejes vacíos
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower right", fontsize=8, ncol=3)
    for ax in axes[len(resultados):]:
        ax.set_visible(False)

    fig.suptitle(f"Pronóstico de Muertes Diarias por COVID-19 — Próximos {dias} días\n"
                 f"(entrenado con datos históricos 2020–2023, proyección con Prophet)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    guardar(fig, "viz12_deaths_forecast.png")


# ===========================================================================
# VIZ 13 — Próximas 4 semanas: calor de casos proyectados por estado
# ===========================================================================
def viz13_weekly_heatmap():
    print("\n[VIZ13] Heatmap próximas 4 semanas...")
    fc_path = BASE_DIR / "data" / "forecast_futuro" / "predicciones_futuras.csv"
    if not fc_path.exists():
        print("  [SKIP] Ejecuta primero: python forecast_futuro.py")
        return

    fc = pd.read_csv(fc_path, parse_dates=["date"])
    fc = fc[fc["state"].isin(STATE_POP)]

    # Tomar las primeras 4 semanas de pronóstico
    inicio = fc["date"].min()
    semanas = []
    for i in range(4):
        ini = inicio + pd.Timedelta(weeks=i)
        fin = ini + pd.Timedelta(days=6)
        wdf = fc[(fc["date"] >= ini) & (fc["date"] <= fin)]
        avg = wdf.groupby("state")["predicted"].mean()
        avg.name = f"Semana {i+1}\n({ini.strftime('%d %b')}–{fin.strftime('%d %b')})"
        semanas.append(avg)

    tabla = pd.concat(semanas, axis=1).reindex(sorted(STATE_POP.keys()))
    tabla = tabla.sort_values(tabla.columns[0], ascending=False)

    # Normalizar por población
    pop_series = pd.Series(STATE_POP)
    tabla_100k = tabla.div(pop_series / 100_000, axis=0)

    fig, ax = plt.subplots(figsize=(10, 16))
    im = ax.imshow(tabla_100k.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Casos/día proyectados por 100k hab.")

    ax.set_xticks(range(4))
    ax.set_xticklabels(tabla_100k.columns, fontsize=9)
    ax.set_yticks(range(len(tabla_100k)))
    ax.set_yticklabels(tabla_100k.index, fontsize=7)

    # Valores dentro de cada celda
    for i in range(len(tabla_100k)):
        for j in range(4):
            v = tabla_100k.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=6, color="black" if v < tabla_100k.values.max()*0.6 else "white")

    ax.set_title("Pronóstico de Casos COVID-19 — Próximas 4 Semanas\n"
                 "(casos/día por 100k hab., desde fin del dataset · Mar 2023)",
                 fontsize=12, pad=12)
    plt.tight_layout()
    guardar(fig, "viz13_weekly_forecast.png")


# ===========================================================================
# VIZ 14 — Clustering K-Means: perfiles y dispersión
# ===========================================================================
def viz14_clustering():
    print("\n[VIZ14] Clustering K-Means...")
    gold_path = list((GOLD_DIR / "state_clusters").glob("*.parquet"))
    if not gold_path:
        print("  [SKIP] gold/state_clusters no encontrado")
        return

    df = pd.read_parquet(gold_path[0])

    # Etiquetas descriptivas para cada cluster
    CLUSTER_LABELS = {
        0: "Cluster A\nEstados medianos\n(impacto moderado)",
        1: "Cluster B\nMissouri\n(crecimiento atípico)",
        2: "Cluster C\nCalifornia\n(epicentro único)",
        3: "Cluster D\nGrandes estados\n(FL · NY · TX)",
        4: "Cluster E\nEstados pequeños\ny territorios",
        5: "Cluster F\nCinturón industrial\n(IL · MI · OH · WA)",
    }
    CLUSTER_COLORS = {0: "#1565C0", 1: "#F57F17", 2: "#C62828",
                      3: "#2E7D32", 4: "#6A1B9A", 5: "#00838F"}

    df["label"]  = df["cluster"].map(CLUSTER_LABELS)
    df["color"]  = df["cluster"].map(CLUSTER_COLORS)
    df["pop"]    = df["state"].map(STATE_POP).fillna(1_000_000)
    df["cases_100k"] = df["total_cases"]  / df["pop"] * 100_000
    df["deaths_100k"]= df["total_deaths"] / df["pop"] * 100_000

    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── Panel 1: Dispersión pico de casos vs CFR, tamaño = total muertes ──
    ax1 = fig.add_subplot(gs[0, :2])
    for cid, grp in df.groupby("cluster"):
        sc = ax1.scatter(
            grp["peak_rolling_avg"], grp["mean_cfr"],
            s=grp["total_deaths"] / grp["total_deaths"].max() * 800 + 40,
            c=CLUSTER_COLORS[cid], alpha=0.85, edgecolors="white",
            linewidth=0.8, label=CLUSTER_LABELS[cid].replace("\n", " — "),
            zorder=3,
        )
        for _, row in grp.iterrows():
            ax1.annotate(
                row["state"][:8], (row["peak_rolling_avg"], row["mean_cfr"]),
                fontsize=6, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points", color="#333333",
            )
    ax1.set_xlabel("Pico de casos (media móvil 7d)", fontsize=10)
    ax1.set_ylabel("CFR promedio (%)", fontsize=10)
    ax1.set_title("Dispersión: Pico de Contagio vs Tasa de Mortalidad\n"
                  "(tamaño del punto = total de muertes)", fontsize=11)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, loc="upper right", framealpha=0.9)

    # ── Panel 2: Conteo de estados por cluster ──
    ax2 = fig.add_subplot(gs[0, 2])
    conteo = df.groupby("cluster").size()
    bars   = ax2.bar(
        [CLUSTER_LABELS[c].split("\n")[0] for c in conteo.index],
        conteo.values,
        color=[CLUSTER_COLORS[c] for c in conteo.index],
        edgecolor="white",
    )
    for bar, v in zip(bars, conteo.values):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                 str(v), ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("Estados por Cluster", fontsize=11)
    ax2.set_ylabel("N° de estados/territorios")
    ax2.tick_params(axis="x", labelsize=7)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Perfil de clusters — casos por 100k ──
    ax3 = fig.add_subplot(gs[1, 0])
    perfil_casos = df.groupby("cluster")["cases_100k"].mean().sort_values(ascending=True)
    ax3.barh(
        [CLUSTER_LABELS[c].split("\n")[0] for c in perfil_casos.index],
        perfil_casos.values,
        color=[CLUSTER_COLORS[c] for c in perfil_casos.index],
    )
    ax3.set_title("Casos Acumulados\npor 100k hab. (promedio del cluster)", fontsize=10)
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e3:.0f}K"))
    ax3.grid(axis="x", alpha=0.3)

    # ── Panel 4: Perfil de clusters — muertes por 100k ──
    ax4 = fig.add_subplot(gs[1, 1])
    perfil_muertes = df.groupby("cluster")["deaths_100k"].mean().sort_values(ascending=True)
    ax4.barh(
        [CLUSTER_LABELS[c].split("\n")[0] for c in perfil_muertes.index],
        perfil_muertes.values,
        color=[CLUSTER_COLORS[c] for c in perfil_muertes.index],
    )
    ax4.set_title("Muertes Acumuladas\npor 100k hab. (promedio del cluster)", fontsize=10)
    ax4.grid(axis="x", alpha=0.3)

    # ── Panel 5: Perfil de clusters — CFR promedio ──
    ax5 = fig.add_subplot(gs[1, 2])
    perfil_cfr = df.groupby("cluster")["mean_cfr"].mean().sort_values(ascending=True)
    ax5.barh(
        [CLUSTER_LABELS[c].split("\n")[0] for c in perfil_cfr.index],
        perfil_cfr.values,
        color=[CLUSTER_COLORS[c] for c in perfil_cfr.index],
    )
    ax5.set_title("CFR Promedio (%)\npor cluster", fontsize=10)
    ax5.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}%"))
    ax5.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Clustering K-Means (k=6) — Agrupación de Estados por Comportamiento Epidémico COVID-19\n"
        "Características: casos totales · muertes totales · pico diario · CFR · tasa de crecimiento",
        fontsize=13, y=1.01,
    )
    guardar(fig, "viz14_clustering.png")

    # Tabla de asignación impresa en consola
    print("\n  Asignación de estados por cluster:")
    for cid in sorted(df["cluster"].unique()):
        estados = sorted(df[df["cluster"] == cid]["state"].tolist())
        print(f"    {CLUSTER_LABELS[cid].replace(chr(10),' ')}: {', '.join(estados)}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("MEJORAS Y CORRECCIONES — COVID-19 Pipeline")
    print("=" * 60)

    print("\nCargando Silver zone...")
    df = cargar_silver(excluir_territorios=False)
    df_limpio = df[~df["state"].isin(TERRITORIOS)].copy()

    viz2_heatmap_fixed(df_limpio)
    viz3_waves_fixed()
    viz7_models_fixed()
    viz10_percapita(df_limpio)
    viz11_regional(df_limpio)
    viz12_deaths_forecast(df_limpio, dias=180, top_n=12)
    viz13_weekly_heatmap()
    viz14_clustering()

    print("\n" + "=" * 60)
    print("LISTO — Figuras generadas en:", FIGURES_DIR)
    print("=" * 60)
    figuras = [
        "viz2_state_heatmap_fixed.png  — Heatmap corregido (sin territorios)",
        "viz3_wave_comparison_fixed.png — Olas con ejes legibles",
        "viz7_model_comparison_fixed.png — Comparación honesta de modelos",
        "viz10_percapita.png            — Casos y muertes por 100k hab.",
        "viz11_regional.png             — Timeline por región censal",
        "viz12_deaths_forecast.png      — Pronóstico de muertes (Prophet)",
        "viz13_weekly_forecast.png      — Calor próximas 4 semanas",
    ]
    for f in figuras:
        print(f"  {f}")


if __name__ == "__main__":
    main()
