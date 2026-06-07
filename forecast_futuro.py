"""
forecast_futuro.py — Predicciones futuras con Prophet usando datos ya procesados.

Carga la Silver zone (parquet por estado) sin necesidad de Spark ni de
re-correr el pipeline. Entrena Prophet con toda la serie histórica y
proyecta N días hacia adelante. Guarda CSV + gráficas PNG.

Uso:
    python forecast_futuro.py                      # 365 días (1 año)
    python forecast_futuro.py --dias 730           # 2 años
    python forecast_futuro.py --estados California Texas  # solo esos estados
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Rutas (relativas al script — mismo directorio que main.py)
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
SILVER_DIR  = BASE_DIR / "data" / "processed" / "silver" / "states"
OUT_DIR     = BASE_DIR / "data" / "forecast_futuro"
FIGURES_DIR = OUT_DIR / "figures"

PROPHET_PARAMS = {
    "seasonality_mode":        "additive",
    "changepoint_prior_scale": 0.05,
    "weekly_seasonality":      True,
    "yearly_seasonality":      True,
    "daily_seasonality":       False,
    "uncertainty_samples":     300,
}

TARGET = "rolling_avg_7d_cases"


# ---------------------------------------------------------------------------
# Carga de datos desde parquet particionados (sin Spark)
# ---------------------------------------------------------------------------
def cargar_silver_states(estados_filtro=None) -> pd.DataFrame:
    """Lee todos los parquet de silver/states y devuelve un DataFrame unificado."""
    partes = []
    for carpeta_estado in SILVER_DIR.iterdir():
        if not carpeta_estado.is_dir():
            continue
        # El nombre de la carpeta es "state=Nombre%20Estado"
        nombre = carpeta_estado.name.replace("state=", "").replace("%20", " ")
        if estados_filtro and nombre not in estados_filtro:
            continue
        for pq in carpeta_estado.glob("*.parquet"):
            df = pd.read_parquet(pq)
            df["state"] = nombre
            partes.append(df)

    if not partes:
        raise ValueError("No se encontraron archivos parquet en " + str(SILVER_DIR))

    df = pd.concat(partes, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["state", "date"]).drop_duplicates(subset=["state", "date"])
    return df


# ---------------------------------------------------------------------------
# Forecast por estado
# ---------------------------------------------------------------------------
def forecast_estado(pdf: pd.DataFrame, dias_futuro: int) -> pd.DataFrame:
    """
    Entrena Prophet con toda la serie histórica del estado y proyecta
    `dias_futuro` días hacia adelante del último dato disponible.
    Devuelve DataFrame con columnas: state, date, predicted, yhat_lower, yhat_upper.
    """
    from prophet import Prophet

    state = pdf["state"].iloc[0]

    serie = pdf[["date", TARGET]].dropna(subset=[TARGET]).copy()
    serie = serie.rename(columns={"date": "ds", TARGET: "y"})
    serie["y"] = serie["y"].clip(lower=0)

    if len(serie) < 60:
        print(f"  [{state}] Serie muy corta ({len(serie)} días), omitido.")
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(**PROPHET_PARAMS)
        model.fit(serie)
        future = model.make_future_dataframe(periods=dias_futuro, freq="D")
        forecast = model.predict(future)

    # Solo filas futuras (posteriores al último dato histórico)
    ultima_fecha = serie["ds"].max()
    fut = forecast[forecast["ds"] > ultima_fecha].copy()

    resultado = pd.DataFrame({
        "state":      state,
        "date":       fut["ds"].values,
        "predicted":  np.maximum(fut["yhat"].values, 0),
        "yhat_lower": np.maximum(fut["yhat_lower"].values, 0),
        "yhat_upper": fut["yhat_upper"].values,
    })

    print(f"  [{state}] OK — {len(resultado)} días pronosticados "
          f"({resultado['date'].min().date()} → {resultado['date'].max().date()})")
    return resultado


# ---------------------------------------------------------------------------
# Gráfica por estado
# ---------------------------------------------------------------------------
def graficar_estado(historico: pd.DataFrame, forecast: pd.DataFrame,
                    state: str, figures_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 5))

    # Últimos 365 días históricos para contexto
    hist = historico[historico["state"] == state].sort_values("date")
    hist_reciente = hist.tail(365)

    ax.plot(hist_reciente["date"], hist_reciente[TARGET],
            color="#2196F3", linewidth=1.2, label="Histórico (últ. 12m)")

    fc = forecast[forecast["state"] == state].sort_values("date")
    ax.plot(fc["date"], fc["predicted"],
            color="#FF5722", linewidth=1.5, linestyle="--", label="Pronóstico Prophet")
    ax.fill_between(fc["date"], fc["yhat_lower"], fc["yhat_upper"],
                    alpha=0.2, color="#FF5722", label="IC 95%")

    # Línea de corte histórico/futuro
    corte = hist["date"].max()
    ax.axvline(corte, color="gray", linestyle=":", linewidth=1)
    ax.text(corte, ax.get_ylim()[1] * 0.95, " Fin datos\n reales",
            fontsize=7, color="gray", va="top")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
    ax.set_title(f"Pronóstico COVID-19 — {state}", fontsize=13, pad=10)
    ax.set_ylabel("Casos diarios (media móvil 7d)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    nombre_archivo = f"forecast_{state.lower().replace(' ', '_')}.png"
    fig.savefig(figures_dir / nombre_archivo, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráfica resumen: top 10 estados por pico de pronóstico
# ---------------------------------------------------------------------------
def graficar_resumen(historico: pd.DataFrame, todos_forecasts: pd.DataFrame,
                     figures_dir: Path):
    resumen = (todos_forecasts.groupby("state")["predicted"]
               .max()
               .sort_values(ascending=False)
               .head(10))

    fig, ax = plt.subplots(figsize=(12, 5))
    colores = plt.cm.Reds(np.linspace(0.4, 0.9, len(resumen)))
    ax.barh(resumen.index[::-1], resumen.values[::-1], color=colores[::-1])
    ax.set_xlabel("Pico de casos diarios pronosticado (media móvil 7d)")
    ax.set_title("Top 10 estados — Pico máximo pronosticado en el período futuro",
                 fontsize=12, pad=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    fig.savefig(figures_dir / "forecast_resumen_top10.png", dpi=130)
    plt.close(fig)
    print(f"\n  [Resumen] Guardado: forecast_resumen_top10.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pronóstico futuro COVID-19 con Prophet")
    parser.add_argument("--dias",    type=int, default=365,
                        help="Días a pronosticar desde el fin del dataset (default: 365)")
    parser.add_argument("--estados", nargs="*", default=None,
                        help="Lista de estados a procesar (default: todos)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"PRONÓSTICO FUTURO — {args.dias} días")
    print("=" * 60)
    print(f"Leyendo Silver zone: {SILVER_DIR}")

    historico = cargar_silver_states(estados_filtro=args.estados)
    estados   = sorted(historico["state"].unique())
    print(f"Estados cargados: {len(estados)}\n")

    resultados = []
    for state in estados:
        pdf = historico[historico["state"] == state].copy()
        fc  = forecast_estado(pdf, args.dias)
        if not fc.empty:
            resultados.append(fc)
            graficar_estado(historico, fc, state, FIGURES_DIR)

    if not resultados:
        print("No se generaron pronósticos.")
        return

    todos = pd.concat(resultados, ignore_index=True)
    graficar_resumen(historico, todos, FIGURES_DIR)

    csv_path = OUT_DIR / "predicciones_futuras.csv"
    todos.to_csv(csv_path, index=False, float_format="%.2f")

    print("\n" + "=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"  Estados procesados : {todos['state'].nunique()}")
    print(f"  Período pronosticado: {todos['date'].min().date()} → {todos['date'].max().date()}")
    print(f"  CSV guardado en    : {csv_path}")
    print(f"  Gráficas en        : {FIGURES_DIR}")
    print("=" * 60)

    # Muestra resumen numérico por estado
    resumen = (todos.groupby("state")
               .agg(pico=("predicted", "max"),
                    promedio=("predicted", "mean"))
               .sort_values("pico", ascending=False)
               .head(10))
    print("\nTop 10 estados — pico pronosticado:")
    print(resumen.to_string(float_format=lambda x: f"{x:,.0f}"))


if __name__ == "__main__":
    main()
