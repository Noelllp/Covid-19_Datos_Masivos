"""
main.py — Punto de entrada del pipeline COVID-19 Big Data.

Orquesta los tres stages del pipeline en orden secuencial:
  Stage 1: Ingesta   → Bronze zone (CSV → Parquet raw)
  Stage 2: Transform → Silver zone (limpieza, deltas, features base)
  Stage 3: Analytics → Gold zone  (descriptivo, MapReduce, ML)

Uso:
  python main.py

  Con spark-submit (entorno cluster):
  spark-submit --master yarn main.py
"""

import logging
import sys
from pathlib import Path

# Añadir el directorio raíz al PYTHONPATH para resolución de imports relativos
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings        import SPARK_CONFIG, DATA_FILES, PROCESSED_DIR, FIGURES_DIR
from src.ingest             import build_spark_session, run_ingestion
from src.transform          import run_transformations
from src.analytics          import run_analytics
from src.mapreduce_jobs     import run_mapreduce_pipeline
from src.features           import run_feature_engineering, print_feature_catalog
from src.models             import run_modeling_strategy
from src.visualizations     import run_visualizations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    # Garantizar que las zonas de datos existen antes de iniciar Spark
    for zone in ["bronze", "silver", "gold"]:
        (Path(PROCESSED_DIR) / zone).mkdir(parents=True, exist_ok=True)

    spark = build_spark_session(SPARK_CONFIG)

    try:
        log.info("=" * 60)
        log.info("PIPELINE COVID-19 BIG DATA — INICIO")
        log.info("=" * 60)

        log.info("STAGE 1: INGESTA")
        raw_datasets = run_ingestion(spark, DATA_FILES)

        log.info("STAGE 2: TRANSFORMACIÓN")
        transformed = run_transformations(raw_datasets)

        # Liberar DataFrames raw de memoria — no se necesitan después de Silver
        for df in raw_datasets.values():
            try:
                df.unpersist()
            except Exception:
                pass

        log.info("STAGE 3A: INGENIERÍA DE FEATURES")
        features_df = run_feature_engineering(transformed["states"])
        print_feature_catalog()

        log.info("STAGE 3B: MODELADO PREDICTIVO")
        modeling_output = run_modeling_strategy(features_df)

        log.info("STAGE 3C: MAPREDUCÉ EXPLÍCITO")
        mr_results = run_mapreduce_pipeline(transformed["states"])

        log.info("STAGE 3D: ANALYTICS DESCRIPTIVO + ML (GOLD ZONE)")
        results = run_analytics(transformed)

        # Enriquecer resultados con la tabla comparativa de modelos
        results["ml"]["comparison_pdf"] = modeling_output.get("comparison_df")

        log.info("STAGE 4: VISUALIZACIONES")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        run_visualizations(
            analytics_results=results,
            transformed=transformed,
            figures_dir=FIGURES_DIR,
        )

        log.info("=" * 60)
        log.info("PIPELINE COMPLETADO EXITOSAMENTE")
        log.info(f"Métricas ML — Validation: {results['ml']['metrics'].get('validation')}")
        log.info(f"Métricas ML — Test:       {results['ml']['metrics'].get('test')}")
        log.info(f"Gold zone en:   {PROCESSED_DIR}/gold/")
        log.info(f"Figuras en:     {FIGURES_DIR}")
        log.info("=" * 60)

    except Exception as exc:
        log.critical(f"PIPELINE FALLIDO: {exc}", exc_info=True)
        sys.exit(1)

    finally:
        # Liberar caché y detener SparkSession limpiamente
        for df in transformed.values():
            try:
                df.unpersist()
            except Exception:
                pass
        spark.stop()
        log.info("SparkSession detenida.")


if __name__ == "__main__":
    main()
