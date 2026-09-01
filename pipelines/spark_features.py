"""
Spark SQL Feature Engineering Pipeline
=======================================
Production PySpark implementation of the 47-feature engineering pipeline.
Designed for Apache Spark 3.x on EMR / Databricks clusters.
Processes ~500M rows/day across all enterprise sites.

Usage:
    spark-submit feature_pipeline.py \
        --input s3://network-data/raw/ \
        --output s3://network-data/features/ \
        --date 2025-03-25
"""

# In production this runs on PySpark. For portability, we include
# both the PySpark implementation and the equivalent Spark SQL strings.

SPARK_SQL_FEATURES = {

    # ─── Temporal Features ──────────────────────────────────────────────────
    "temporal_core": """
        SELECT
            timestamp,
            site,
            total_gbps,
            utilization_pct,
            -- Calendar decomposition
            HOUR(timestamp)                          AS hour_of_day,
            DAYOFWEEK(timestamp) - 1                 AS day_of_week,    -- 0=Mon
            DAYOFMONTH(timestamp)                    AS day_of_month,
            WEEKOFYEAR(timestamp)                    AS week_of_year,
            MONTH(timestamp)                         AS month,
            QUARTER(timestamp)                       AS quarter,
            -- Binary flags
            CASE WHEN DAYOFWEEK(timestamp) IN (1,7)
                 THEN 1 ELSE 0 END                   AS is_weekend,
            CASE WHEN HOUR(timestamp) BETWEEN 9 AND 17
                  AND DAYOFWEEK(timestamp) NOT IN (1,7)
                 THEN 1 ELSE 0 END                   AS is_business_hour,
            CASE WHEN DAYOFMONTH(timestamp) >= 28
                 THEN 1 ELSE 0 END                   AS is_month_end,
            CASE WHEN MONTH(timestamp) IN (3,6,9,12)
                  AND DAYOFMONTH(timestamp) >= 28
                 THEN 1 ELSE 0 END                   AS is_quarter_end,
            -- Cyclical encodings (prevent ordinal discontinuity)
            SIN(2 * PI() * HOUR(timestamp) / 24.0)  AS hour_sin,
            COS(2 * PI() * HOUR(timestamp) / 24.0)  AS hour_cos,
            SIN(2 * PI() * (DAYOFWEEK(timestamp)-1) / 7.0) AS dow_sin,
            COS(2 * PI() * (DAYOFWEEK(timestamp)-1) / 7.0) AS dow_cos,
            SIN(2 * PI() * DAYOFYEAR(timestamp) / 365.0)   AS day_sin,
            COS(2 * PI() * DAYOFYEAR(timestamp) / 365.0)   AS day_cos
        FROM network_traffic_raw
    """,

    # ─── Lag Features (window functions) ────────────────────────────────────
    "lag_features": """
        SELECT
            *,
            LAG(total_gbps, 1)   OVER w AS lag_1h,
            LAG(total_gbps, 2)   OVER w AS lag_2h,
            LAG(total_gbps, 3)   OVER w AS lag_3h,
            LAG(total_gbps, 6)   OVER w AS lag_6h,
            LAG(total_gbps, 12)  OVER w AS lag_12h,
            LAG(total_gbps, 24)  OVER w AS lag_24h,
            LAG(total_gbps, 48)  OVER w AS lag_48h,
            LAG(total_gbps, 168) OVER w AS lag_168h,     -- 1 week same hour
            -- Previous period ratios
            SAFE_DIVIDE(total_gbps, LAG(total_gbps, 24) OVER w) AS vs_same_hour_yesterday,
            SAFE_DIVIDE(total_gbps, LAG(total_gbps, 168) OVER w) AS vs_same_hour_last_week
        FROM temporal_enriched
        WINDOW w AS (PARTITION BY site ORDER BY timestamp ASC)
    """,

    # ─── Rolling Statistics ──────────────────────────────────────────────────
    "rolling_stats": """
        SELECT
            *,
            -- 3-hour window
            AVG(total_gbps) OVER w3  AS roll_mean_3h,
            STDDEV(total_gbps) OVER w3  AS roll_std_3h,
            -- 6-hour window
            AVG(total_gbps) OVER w6  AS roll_mean_6h,
            STDDEV(total_gbps) OVER w6  AS roll_std_6h,
            -- 24-hour window
            AVG(total_gbps) OVER w24 AS roll_mean_24h,
            STDDEV(total_gbps) OVER w24 AS roll_std_24h,
            MAX(total_gbps) OVER w24 AS roll_max_24h,
            MIN(total_gbps) OVER w24 AS roll_min_24h,
            -- 168-hour (weekly) window
            AVG(total_gbps) OVER w168 AS roll_mean_168h
        FROM lag_enriched
        WINDOW
            w3   AS (PARTITION BY site ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
            w6   AS (PARTITION BY site ORDER BY timestamp ROWS BETWEEN 5 PRECEDING AND CURRENT ROW),
            w24  AS (PARTITION BY site ORDER BY timestamp ROWS BETWEEN 23 PRECEDING AND CURRENT ROW),
            w168 AS (PARTITION BY site ORDER BY timestamp ROWS BETWEEN 167 PRECEDING AND CURRENT ROW)
    """,

    # ─── Derived KPIs ────────────────────────────────────────────────────────
    "derived_kpis": """
        SELECT
            *,
            -- Capacity and headroom
            100.0 - utilization_pct                         AS bandwidth_headroom_pct,
            total_gbps / capacity_gbps                      AS traffic_intensity_index,
            -- Composite quality score (0-100, higher is better)
            GREATEST(0, LEAST(100,
                100
                - (latency_ms / 5.0)
                - (packet_loss_pct * 20.0)
                - (jitter_ms / 2.0)
            ))                                              AS quality_score,
            -- Congestion risk tier
            CASE
                WHEN utilization_pct > 80 THEN 'HIGH'
                WHEN utilization_pct > 65 THEN 'MEDIUM'
                ELSE 'LOW'
            END                                             AS congestion_risk,
            -- Efficiency: throughput per unit latency
            total_gbps / GREATEST(latency_ms, 1.0)         AS efficiency_ratio,
            -- Growth rate: vs 24h prior
            SAFE_DIVIDE(total_gbps - lag_24h, lag_24h)     AS traffic_growth_rate_24h,
            -- Peak-to-average ratio
            SAFE_DIVIDE(roll_max_24h, NULLIF(roll_mean_24h, 0)) AS peak_to_avg_ratio
        FROM rolling_enriched
    """,

    # ─── Cross-Site Signals ───────────────────────────────────────────────────
    "cross_site_signals": """
        SELECT
            t.*,
            -- Rank by utilization within same timestamp bucket
            RANK() OVER (
                PARTITION BY DATE_TRUNC('hour', timestamp)
                ORDER BY utilization_pct DESC
            )                                               AS site_rank_current_hour,
            -- Relative to global average at same hour
            utilization_pct / AVG(utilization_pct) OVER (
                PARTITION BY DATE_TRUNC('hour', timestamp)
            )                                               AS site_relative_to_avg,
            -- Cross-site stddev (diversity index — higher = more imbalanced)
            STDDEV(utilization_pct) OVER (
                PARTITION BY DATE_TRUNC('hour', timestamp)
            )                                               AS cross_site_stddev
        FROM derived_enriched t
    """,

    # ─── Anomaly Indicators ───────────────────────────────────────────────────
    "anomaly_indicators": """
        SELECT
            *,
            -- Rolling z-score (24h window)
            SAFE_DIVIDE(
                total_gbps - roll_mean_24h,
                NULLIF(roll_std_24h, 0)
            )                                               AS zscore_24h,
            -- Short z-score (3h window)
            SAFE_DIVIDE(
                total_gbps - roll_mean_3h,
                NULLIF(roll_std_3h, 0)
            )                                               AS zscore_1h,
            -- IQR-based outlier flag
            CASE WHEN ABS(total_gbps - roll_mean_24h)
                      > 1.5 * (
                            PERCENTILE(total_gbps, 0.75) OVER w24
                          - PERCENTILE(total_gbps, 0.25) OVER w24
                         )
                 THEN 1 ELSE 0
            END                                             AS iqr_flag_24h,
            -- Sudden spike: z > 3 in short window
            CASE WHEN ABS(
                SAFE_DIVIDE(total_gbps - roll_mean_3h, NULLIF(roll_std_3h,0))
            ) > 3 THEN 1 ELSE 0 END                        AS sudden_spike_flag,
            -- Sustained high: min of last 6h > 1.5x rolling average
            CASE WHEN MIN(total_gbps) OVER w6
                      > 1.5 * roll_mean_24h
                 THEN 1 ELSE 0
            END                                             AS sustained_high_flag
        FROM cross_site_enriched
        WINDOW
            w24 AS (PARTITION BY site ORDER BY timestamp ROWS BETWEEN 23 PRECEDING AND CURRENT ROW),
            w6  AS (PARTITION BY site ORDER BY timestamp ROWS BETWEEN 5  PRECEDING AND CURRENT ROW)
    """,
}

# ─── PySpark Pipeline Class ──────────────────────────────────────────────────

PYSPARK_PIPELINE = '''
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import argparse

class SparkFeaturePipeline:
    """
    Production Spark feature engineering pipeline.
    Processes ~500M rows/day across 5 enterprise network sites.
    Outputs Parquet features to S3 for SARIMAX model ingestion.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.site_window = Window.partitionBy("site").orderBy("timestamp")
        self.hour_window = Window.partitionBy(F.date_trunc("hour", F.col("timestamp")))

    def build_window(self, rows_back: int):
        return (Window.partitionBy("site")
                      .orderBy("timestamp")
                      .rowsBetween(-rows_back, 0))

    def add_temporal_features(self, df):
        return df.withColumns({
            "hour_of_day":       F.hour("timestamp"),
            "day_of_week":       F.dayofweek("timestamp") - F.lit(1),
            "day_of_month":      F.dayofmonth("timestamp"),
            "week_of_year":      F.weekofyear("timestamp"),
            "month":             F.month("timestamp"),
            "quarter":           F.quarter("timestamp"),
            "is_weekend":        ((F.dayofweek("timestamp")).isin(1, 7)).cast("int"),
            "is_business_hour":  (
                F.hour("timestamp").between(9, 17) &
                (~F.dayofweek("timestamp").isin(1, 7))
            ).cast("int"),
            "is_month_end":      (F.dayofmonth("timestamp") >= 28).cast("int"),
            "hour_sin":          F.sin(2 * 3.14159 * F.hour("timestamp") / 24.0),
            "hour_cos":          F.cos(2 * 3.14159 * F.hour("timestamp") / 24.0),
            "dow_sin":           F.sin(2 * 3.14159 * (F.dayofweek("timestamp")-1) / 7.0),
            "dow_cos":           F.cos(2 * 3.14159 * (F.dayofweek("timestamp")-1) / 7.0),
            "day_sin":           F.sin(2 * 3.14159 * F.dayofyear("timestamp") / 365.0),
            "day_cos":           F.cos(2 * 3.14159 * F.dayofyear("timestamp") / 365.0),
        })

    def add_lag_features(self, df):
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df = df.withColumn(f"lag_{lag}h", F.lag("total_gbps", lag).over(self.site_window))
        return df

    def add_rolling_stats(self, df):
        for window_size, suffix in [(3, "3h"), (6, "6h"), (24, "24h"), (168, "168h")]:
            w = self.build_window(window_size - 1)
            df = df.withColumn(f"roll_mean_{suffix}", F.avg("total_gbps").over(w))
            if suffix != "168h":
                df = df.withColumn(f"roll_std_{suffix}", F.stddev("total_gbps").over(w))
            if suffix == "24h":
                df = df.withColumn("roll_max_24h", F.max("total_gbps").over(w))
                df = df.withColumn("roll_min_24h", F.min("total_gbps").over(w))
        return df

    def add_derived_kpis(self, df):
        return df.withColumns({
            "bandwidth_headroom_pct": F.lit(100.0) - F.col("utilization_pct"),
            "traffic_intensity_index": F.col("total_gbps") / F.col("capacity_gbps"),
            "quality_score": F.greatest(
                F.lit(0),
                F.least(F.lit(100),
                    F.lit(100)
                    - (F.col("latency_ms") / 5.0)
                    - (F.col("packet_loss_pct") * 20.0)
                    - (F.col("jitter_ms") / 2.0)
                )
            ),
            "congestion_risk": F.when(F.col("utilization_pct") > 80, "HIGH")
                                .when(F.col("utilization_pct") > 65, "MEDIUM")
                                .otherwise("LOW"),
            "efficiency_ratio": F.col("total_gbps") / F.greatest(F.col("latency_ms"), F.lit(1.0)),
            "traffic_growth_rate": (
                (F.col("total_gbps") - F.col("lag_24h")) /
                F.when(F.col("lag_24h") != 0, F.col("lag_24h")).otherwise(None)
            ),
            "peak_to_avg_ratio": (
                F.col("roll_max_24h") /
                F.when(F.col("roll_mean_24h") != 0, F.col("roll_mean_24h")).otherwise(None)
            ),
        })

    def add_cross_site_signals(self, df):
        rank_window = Window.partitionBy(F.date_trunc("hour", F.col("timestamp"))).orderBy(F.col("utilization_pct").desc())
        return df.withColumns({
            "site_rank_current_hour": F.rank().over(rank_window),
            "site_relative_to_avg": F.col("utilization_pct") / F.avg("utilization_pct").over(self.hour_window),
            "cross_site_stddev": F.stddev("utilization_pct").over(self.hour_window),
        })

    def add_anomaly_indicators(self, df):
        w24 = self.build_window(23)
        w6 = self.build_window(5)
        df = df.withColumns({
            "zscore_24h": (
                (F.col("total_gbps") - F.col("roll_mean_24h")) /
                F.when(F.col("roll_std_24h") != 0, F.col("roll_std_24h")).otherwise(None)
            ),
            "zscore_1h": (
                (F.col("total_gbps") - F.col("roll_mean_3h")) /
                F.when(F.col("roll_std_3h") != 0, F.col("roll_std_3h")).otherwise(None)
            ),
            "sudden_spike_flag": (
                F.abs(
                    (F.col("total_gbps") - F.col("roll_mean_3h")) /
                    F.when(F.col("roll_std_3h") != 0, F.col("roll_std_3h")).otherwise(F.lit(999))
                ) > 3
            ).cast("int"),
        })
        return df

    def run(self, input_path: str, output_path: str, date: str):
        print(f"[SparkFeaturePipeline] Reading from {input_path} for date={date}")
        df = (self.spark.read.parquet(input_path)
                  .filter(F.col("date") == date)
                  .repartition(200, "site"))

        print("[SparkFeaturePipeline] Applying temporal features...")
        df = self.add_temporal_features(df)
        print("[SparkFeaturePipeline] Applying lag features...")
        df = self.add_lag_features(df)
        print("[SparkFeaturePipeline] Applying rolling statistics...")
        df = self.add_rolling_stats(df)
        print("[SparkFeaturePipeline] Applying derived KPIs...")
        df = self.add_derived_kpis(df)
        print("[SparkFeaturePipeline] Applying cross-site signals...")
        df = self.add_cross_site_signals(df)
        print("[SparkFeaturePipeline] Applying anomaly indicators...")
        df = self.add_anomaly_indicators(df)

        feature_count = len(df.columns)
        print(f"[SparkFeaturePipeline] {feature_count} total columns, writing to {output_path}")
        (df.write
           .mode("overwrite")
           .partitionBy("site", "date")
           .parquet(output_path))
        print(f"[SparkFeaturePipeline] Done. Processed {df.count():,} rows.")
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    spark = (SparkSession.builder
             .appName("NetworkTrafficFeaturePipeline")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.shuffle.partitions", "400")
             .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
             .getOrCreate())

    pipeline = SparkFeaturePipeline(spark)
    pipeline.run(args.input, args.output, args.date)
    spark.stop()
'''


def print_sql_summary():
    print("Spark SQL Feature Pipeline — Query Summary")
    print("=" * 55)
    for stage, sql in SPARK_SQL_FEATURES.items():
        lines = [l.strip() for l in sql.strip().split('\n') if l.strip()]
        feature_lines = [l for l in lines if ' AS ' in l]
        print(f"\nStage: {stage}")
        print(f"  Features in stage: {len(feature_lines)}")
        print(f"  SQL lines: {len(lines)}")


if __name__ == "__main__":
    print_sql_summary()
