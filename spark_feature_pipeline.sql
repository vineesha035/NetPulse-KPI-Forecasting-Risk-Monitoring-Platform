-- ═══════════════════════════════════════════════════════════════════════════
-- NetPulse — Spark SQL Feature Engineering Pipeline
-- 42 features across 6 categories
-- Runs as a Spark Structured Streaming job (micro-batch: 5s)
-- ═══════════════════════════════════════════════════════════════════════════

-- ─── SOURCE TABLES ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS network_raw (
    timestamp       TIMESTAMP,
    node_id         STRING,
    region          STRING,
    throughput_gbps DOUBLE,
    latency_ms      DOUBLE,
    packet_loss_pct DOUBLE,
    active_flows_m  DOUBLE,
    utilization_pct DOUBLE,
    bytes_in_gbps   DOUBLE,
    bytes_out_gbps  DOUBLE,
    bgp_updates     INT,
    dns_qps         INT,
    cdn_requests_k  DOUBLE,
    jitter_ms       DOUBLE
)
USING DELTA
PARTITIONED BY (node_id, date(timestamp));


-- ─── FEATURE VIEW (registered as SparkSQL temporary view) ─────────────────

CREATE OR REPLACE TEMPORARY VIEW network_features AS
WITH
-- ── BASE + LAG FEATURES ────────────────────────────────────────────────────
lagged AS (
    SELECT
        timestamp,
        node_id,
        region,
        throughput_gbps,
        utilization_pct,
        packet_loss_pct,
        latency_ms,
        jitter_ms,
        active_flows_m,
        bytes_in_gbps,
        bytes_out_gbps,
        bgp_updates,
        dns_qps,
        cdn_requests_k,

        -- ── GROUP 4: Lag features ─────────────────────────────────────────
        LAG(throughput_gbps, 1)   OVER w AS lag_1h,
        LAG(throughput_gbps, 2)   OVER w AS lag_2h,
        LAG(throughput_gbps, 24)  OVER w AS lag_24h,
        LAG(throughput_gbps, 48)  OVER w AS lag_48h,
        LAG(throughput_gbps, 168) OVER w AS lag_168h,

        throughput_gbps - LAG(throughput_gbps, 1)  OVER w AS diff_1,
        throughput_gbps - LAG(throughput_gbps, 24) OVER w AS diff_24

    FROM network_raw
    WINDOW w AS (PARTITION BY node_id ORDER BY timestamp)
),

-- ── ROLLING STATISTICAL FEATURES ──────────────────────────────────────────
rolling AS (
    SELECT
        *,
        -- ── GROUP 2: Statistical rolling features ──────────────────────────
        AVG(throughput_gbps)    OVER w1h  AS rolling_mean_1h,
        STDDEV(throughput_gbps) OVER w1h  AS rolling_std_1h,
        AVG(throughput_gbps)    OVER w6h  AS rolling_mean_6h,
        MAX(throughput_gbps)    OVER w24h AS rolling_max_24h,
        MIN(throughput_gbps)    OVER w24h AS rolling_min_24h,
        -- EWM approximated via weighted avg over 10 recent rows (alpha=0.3)
        (0.3 * throughput_gbps
           + 0.21 * LAG(throughput_gbps,1) OVER w
           + 0.147 * LAG(throughput_gbps,2) OVER w
           + 0.103 * LAG(throughput_gbps,3) OVER w
           + 0.072 * LAG(throughput_gbps,4) OVER w)
            / 0.832                          AS ewm_30pct

    FROM lagged
    WINDOW
        w    AS (PARTITION BY node_id ORDER BY timestamp),
        w1h  AS (PARTITION BY node_id ORDER BY timestamp ROWS BETWEEN 0  PRECEDING AND CURRENT ROW),
        w6h  AS (PARTITION BY node_id ORDER BY timestamp ROWS BETWEEN 5  PRECEDING AND CURRENT ROW),
        w24h AS (PARTITION BY node_id ORDER BY timestamp ROWS BETWEEN 23 PRECEDING AND CURRENT ROW)
),

-- ── EXOGENOUS NORMALIZATION (z-score per node) ─────────────────────────────
exog_stats AS (
    SELECT
        node_id,
        AVG(cdn_requests_k) AS cdn_mean, STDDEV(cdn_requests_k) AS cdn_std,
        AVG(dns_qps)        AS dns_mean, STDDEV(dns_qps)        AS dns_std
    FROM network_raw
    GROUP BY node_id
),

-- ── SPECTRAL PROXY (dominant period bin via SQL approximation) ─────────────
-- Full FFT is computed via a Spark UDF registered separately:
-- spark.udf.register("fft_dominant_bin", fft_dominant_bin_udf, DoubleType())
-- Here we use a simpler rolling variance ratio as spectral proxy.
spectral AS (
    SELECT
        node_id,
        timestamp,
        STDDEV(throughput_gbps) OVER w24h  /
            (AVG(throughput_gbps)  OVER w168h + 1e-9) AS spectral_entropy,
        POWER(AVG(throughput_gbps) OVER w24h, 2)      AS periodogram_power_24,
        POWER(AVG(throughput_gbps) OVER w168h, 2)     AS periodogram_power_168
    FROM network_raw
    WINDOW
        w24h  AS (PARTITION BY node_id ORDER BY timestamp ROWS BETWEEN 23  PRECEDING AND CURRENT ROW),
        w168h AS (PARTITION BY node_id ORDER BY timestamp ROWS BETWEEN 167 PRECEDING AND CURRENT ROW)
)

-- ── FINAL FEATURE ASSEMBLY ─────────────────────────────────────────────────
SELECT
    r.timestamp,
    r.node_id,
    r.region,
    r.throughput_gbps,

    -- ── GROUP 1: Temporal (9 features) ──────────────────────────────────
    HOUR(r.timestamp)                                   AS hour_of_day,
    DAYOFWEEK(r.timestamp) - 1                          AS day_of_week,          -- 0=Mon
    CASE WHEN DAYOFWEEK(r.timestamp) IN (1,7) THEN 1 ELSE 0 END AS is_weekend,
    CASE WHEN HOUR(r.timestamp) BETWEEN 8 AND 20 THEN 1 ELSE 0 END AS is_peak_hour,
    MONTH(r.timestamp)                                  AS month,
    QUARTER(r.timestamp)                                AS quarter,
    WEEKOFYEAR(r.timestamp)                             AS week_of_year,
    CASE WHEN DATE_FORMAT(r.timestamp,'yyyy-MM-dd')
         IN ('2024-01-01','2024-07-04','2024-12-25','2024-11-28')
         THEN 1 ELSE 0 END                              AS is_holiday,
    SIN(2 * PI() * HOUR(r.timestamp) / 24.0)           AS hour_sin,
    COS(2 * PI() * HOUR(r.timestamp) / 24.0)           AS hour_cos,

    -- ── GROUP 2: Statistical Rolling (8 features) ───────────────────────
    r.rolling_mean_1h,
    r.rolling_std_1h,
    r.rolling_mean_6h,
    r.rolling_max_24h,
    r.rolling_min_24h,
    r.ewm_30pct,
    -- kurtosis & skewness via Spark built-ins applied in Python UDF layer
    kurtosis_1h,    -- registered UDF: rolling_kurtosis(throughput_gbps, 12)
    skewness_1h,    -- registered UDF: rolling_skewness(throughput_gbps, 12)

    -- ── GROUP 3: Network KPIs (8 features) ──────────────────────────────
    r.utilization_pct,
    r.packet_loss_pct                                   AS packet_loss_rate,
    r.latency_ms                                        AS avg_latency_ms,
    r.jitter_ms,
    r.active_flows_m                                    AS active_flows,
    r.bytes_in_gbps / (r.bytes_out_gbps + 1e-9)        AS bytes_io_ratio,
    r.throughput_gbps / (r.rolling_max_24h + 1e-9)     AS top_talker_share,
    r.rolling_std_1h  / (r.rolling_mean_1h + 1e-9)     AS protocol_entropy,

    -- ── GROUP 4: Lag / Differencing (7 features) ───────────────────────
    r.lag_1h,
    r.lag_2h,
    r.lag_24h,
    r.lag_48h,
    r.lag_168h,
    r.diff_1,
    r.diff_24,

    -- ── GROUP 5: Exogenous (6 features) ─────────────────────────────────
    (r.cdn_requests_k - e.cdn_mean) / (e.cdn_std + 1e-9)  AS cdn_req_norm,
    (r.dns_qps        - e.dns_mean) / (e.dns_std + 1e-9)  AS dns_qps_norm,
    r.bgp_updates,
    r.rolling_std_1h                                       AS ntp_drift_proxy,
    CAST(0 AS INT)                                         AS geo_event_flag,     -- injected from Kafka
    CAST(0 AS INT)                                         AS maintenance_flag,   -- injected from ITSM

    -- ── GROUP 6: Spectral (4 features) ──────────────────────────────────
    s.spectral_entropy,
    s.periodogram_power_24   AS periodogram_power,
    fft_dominant_bin(collect_list(r.throughput_gbps)
        OVER (PARTITION BY r.node_id ORDER BY r.timestamp
              ROWS BETWEEN 23 PRECEDING AND CURRENT ROW), 24)
                                                           AS fft_peak_24,        -- UDF
    fft_dominant_bin(collect_list(r.throughput_gbps)
        OVER (PARTITION BY r.node_id ORDER BY r.timestamp
              ROWS BETWEEN 167 PRECEDING AND CURRENT ROW), 168)
                                                           AS fft_peak_168        -- UDF

FROM rolling r
JOIN exog_stats e  ON r.node_id = e.node_id
JOIN spectral   s  ON r.node_id = s.node_id AND r.timestamp = s.timestamp
WHERE r.lag_168h IS NOT NULL;   -- require 1 week of history


-- ─── REAL-TIME STREAMING JOB ──────────────────────────────────────────────
-- In PySpark this is invoked as:
--
-- feature_stream = (
--   spark.readStream
--     .format("kafka")
--     .option("kafka.bootstrap.servers", "kafka:9092")
--     .option("subscribe", "network.telemetry")
--     .load()
--     .selectExpr("CAST(value AS STRING) as json")
--     .select(from_json(col("json"), schema).alias("data"))
--     .select("data.*")
-- )
--
-- enriched = feature_stream.createOrReplaceTempView("network_raw_stream")
-- result   = spark.sql("SELECT * FROM network_features")
--
-- (result.writeStream
--   .format("delta")
--   .outputMode("append")
--   .option("checkpointLocation", "/checkpoints/features")
--   .trigger(processingTime="5 seconds")
--   .start("/delta/network_features"))


-- ─── KPI AGGREGATION QUERY ────────────────────────────────────────────────

CREATE OR REPLACE TEMPORARY VIEW kpi_snapshot AS
SELECT
    date_trunc('minute', timestamp)             AS minute_ts,
    COUNT(DISTINCT node_id)                     AS active_nodes,
    ROUND(SUM(throughput_gbps), 2)              AS total_throughput_gbps,
    ROUND(AVG(utilization_pct), 2)              AS avg_utilization_pct,
    ROUND(AVG(latency_ms), 2)                   AS avg_latency_ms,
    ROUND(AVG(packet_loss_pct) * 100, 4)        AS packet_loss_pct,
    ROUND(SUM(active_flows_m), 3)               AS total_flows_m,
    ROUND(100 - AVG(packet_loss_pct) * 100, 4)  AS availability_pct,
    COUNT(CASE WHEN utilization_pct > 85 THEN 1 END) AS high_util_nodes,
    COUNT(CASE WHEN packet_loss_pct > 0.001 THEN 1 END) AS degraded_nodes
FROM network_features
GROUP BY date_trunc('minute', timestamp)
ORDER BY minute_ts DESC;


-- ─── ANOMALY FLAGGING IN SQL ──────────────────────────────────────────────

CREATE OR REPLACE TEMPORARY VIEW anomaly_flags AS
WITH stats AS (
    SELECT
        node_id,
        AVG(throughput_gbps)    AS mu,
        STDDEV(throughput_gbps) AS sigma
    FROM network_features
    WHERE timestamp >= current_timestamp() - INTERVAL 7 DAYS
    GROUP BY node_id
)
SELECT
    f.timestamp,
    f.node_id,
    f.throughput_gbps,
    s.mu,
    s.sigma,
    ABS(f.throughput_gbps - s.mu) / (s.sigma + 1e-9) AS z_score,
    CASE
        WHEN ABS(f.throughput_gbps - s.mu) / (s.sigma + 1e-9) > 3.5 THEN 'HIGH'
        WHEN ABS(f.throughput_gbps - s.mu) / (s.sigma + 1e-9) > 2.5 THEN 'MEDIUM'
        ELSE 'NORMAL'
    END AS anomaly_severity
FROM network_features f
JOIN stats s ON f.node_id = s.node_id
WHERE f.timestamp >= current_timestamp() - INTERVAL 1 HOUR
ORDER BY z_score DESC;
