"""
tools/metrics_tool.py — System Metrics Query Tool (Simulated)
==============================================================
CONCEPT: Metrics in Incident Investigation

  Metrics tell you the WHAT and WHEN of a problem:
    - CPU spike at 14:32 → when did the incident start?
    - Memory at 98% → what resource is exhausted?
    - Error rate at 67% → how bad is the impact?
    - Request latency at 8.2s → is the service degraded?

  Combined with logs (which tell you WHY), metrics help the
  Incident Agent build a complete picture of what happened.

CONCEPT: The Golden Signals (SRE)
  Google's SRE book defines "The Four Golden Signals" for monitoring:
    1. LATENCY  — How long requests take (p50, p95, p99)
    2. TRAFFIC  — How many requests per second
    3. ERRORS   — Error rate (% of failed requests)
    4. SATURATION — How "full" is the service (CPU, memory, queue)

  We simulate all four golden signals for each service.

CONCEPT: Time Series Data
  Real metrics are time series: (timestamp, value) pairs.
  Systems like Prometheus store metrics with 15s or 30s intervals.

  Our simulation generates realistic time series that show:
    - Gradual build-up → "memory leak" pattern
    - Sudden spike → "traffic surge" or "bug deployed" pattern
    - Sustained high → "not enough capacity" pattern
"""

import random
import math
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── Input Schemas ────────────────────────────────────────────────────────────

class MetricsQueryInput(BaseModel):
    """Input schema for querying system metrics."""
    service: str = Field(
        description=(
            "Service or resource to query metrics for. "
            "Examples: 'payment-service', 'auth-service', 'api-gateway', "
            "'postgres', 'redis', 'kubernetes-cluster', 'node-worker-1'."
        )
    )
    metric: str = Field(
        description=(
            "Metric to query. Available metrics:\n"
            "  'cpu'        — CPU usage percentage (0-100%)\n"
            "  'memory'     — Memory usage percentage (0-100%)\n"
            "  'error_rate' — Percentage of requests returning errors (0-100%)\n"
            "  'latency'    — Request latency in milliseconds (p95)\n"
            "  'rps'        — Requests per second (throughput)\n"
            "  'disk'       — Disk usage percentage (0-100%)\n"
            "  'connections'— Active connections count\n"
            "  'all'        — All available metrics (summary)"
        )
    )
    time_window: str = Field(
        default="1h",
        description=(
            "Time window for the metric query. "
            "Format: '<number><unit>' — '30m', '1h', '6h', '24h'. "
            "Returns a time series with data points across this window."
        )
    )


# ─── Realistic Metric Simulation ──────────────────────────────────────────────

def _get_base_metrics(service: str) -> dict:
    """
    Define realistic baseline and stressed metrics per service type.

    CONCEPT: Anomaly Pattern
      Normal baselines + stress patterns simulate what an incident looks like.
      The agent sees elevated metrics and can correlate with logs.
    """
    # Service-specific baseline and stressed values
    profiles = {
        "payment-service": {
            "cpu":         {"normal": 35, "stressed": 92, "unit": "%"},
            "memory":      {"normal": 65, "stressed": 98, "unit": "%"},
            "error_rate":  {"normal": 0.5, "stressed": 67, "unit": "%"},
            "latency":     {"normal": 250, "stressed": 8200, "unit": "ms"},
            "rps":         {"normal": 450, "stressed": 680, "unit": "req/s"},
            "connections": {"normal": 25, "stressed": 50, "unit": "connections"},
        },
        "auth-service": {
            "cpu":         {"normal": 20, "stressed": 75, "unit": "%"},
            "memory":      {"normal": 45, "stressed": 82, "unit": "%"},
            "error_rate":  {"normal": 0.2, "stressed": 34, "unit": "%"},
            "latency":     {"normal": 80, "stressed": 2400, "unit": "ms"},
            "rps":         {"normal": 320, "stressed": 420, "unit": "req/s"},
            "connections": {"normal": 40, "stressed": 98, "unit": "connections"},
        },
        "api-gateway": {
            "cpu":         {"normal": 15, "stressed": 60, "unit": "%"},
            "memory":      {"normal": 30, "stressed": 71, "unit": "%"},
            "error_rate":  {"normal": 0.1, "stressed": 45, "unit": "%"},
            "latency":     {"normal": 50, "stressed": 12000, "unit": "ms"},
            "rps":         {"normal": 1200, "stressed": 1850, "unit": "req/s"},
            "connections": {"normal": 150, "stressed": 500, "unit": "connections"},
        },
        "postgres": {
            "cpu":         {"normal": 25, "stressed": 88, "unit": "%"},
            "memory":      {"normal": 70, "stressed": 95, "unit": "%"},
            "error_rate":  {"normal": 0.0, "stressed": 15, "unit": "%"},
            "latency":     {"normal": 5, "stressed": 12000, "unit": "ms"},
            "rps":         {"normal": 800, "stressed": 1200, "unit": "queries/s"},
            "connections": {"normal": 45, "stressed": 103, "unit": "connections"},
            "disk":        {"normal": 58, "stressed": 97, "unit": "%"},
        },
        "redis": {
            "cpu":         {"normal": 8, "stressed": 45, "unit": "%"},
            "memory":      {"normal": 40, "stressed": 94, "unit": "%"},
            "error_rate":  {"normal": 0.0, "stressed": 28, "unit": "%"},
            "latency":     {"normal": 1, "stressed": 450, "unit": "ms"},
            "rps":         {"normal": 5000, "stressed": 3200, "unit": "ops/s"},
            "connections": {"normal": 80, "stressed": 500, "unit": "connections"},
        },
    }

    # Default profile for unknown services
    default = {
        "cpu":         {"normal": 30, "stressed": 85, "unit": "%"},
        "memory":      {"normal": 50, "stressed": 90, "unit": "%"},
        "error_rate":  {"normal": 0.5, "stressed": 40, "unit": "%"},
        "latency":     {"normal": 200, "stressed": 5000, "unit": "ms"},
        "rps":         {"normal": 100, "stressed": 150, "unit": "req/s"},
        "connections": {"normal": 20, "stressed": 80, "unit": "connections"},
        "disk":        {"normal": 45, "stressed": 88, "unit": "%"},
    }

    return profiles.get(service.lower(), default)


def _generate_time_series(
    metric_name: str,
    base_metrics: dict,
    minutes: int,
    data_points: int = 12,
) -> list[dict]:
    """
    Generate a realistic time series for a metric.

    CONCEPT: Memory Leak Pattern
      Memory metrics often show a GRADUAL INCREASE over time:
        Normal → 65% → 70% → 78% → 85% → 91% → 98% → OOM KILL

      We simulate this by having the metric "ramp up" toward the stressed
      value in the second half of the time window, simulating an incident
      that got progressively worse.
    """
    metric_info = base_metrics.get(metric_name)
    if not metric_info:
        return []

    normal = metric_info["normal"]
    stressed = metric_info["stressed"]
    unit = metric_info["unit"]

    now = datetime.now(timezone.utc)
    interval_minutes = max(1, minutes // data_points)
    series = []

    for i in range(data_points):
        # Progress through the time window (0.0 = oldest, 1.0 = now)
        progress = i / (data_points - 1) if data_points > 1 else 0

        # "Incident onset" happens around 60% through the window
        # This simulates a degradation that built up before we started looking
        if progress < 0.5:
            # Normal operation phase
            value = normal + random.gauss(0, normal * 0.08)  # ±8% noise
        elif progress < 0.7:
            # Gradual degradation phase
            blend = (progress - 0.5) / 0.2
            value = normal + (stressed - normal) * blend * 0.5 + random.gauss(0, normal * 0.1)
        else:
            # Stressed/incident phase
            blend = (progress - 0.7) / 0.3
            value = normal + (stressed - normal) * (0.5 + blend * 0.5) + random.gauss(0, stressed * 0.05)

        # Clamp to realistic range
        value = max(0, min(value, 100 if "%" in unit else stressed * 1.1))

        # Generate timestamp for this data point
        offset_minutes = (data_points - 1 - i) * interval_minutes
        timestamp = now - timedelta(minutes=offset_minutes)

        series.append({
            "timestamp": timestamp.strftime("%H:%M"),
            "value": round(value, 1),
            "unit": unit,
        })

    return series


def _parse_time_window(time_window: str) -> int:
    """Parse time window string to minutes."""
    try:
        unit = time_window[-1].lower()
        value = int(time_window[:-1])
        return value * {"m": 1, "h": 60, "d": 1440}.get(unit, 60)
    except (ValueError, IndexError):
        return 60


def _format_time_series(metric_name: str, series: list[dict], service: str) -> str:
    """Format a time series as an ASCII sparkline + stats."""
    if not series:
        return f"No data for {metric_name}"

    values = [p["value"] for p in series]
    unit = series[0]["unit"]
    current = values[-1]
    peak = max(values)
    avg = sum(values) / len(values)

    # Determine status
    if metric_name in ("cpu", "memory", "disk", "error_rate"):
        if current > 85:
            status = "🔴 CRITICAL"
        elif current > 70:
            status = "🟡 WARNING"
        else:
            status = "🟢 NORMAL"
    else:
        # For latency, higher is worse; for rps, we check relative change
        baseline = values[0]  # First data point as baseline
        change_pct = ((current - baseline) / max(baseline, 1)) * 100
        if change_pct > 100:
            status = "🔴 CRITICAL (>2x baseline)"
        elif change_pct > 50:
            status = "🟡 WARNING (>1.5x baseline)"
        else:
            status = "🟢 NORMAL"

    # Simple sparkline using block chars
    normalized = [(v - min(values)) / max(max(values) - min(values), 1) for v in values]
    blocks = " ▁▂▃▄▅▆▇█"
    sparkline = "".join(blocks[int(n * 8)] for n in normalized)

    return (
        f"  {metric_name.upper():12} {status}\n"
        f"  Current: {current:.1f}{unit} | Peak: {peak:.1f}{unit} | Avg: {avg:.1f}{unit}\n"
        f"  Trend:   [{sparkline}]  ({series[0]['timestamp']} → {series[-1]['timestamp']})"
    )


# ─── Tool Factory ─────────────────────────────────────────────────────────────

def build_metrics_tool() -> StructuredTool:
    """
    Returns the metrics query tool (stateless — no DB needed).

    Returns:
        A configured StructuredTool for metrics querying.
    """

    def query_metrics(
        service: str,
        metric: str,
        time_window: str = "1h",
    ) -> str:
        """
        Query system metrics for a service over a time window.

        Use this during incident investigation to:
        - Check if CPU/memory is the bottleneck (saturation)
        - See error rates spike during the incident
        - Measure request latency degradation
        - Find when the issue started (time series trend)
        - Check all metrics at once with metric='all'

        ⚠️ Note: This tool uses simulated metrics for learning purposes.
        In production, this would query Prometheus, Datadog, CloudWatch, etc.

        Returns a formatted metric report with current values, trends, and status.
        """
        logger.info("metrics_tool_invoked", service=service, metric=metric, time_window=time_window)

        minutes = _parse_time_window(time_window)
        base_metrics = _get_base_metrics(service)

        # Which metrics to show
        if metric.lower() == "all":
            metrics_to_show = list(base_metrics.keys())
        elif metric.lower() in base_metrics:
            metrics_to_show = [metric.lower()]
        else:
            available = ", ".join(base_metrics.keys())
            return f"Unknown metric '{metric}'. Available for {service}: {available}"

        lines = [
            f"📊 Metrics for '{service}' (last {time_window})\n"
            f"[Note: Simulated data for learning — represents realistic patterns]\n"
        ]

        for metric_name in metrics_to_show:
            series = _generate_time_series(metric_name, base_metrics, minutes)
            if series:
                lines.append(_format_time_series(metric_name, series, service))
                lines.append("")

        return "\n".join(lines)

    return StructuredTool.from_function(
        func=query_metrics,
        name="query_metrics",
        description=(
            "Query system metrics (CPU, memory, error rate, latency, etc.) for a service. "
            "Use during incident investigation to identify resource bottlenecks, "
            "measure service degradation, and find when the issue started. "
            "Use metric='all' to get a full dashboard view. "
            "Key services: 'payment-service', 'auth-service', 'api-gateway', 'postgres', 'redis'."
        ),
        args_schema=MetricsQueryInput,
    )
